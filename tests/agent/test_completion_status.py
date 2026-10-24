"""Local turns need an explicit end signal; older cached prompts adopt it by a bounded nudge."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.completion_status import (
    COMPLETION_STATUS_MARKER,
    COMPLETION_STATUS_NUDGE,
    COMPLETION_STATUS_SIGNATURE,
    CompletionStatusStreamFilter,
    completion_status_required,
    extract_completion_status,
    is_local_ai_server_gateway_url,
)
from agent.stream_delivery import StreamDeliveryMixin


def test_completion_status_is_required_by_protocol_or_legacy_local_route():
    assert is_local_ai_server_gateway_url("http://127.0.0.1:8080/v1")
    assert is_local_ai_server_gateway_url("http://[::1]:8080/v1")
    assert not is_local_ai_server_gateway_url("http://127.0.0.1:8787/v1")
    assert not is_local_ai_server_gateway_url("https://api.example/v1")
    assert completion_status_required(f"{COMPLETION_STATUS_SIGNATURE}\n")
    assert completion_status_required("legacy prompt", base_url="http://localhost:8080/v1")
    assert not completion_status_required("legacy prompt", base_url="https://api.example/v1")


def test_new_local_system_prompt_carries_the_completion_protocol():
    agent = _local_agent()
    prompt = agent._build_system_prompt()
    assert COMPLETION_STATUS_SIGNATURE in prompt
    assert COMPLETION_STATUS_MARKER in prompt


def test_extract_completion_status_requires_the_exact_terminal_line():
    body, marked = extract_completion_status(f"Finished the task.\n{COMPLETION_STATUS_MARKER}\n")
    assert body == "Finished the task."
    assert marked is True
    assert extract_completion_status(f"The marker is {COMPLETION_STATUS_MARKER}.") == (
        f"The marker is {COMPLETION_STATUS_MARKER}.", False,
    )


def test_stream_filter_hides_marker_split_across_chunks_but_restores_nonterminal_text():
    filt = CompletionStatusStreamFilter()
    response = "Answer complete.\n" + COMPLETION_STATUS_MARKER
    pieces = [response[:11], response[11:24], response[24:]]
    live = "".join(filt.feed(part) for part in pieces)
    live += filt.finish(marker_present=True)
    assert live == "Answer complete."
    assert COMPLETION_STATUS_MARKER not in live

    retry = CompletionStatusStreamFilter()
    nonterminal = f"{COMPLETION_STATUS_MARKER}\nThis line continues the reply."
    live_retry = retry.feed(nonterminal[:20]) + retry.feed(nonterminal[20:])
    live_retry += retry.finish(marker_present=False)
    assert live_retry == nonterminal


class _StreamHarness(StreamDeliveryMixin):
    def __init__(self):
        self.visible = []
        self.base_url = "http://127.0.0.1:8080/v1"
        self._cached_system_prompt = "legacy prompt"
        self.stream_delta_callback = self.visible.append
        self._stream_callback = None
        self._stream_writer_superseded = lambda: False
        self._strip_think_blocks = lambda text: text
        self._enqueue_stream_hook = lambda *args, **kwargs: None


def test_stream_delivery_filters_terminal_marker_before_the_ui_callback():
    stream = _StreamHarness()
    stream._reset_stream_delivery_tracking()
    content = f"Readback verified.\n{COMPLETION_STATUS_MARKER}"
    for piece in (content[:19], content[19:30], content[30:]):
        stream._fire_stream_delta(piece)
    stream._finish_completion_status_stream(marker_present=True)
    assert "".join(stream.visible) == "Readback verified."


def _response(content):
    return SimpleNamespace(
        id="chatcmpl-status", choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop", index=0,
        )], model="local-model", usage=None,
    )


def _local_agent(base_url="http://127.0.0.1:8080/v1"):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url=base_url, provider="custom", model="local-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        )
    agent.client = MagicMock()
    # Simulate a restored session whose saved system prompt predates the protocol.
    agent._cached_system_prompt = "You are Hermes. Keep working until finished."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    return agent


def _run(agent, responses):
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("Add the domain definition and verify it.")


def test_legacy_local_prompt_retries_progress_narration_until_marked_final():
    agent = _local_agent()
    agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
    result = _run(agent, [
        _response("Now I need to add the domain definition itself."),
        _response(f"The definition is added and verified.\n{COMPLETION_STATUS_MARKER}"),
    ])

    assert result["completed"] is True
    assert result["final_response"] == "The definition is added and verified."
    assert agent.client.chat.completions.create.call_count == 2
    second_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_messages[0]["content"] == agent._cached_system_prompt
    assert second_messages[-1]["content"] == COMPLETION_STATUS_NUDGE
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
    assert all(
        COMPLETION_STATUS_MARKER not in str(m.get("content") or "")
        for m in result["messages"] if m.get("role") == "assistant"
    )


def test_unconfirmed_local_response_exhausts_two_nudges_as_partial_not_complete():
    agent = _local_agent()
    result = _run(agent, [
        _response("Now I need to define the domain."),
        _response("I should add the metadata next."),
        _response("The metadata still needs to be added."),
    ])

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["turn_exit_reason"] == "completion_status_unconfirmed"
    assert result["failure_reason"] == "truncated"
    assert result["final_response"] == "The metadata still needs to be added."
    assert agent.client.chat.completions.create.call_count == 3
    assert not any(m.get("_completion_status_retry") for m in result["messages"])


def test_marker_without_a_user_facing_answer_does_not_confirm_completion():
    agent = _local_agent()
    result = _run(agent, [
        _response(COMPLETION_STATUS_MARKER),
        _response(f"The requested work is complete.\n{COMPLETION_STATUS_MARKER}"),
    ])

    assert result["completed"] is True
    assert result["final_response"] == "The requested work is complete."
    assert agent.client.chat.completions.create.call_count == 2
    assert not any(m.get("_completion_status_retry") for m in result["messages"])


def test_remote_legacy_prompt_does_not_require_local_completion_marker():
    agent = _local_agent(base_url="https://api.example/v1")
    result = _run(agent, [_response("The short answer is 42.")])
    assert result["completed"] is True
    assert result["final_response"] == "The short answer is 42."
    assert agent.client.chat.completions.create.call_count == 1
