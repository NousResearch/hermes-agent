"""Post-response retry boundary regressions (non-streaming transports).

Once the provider has returned a terminal response, any Hermes-local
processing error must stop the turn through the unified finalizer with
turn_exit_reason="local_post_response_error" — never another model request,
no continuation, no length stub. Network/transport failures BEFORE the
terminal response keep their baseline retry classification.

Boundary flags live at the raw provider call in every dispatch branch; the
tests use the real AIAgent.run_conversation entry and assert the actual
provider wire call count.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
import run_agent


def _ns_response(text="ok"):
    msg = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _ns_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key",
            base_url="https://provider.invalid/v1",
            provider="custom",
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "system"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    agent.client = MagicMock()
    return agent


def _ns_run(agent):
    with (
        patch("run_agent.jittered_backoff", return_value=0),
        patch("time.sleep", return_value=None),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("ping")


def test_chat_raw_return_then_wrapper_local_failure_calls_provider_once():
    agent = _ns_agent()
    create = agent.client.chat.completions.create
    create.return_value = _ns_response("visible")

    with patch(
        "agent.chat_completion_helpers._reset_stale_streak",
        side_effect=RuntimeError("post-return local state"),
    ):
        result = _ns_run(agent)

    assert create.call_count == 1
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert result["final_response"]


def test_network_error_before_raw_response_still_retries():
    agent = _ns_agent()
    create = agent.client.chat.completions.create
    create.side_effect = [ConnectionError("wire down"), _ns_response("recovered")]

    result = _ns_run(agent)

    assert create.call_count == 2
    assert result["failed"] is False
    assert result["completed"] is True
    assert result["final_response"] == "recovered"


def test_bedrock_raw_return_then_local_normalization_failure_calls_once(monkeypatch):
    agent = _ns_agent()
    agent.api_mode = "bedrock_converse"
    agent.provider = "bedrock"

    client = MagicMock()
    client.converse.return_value = {"output": {"message": {"content": []}}}

    monkeypatch.setattr(
        "agent.bedrock_adapter._get_bedrock_runtime_client",
        lambda _region: client,
    )
    monkeypatch.setattr(
        "agent.bedrock_adapter.normalize_converse_response",
        lambda _raw: (_ for _ in ()).throw(KeyError("local normalization")),
    )
    monkeypatch.setattr(
        agent,
        "_build_api_kwargs",
        lambda _messages: {
            "__bedrock_region__": "us-east-1",
            "__bedrock_converse__": True,
        },
    )

    result = _ns_run(agent)

    assert client.converse.call_count == 1
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "local_post_response_error"


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, call_id: str = "call-structured-content-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(content, *, finish_reason: str = "stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, interim_callback=None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="test-model",
            api_key="test-key-not-a-secret",
            base_url="https://test.invalid/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            interim_assistant_callback=interim_callback,
        )
    agent._cached_system_prompt = "You are a test double."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = set(tool_names)
    agent.max_iterations = 4
    return agent


def _run(agent: AIAgent, request, *, extra_patches=()):
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", request))
        save = stack.enter_context(patch.object(agent, "_save_trajectory"))
        cleanup = stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        persist = stack.enter_context(patch.object(agent, "_persist_session"))
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)
        result = agent.run_conversation("structured content regression")
    return result, save, cleanup, persist


class _TextPart:
    type = "output_text"
    text = "object text"


class _TextPartWithExplosiveLegacyContent:
    type = "output_text"
    text = "safe object text"

    @property
    def content(self):
        raise RuntimeError("legacy content must not be read")


class _Opaque:
    def __str__(self) -> str:
        return "MUST_NOT_LEAK"


class _ExplosiveContentObject:
    @property
    def type(self):
        raise RuntimeError("must be ignored")




@pytest.mark.parametrize(
    "exc_type",
    [TypeError, AttributeError, ValueError, KeyError, IndexError, RuntimeError],
)
def test_final_materialization_local_errors_stop_after_one_request(exc_type):
    agent = _make_agent()
    agent._stream_callback = MagicMock()
    request = MagicMock(return_value=_response("done"))
    result, save, cleanup, persist = _run(
        agent,
        request,
        extra_patches=(
            patch.object(agent, "_build_assistant_message", side_effect=exc_type("injected")),
        ),
    )

    assert request.call_count == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert result["final_response"]
    assert result["messages"][-1]["role"] == "assistant"
    assert save.call_count == cleanup.call_count == 1
    assert persist.call_count >= 1
    assert agent._stream_callback is None


def test_length_path_local_runtime_error_stops_after_one_request():
    agent = _make_agent()
    request = MagicMock(
        return_value=_response(
            [{"type": "output_text", "text": "partial"}],
            finish_reason="length",
        )
    )
    result, _, cleanup, persist = _run(
        agent,
        request,
        extra_patches=(
            patch.object(agent, "_build_assistant_message", side_effect=RuntimeError("injected")),
        ),
    )

    assert request.call_count == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


def test_early_normalization_runtime_error_stops_after_one_request():
    agent = _make_agent()
    request = MagicMock(return_value=_response("done"))
    result, _, cleanup, persist = _run(
        agent,
        request,
        extra_patches=(
            patch(
                "agent.chat_completion_helpers.flatten_message_text",
                side_effect=RuntimeError("normalization injected"),
            ),
        ),
    )

    assert request.call_count == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


@pytest.mark.parametrize("interim_enabled", [False, True])
def test_normal_tool_then_final_text_path_remains_reachable(interim_enabled):
    callback = MagicMock() if interim_enabled else None
    agent = _make_agent("review_tool", interim_callback=callback)
    request = MagicMock(
        side_effect=[
            _response(
                "working",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("review_tool")],
            ),
            _response("finished"),
        ]
    )
    with patch("run_agent.handle_function_call", return_value="ok") as tool:
        result, _, cleanup, persist = _run(agent, request)

    assert request.call_count == 2
    assert tool.call_count == 1
    assert result["final_response"] == "finished"
    assert result["completed"] is True
    assert result["failed"] is False
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    if interim_enabled:
        callback.assert_called()
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


def test_tool_materialization_runtime_error_stops_without_executing_tool():
    agent = _make_agent("review_tool")
    request = MagicMock(
        return_value=_response(
            "working",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("review_tool")],
        )
    )
    with patch("run_agent.handle_function_call") as tool:
        result, _, cleanup, persist = _run(
            agent,
            request,
            extra_patches=(
                patch.object(
                    agent,
                    "_build_assistant_message",
                    side_effect=RuntimeError("tool materialization injected"),
                ),
            ),
        )

    assert request.call_count == 1
    assert tool.call_count == 0
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


def test_response_side_middleware_error_stops_after_one_provider_request():
    agent = _make_agent()
    request = MagicMock(return_value=_response("done"))

    def response_side_failure(api_kwargs, call_provider, **_kwargs):
        call_provider(api_kwargs)
        raise RuntimeError("response middleware injected")

    result, _, cleanup, persist = _run(
        agent,
        request,
        extra_patches=(
            patch(
                "hermes_cli.middleware.run_llm_execution_middleware",
                side_effect=response_side_failure,
            ),
        ),
    )

    assert request.call_count == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


@pytest.mark.parametrize("interim_enabled", [False, True])
def test_interim_visible_text_local_error_stops_after_one_request(interim_enabled):
    callback = MagicMock() if interim_enabled else None
    agent = _make_agent("review_tool", interim_callback=callback)
    request = MagicMock(
        return_value=_response(
            "working",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("review_tool")],
        )
    )
    with patch("run_agent.handle_function_call") as tool:
        result, _, cleanup, persist = _run(
            agent,
            request,
            extra_patches=(
                patch.object(
                    agent,
                    "_interim_assistant_visible_text",
                    side_effect=RuntimeError("interim injected"),
                ),
            ),
        )

    assert request.call_count == 1
    assert tool.call_count == 0
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


def test_network_error_then_success_still_retries_provider():
    agent = _make_agent()
    request = MagicMock(side_effect=[ConnectionError("transient"), _response("recovered")])
    result, _, _, _ = _run(
        agent,
        request,
        extra_patches=(
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch("agent.conversation_loop.time.sleep"),
        ),
    )

    assert request.call_count == 2
    assert result["final_response"] == "recovered"
    assert result["completed"] is True
    assert result["failed"] is False


def test_persistent_network_error_honors_retry_limit():
    agent = _make_agent()
    agent._api_max_retries = 3
    request = MagicMock(side_effect=ConnectionError("persistent"))
    result, _, _, _ = _run(
        agent,
        request,
        extra_patches=(
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch("agent.conversation_loop.time.sleep"),
        ),
    )

    assert request.call_count == 3
    assert result["completed"] is False
    assert result["failed"] is True


def test_local_error_after_final_row_keeps_transcript_and_result_aligned():
    agent = _make_agent()
    agent.quiet_mode = False
    request = MagicMock(return_value=_response("model answer"))

    def fail_after_final_append(message, *_args, **_kwargs):
        if "Conversation completed" in str(message):
            raise RuntimeError("display completion injected")

    result, _, cleanup, persist = _run(
        agent,
        request,
        extra_patches=(
            patch.object(agent, "_safe_print", side_effect=fail_after_final_append),
        ),
    )

    assert request.call_count == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == result["final_response"]
    assert result["messages"][-1].get("tool_calls") is None
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


def test_finalizer_cleanup_and_persist_failures_do_not_request_provider_again():
    agent = _make_agent()
    request = MagicMock(return_value=_response("done"))
    with (
        patch.object(agent, "_interruptible_api_call", request),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources", side_effect=RuntimeError("cleanup")),
        patch.object(agent, "_persist_session", side_effect=RuntimeError("persist")),
    ):
        result = agent.run_conversation("cleanup failure regression")

    assert request.call_count == 1
    assert result["completed"] is True
    assert result["failed"] is False
    assert len(result.get("cleanup_errors", [])) == 2


# ── terminal local-failure hardening: role-safety + generic surfaces ─────────

def _adjacent_assistant_pairs(messages):
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return [(i, roles[i]) for i in range(1, len(roles))
            if roles[i] == "assistant" and roles[i - 1] == "assistant"]


def _dangling_tool_call_ids(messages):
    answered = {
        m.get("tool_call_id")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    }
    dangling = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id") not in answered:
                    dangling.append(tc.get("id"))
    return dangling


def _persisted_payloads(persist):
    out = []
    for call in persist.call_args_list:
        payload = call.args[0] if call.args else []
        out.append(payload)
    return out


def test_adjacent_assistant_never_persisted_after_terminal_local_failure():
    """RED->GREEN: a local failure AFTER a plain assistant row was appended
    must NOT produce assistant->assistant in the transcript. The fixed note
    is merged into the existing row, preserving the delivered answer."""
    agent = _make_agent()
    agent.quiet_mode = False
    request = MagicMock(return_value=_response("model answer"))

    def fail_after_final_append(message, *_a, **_k):
        if "Conversation completed" in str(message):
            raise RuntimeError("display completion injected")

    result, _, _, persist = _run(
        agent,
        request,
        extra_patches=(
            patch.object(agent, "_safe_print", side_effect=fail_after_final_append),
        ),
    )

    assert request.call_count == 1
    assert result["turn_exit_reason"] == "local_post_response_error"
    # No adjacent assistant rows in the live transcript...
    assert _adjacent_assistant_pairs(result["messages"]) == []
    # ...and none in any persisted payload either.
    for payload in _persisted_payloads(persist):
        assert _adjacent_assistant_pairs(payload) == []
    # The delivered answer is preserved, with the fixed note appended to the
    # SAME row (no second assistant row).
    tail = result["messages"][-1]
    assert tail["role"] == "assistant"
    assert tail["content"].startswith("model answer")
    assert "local error" in tail["content"]
    assert result["final_response"] == tail["content"]


def test_local_failure_note_has_no_exception_details_inner_exit():
    """Inner exit: RuntimeError text never reaches final_response,
    transcript, or persisted payloads — but DOES stay in the logs."""
    import io, logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logging.getLogger().addHandler(handler)
    try:
        agent = _make_agent()
        request = MagicMock(return_value=_response("done"))
        result, _, _, persist = _run(
            agent,
            request,
            extra_patches=(
                patch.object(
                    agent,
                    "_build_assistant_message",
                    side_effect=RuntimeError("sentinel-secret-XYZ"),
                ),
            ),
        )
    finally:
        logging.getLogger().removeHandler(handler)

    assert result["turn_exit_reason"] == "local_post_response_error"
    assert "sentinel-secret-XYZ" not in (result["final_response"] or "")
    assert "RuntimeError" not in (result["final_response"] or "")
    for m in result["messages"]:
        if isinstance(m.get("content"), str):
            assert "sentinel-secret-XYZ" not in m["content"]
    for payload in _persisted_payloads(persist):
        for m in payload:
            if isinstance(m.get("content"), str):
                assert "sentinel-secret-XYZ" not in m["content"]
    # The full exception stays in the logs.
    assert "sentinel-secret-XYZ" in log_stream.getvalue()


def test_local_failure_note_has_no_exception_details_outer_exit():
    """Outer exit (display completion failure after final append): same
    secrecy contract."""
    agent = _make_agent()
    agent.quiet_mode = False
    request = MagicMock(return_value=_response("model answer"))

    def fail_after_final_append(message, *_a, **_k):
        if "Conversation completed" in str(message):
            raise RuntimeError("sentinel-secret-XYZ")

    result, _, _, persist = _run(
        agent,
        request,
        extra_patches=(
            patch.object(agent, "_safe_print", side_effect=fail_after_final_append),
        ),
    )

    assert result["turn_exit_reason"] == "local_post_response_error"
    assert "sentinel-secret-XYZ" not in (result["final_response"] or "")
    assert "RuntimeError" not in (result["final_response"] or "")
    for m in result["messages"]:
        if isinstance(m.get("content"), str):
            assert "sentinel-secret-XYZ" not in m["content"]
    for payload in _persisted_payloads(persist):
        for m in payload:
            if isinstance(m.get("content"), str):
                assert "sentinel-secret-XYZ" not in m["content"]


def test_synthetic_tool_result_is_generic_when_tool_dispatch_fails():
    """A post-append failure inside the tool dispatcher must synthesize
    generic tool results (no exception text) and close every tool_call."""
    agent = _make_agent("review_tool")
    request = MagicMock(
        return_value=_response(
            "working",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("review_tool")],
        )
    )
    result, _, _, persist = _run(
        agent,
        request,
        extra_patches=(
            patch.object(
                agent,
                "_execute_tool_calls",
                side_effect=RuntimeError("sentinel-secret-XYZ"),
            ),
        ),
    )

    assert result["turn_exit_reason"] == "local_post_response_error"
    # Pairing invariant: every tool_call.id got a matching tool result.
    assert _dangling_tool_call_ids(result["messages"]) == []
    # Synthetic tool results carry the fixed generic note only.
    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert tool_rows, "expected a synthetic tool result"
    for row in tool_rows:
        assert "sentinel-secret-XYZ" not in row["content"]
        assert "RuntimeError" not in row["content"]
    # No adjacent assistants anywhere (live or persisted).
    assert _adjacent_assistant_pairs(result["messages"]) == []
    for payload in _persisted_payloads(persist):
        assert _adjacent_assistant_pairs(payload) == []
        for m in payload:
            if isinstance(m.get("content"), str):
                assert "sentinel-secret-XYZ" not in m["content"]




def test_cli_visible_output_has_no_exception_details(capsys):
    """The error line printed to the terminal must be generic."""
    agent = _make_agent()
    request = MagicMock(return_value=_response("done"))
    result, _, _, _ = _run(
        agent,
        request,
        extra_patches=(
            patch.object(
                agent,
                "_build_assistant_message",
                side_effect=RuntimeError("sentinel-secret-XYZ"),
            ),
        ),
    )
    out = capsys.readouterr().out
    assert result["turn_exit_reason"] == "local_post_response_error"
    assert "sentinel-secret-XYZ" not in out
    assert "RuntimeError" not in out


def test_partial_visible_text_preserved_after_terminal_local_failure():
    """Text already delivered to the user stays in the transcript; the
    failure is appended to the SAME row, and the provider is NOT called again."""
    from types import SimpleNamespace as _SN

    deltas = []

    def _mk_chunk(text=None, finish=None, usage=None):
        delta = _SN(content=text, reasoning_content=None, reasoning=None, tool_calls=None)
        return _SN(choices=[_SN(delta=delta, finish_reason=finish)], model="m", usage=usage)

    def _stream():
        usage = _SN(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        yield _mk_chunk("visible-part")
        yield _mk_chunk("-tail", finish="stop", usage=usage)

    class _FakeStreamClient:
        def __init__(self):
            self.calls = 0
            self.chat = _SN(completions=_SN(create=self._create))

        def _create(self, **kw):
            self.calls += 1
            return _stream()

        def close(self):
            pass

    fake = _FakeStreamClient()
    agent = _make_agent()
    agent.quiet_mode = False
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda text: deltas.append(text)
    agent.client = fake

    from contextlib import ExitStack as _ES
    with _ES() as st:
        st.enter_context(patch("run_agent.jittered_backoff", return_value=0))
        st.enter_context(patch("time.sleep", return_value=None))
        persist = st.enter_context(patch.object(agent, "_persist_session"))
        st.enter_context(patch.object(agent, "_save_trajectory"))
        st.enter_context(patch.object(agent, "_cleanup_task_resources"))
        st.enter_context(patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake))
        # Local failure strictly after the terminal frame.
        st.enter_context(patch.object(
            agent, "_build_assistant_message",
            side_effect=RuntimeError("post-terminal materialization failure"),
        ))
        result = agent.run_conversation("visible partial probe")

    assert fake.calls == 1
    assert result["turn_exit_reason"] == "local_post_response_error"
    # The streamed text the user saw is still delivered intact.
    assert "".join(deltas) == "visible-part-tail"
    # The failure note is present and generic.
    assert "local error" in (result["final_response"] or "")
    assert "post-terminal materialization failure" not in (result["final_response"] or "")
    for payload in _persisted_payloads(persist):
        assert _adjacent_assistant_pairs(payload) == []


def test_non_local_provider_error_does_not_show_local_failure_note(capsys):
    """Narrow regression: a retryable provider/transport error must keep its
    original provider error display semantics and must NOT be reported as a
    local response-processing failure (LOCAL_FAILURE_NOTE)."""
    from agent.message_sanitization import LOCAL_FAILURE_NOTE

    agent = _make_agent()
    request = MagicMock(
        side_effect=[
            ConnectionError("wire down attempt 1"),
            ConnectionError("wire down attempt 2"),
            ConnectionError("wire down attempt 3"),
        ]
    )
    result, _, _, _ = _run(agent, request)

    out = capsys.readouterr().out
    # Retry classification is unchanged: three attempts were made.
    assert request.call_count == 3
    # The provider error display keeps its original semantics...
    assert "wire down" in out
    # ...and is NEVER misreported as a local response-processing failure.
    assert LOCAL_FAILURE_NOTE not in out
    assert result.get("turn_exit_reason") != "local_post_response_error"
    assert result.get("turn_exit_reason") != "local_post_response_error"
