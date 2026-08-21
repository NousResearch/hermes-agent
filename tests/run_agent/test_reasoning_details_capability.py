"""Behavior coverage for deployment-scoped reasoning_details recovery."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.backend_identity import BackendIdentity
from agent.reasoning_replay import remember_reasoning_details_rejection
from hermes_state import SessionDB
from run_agent import AIAgent


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent(
    *,
    provider: str = "custom:strict",
    model: str = "model-a",
    base_url: str = "https://strict.example/v1",
) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider=provider,
            model=model,
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _history(details: list[dict] | None = None) -> list[dict]:
    messages = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    if details is not None:
        messages[1]["reasoning_details"] = details
    return messages


def _unsupported(status_code: int = 400) -> Exception:
    error = Exception(
        "Extra inputs are not permitted, field: messages[1].reasoning_details"
    )
    error.status_code = status_code
    return error


def _chunk(*, content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(model="test/model", choices=[choice])


def _content_stream(text: str):
    return iter([_chunk(content=text), _chunk(finish_reason="stop")])


def _tool_stream():
    function = SimpleNamespace(name="web_search", arguments="{}")
    tool_call = SimpleNamespace(index=0, id="call_1", function=function)
    return iter(
        [
            _chunk(tool_calls=[tool_call]),
            _chunk(finish_reason="tool_calls"),
        ]
    )


def _wire_has_reasoning_details(kwargs: dict) -> bool:
    return any(
        "reasoning_details" in message
        for message in kwargs.get("messages", [])
        if isinstance(message, dict)
    )


@pytest.mark.parametrize("status_code", [400, 422])
def test_real_streaming_learns_once_across_user_turns(status_code):
    agent = _make_agent()
    agent.stream_delta_callback = MagicMock()
    details = [
        {"type": "thinking", "thinking": "private", "signature": "signed"}
    ]
    history = _history(details)
    wire_payloads: list[dict] = []

    def send(**kwargs):
        wire_payloads.append(copy.deepcopy(kwargs))
        assert kwargs.get("stream") is True
        if len(wire_payloads) == 1:
            raise _unsupported(status_code)
        return _content_stream("first recovered" if len(wire_payloads) == 2 else "second turn")

    agent.client.chat.completions.create.side_effect = send
    with (
        patch.object(agent, "_replace_primary_openai_client", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("first", conversation_history=history)
        second = agent.run_conversation(
            "second", conversation_history=first["messages"]
        )

    assert first["completed"] is True
    assert second["completed"] is True
    assert len(wire_payloads) == 3
    assert _wire_has_reasoning_details(wire_payloads[0])
    assert not _wire_has_reasoning_details(wire_payloads[1])
    assert not _wire_has_reasoning_details(wire_payloads[2])
    assert history[1]["reasoning_details"] is details
    assert any(
        message.get("reasoning_details") is details
        for message in first["messages"]
    )
    assert len(agent._reasoning_details_rejected_backends) == 1


def test_tool_loop_followup_is_filtered_before_its_first_attempt():
    agent = _make_agent()
    agent.stream_delta_callback = MagicMock()
    details = [{"type": "thinking", "signature": "signed"}]
    wire_payloads: list[dict] = []

    def send(**kwargs):
        wire_payloads.append(copy.deepcopy(kwargs))
        assert kwargs.get("stream") is True
        if len(wire_payloads) == 1:
            raise _unsupported(400)
        if len(wire_payloads) == 2:
            return _tool_stream()
        return _content_stream("done")

    agent.client.chat.completions.create.side_effect = send
    with (
        patch.object(agent, "_replace_primary_openai_client", return_value=False),
        patch("run_agent.handle_function_call", return_value="search result"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "next", conversation_history=_history(details)
        )

    assert result["completed"] is True
    assert len(wire_payloads) == 3
    assert _wire_has_reasoning_details(wire_payloads[0])
    assert all(
        not _wire_has_reasoning_details(payload)
        for payload in wire_payloads[1:]
    )


def test_recovery_preserves_returned_history_and_state_db(tmp_path):
    agent = _make_agent()
    agent._disable_streaming = True
    details = [
        {
            "type": "thinking",
            "thinking": "private",
            "signature": "signed",
            "nested": {"opaque": ["value"]},
        }
    ]
    history = _history(details)
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(
        session_id="reasoning-replay-session",
        source="cli",
        model=agent.model,
    )
    # The reasoning-bearing assistant row belongs to the prior turn and would
    # already be durable when this recovery runs.  Seed that real resume shape
    # so the assertion proves the retry path cannot erase it from state.db.
    db.replace_messages("reasoning-replay-session", history)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "reasoning-replay-session"
    agent._last_flushed_db_idx = len(history)
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False

    recovered = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="recovered", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )
    agent.client.chat.completions.create.side_effect = [_unsupported(400), recovered]

    try:
        with (
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "next", conversation_history=history
            )

        durable = db.get_messages_as_conversation(agent.session_id)
    finally:
        db.close()

    assert result["completed"] is True
    assert history[1]["reasoning_details"] is details
    assert any(
        message.get("reasoning_details") is details
        for message in result["messages"]
    )
    persisted = next(
        message for message in durable if message.get("reasoning_details")
    )
    assert persisted["reasoning_details"] == details
    assert persisted["reasoning_details"][0]["nested"] == {"opaque": ["value"]}


def test_model_and_explicit_endpoint_identity_scope():
    endpoint_a = "HTTPS://STRICT.EXAMPLE/v1/"
    agent = _make_agent(model="model-a", base_url=endpoint_a)
    agent._disable_streaming = True
    details = [{"type": "thinking", "signature": "signed"}]
    history = _history(details)
    recovered = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )
    agent.client.chat.completions.create.side_effect = [_unsupported(400), recovered]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("next", conversation_history=history)

    assert result["completed"] is True
    assert agent._reasoning_details_rejected_backends == {
        BackendIdentity.build(
            provider="custom:strict",
            model="model-a",
            base_url=endpoint_a,
        )
    }

    agent.model = "model-b"
    model_b = agent._build_api_kwargs(history)
    assert _wire_has_reasoning_details(model_b)

    agent.model = "model-a"
    agent.base_url = "https://strict.example/v1"
    model_a_again = agent._build_api_kwargs(history)
    assert not _wire_has_reasoning_details(model_a_again)

    agent.base_url = "https://other.example/v1"
    other_endpoint = agent._build_api_kwargs(history)
    assert _wire_has_reasoning_details(other_endpoint)


def test_incomplete_backend_identity_is_not_recorded():
    agent = _make_agent(model="")

    identity, learned = remember_reasoning_details_rejection(agent)

    assert identity.model == ""
    assert learned is False
    assert agent._reasoning_details_rejected_backends == set()


def test_openrouter_signed_replay_keeps_reasoning_details():
    agent = _make_agent(
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        base_url="https://openrouter.ai/api/v1",
    )
    details = [{"type": "thinking", "thinking": "plan", "signature": "signed"}]
    messages = _history(details)

    kwargs = agent._build_api_kwargs(messages)

    assert _wire_has_reasoning_details(kwargs)
    assert messages[1]["reasoning_details"] is details


def test_nous_signed_replay_keeps_anthropic_thinking_block():
    agent = _make_agent(
        provider="nous",
        model="anthropic/claude-sonnet-4",
        base_url="https://inference-api.nousresearch.com/v1",
    )
    details = [{"type": "thinking", "thinking": "plan", "signature": "signed"}]
    messages = _history(details)

    kwargs = agent._build_api_kwargs(messages)

    assistant = next(
        message for message in kwargs["messages"] if message["role"] == "assistant"
    )
    assert assistant["content"][0] == details[0]
    assert messages[1]["reasoning_details"] is details


def test_anthropic_signed_replay_conversion_is_unchanged():
    from agent.anthropic_adapter import convert_messages_to_anthropic

    details = [{"type": "thinking", "thinking": "plan", "signature": "signed"}]
    messages = _history(details)

    _system, converted = convert_messages_to_anthropic(
        messages,
        base_url=None,
        model="claude-opus-4-8",
    )

    assistant = next(message for message in converted if message["role"] == "assistant")
    thinking = [
        block
        for block in assistant["content"]
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    assert thinking == details
    assert messages[1]["reasoning_details"] is details


def test_max_iteration_summary_reuses_learned_filter():
    agent = _make_agent()
    agent._disable_streaming = True
    details = [{"type": "thinking", "signature": "signed"}]
    messages = _history(details)
    identity, learned = remember_reasoning_details_rejection(agent)
    assert learned is True
    assert identity in agent._reasoning_details_rejected_backends
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="summary", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )
    agent.client.chat.completions.create.return_value = response

    result = agent._handle_max_iterations(messages, 60)

    assert result == "summary"
    sent = agent.client.chat.completions.create.call_args.kwargs
    assert not _wire_has_reasoning_details(sent)
    assert messages[1]["reasoning_details"] is details


def test_error_text_without_field_on_wire_does_not_learn_or_retry():
    agent = _make_agent()
    agent._disable_streaming = True
    history = _history()
    agent.client.chat.completions.create.side_effect = _unsupported(400)

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("next", conversation_history=history)

    assert result["completed"] is False
    assert agent.client.chat.completions.create.call_count == 1
    assert agent._reasoning_details_rejected_backends == set()
