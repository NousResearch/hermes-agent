"""Native artifacts survive real route fallback into length-recovery without bypassing transforms."""
import json
from types import SimpleNamespace as NS

import pytest

from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths
from tests.agent.native_responses_image_support import response

pytest_plugins = ("tests.agent.native_responses_image_support",)


def test_native_image_fallback_length_final_is_transformed_before_flush(make_agent, monkeypatch, tmp_path):
    url = "https://fallback.invalid/v1"
    (tmp_path / "config.yaml").write_text(json.dumps({"providers": {"fallback": {
        "base_url": url, "api_mode": "chat_completions", "api_key": "offline-fixture"}}}))
    agent = make_agent(native=True, tools=True)
    agent.max_iterations = 12
    agent._fallback_chain = [{"provider": "custom:fallback", "model": "private-model",
                              "base_url": url, "api_mode": "chat_completions", "api_key": "offline-fixture"}]
    agent._session_db = SessionDB(tmp_path / "fallback.db")
    agent._ensure_db_session()
    monkeypatch.setattr("agent.auxiliary_client._create_openai_client",
                        lambda **kw: NS(base_url=url, api_key="offline-fixture", _custom_headers={}))
    monkeypatch.setattr("model_tools.handle_function_call", lambda *a, **kw: '{"ok": true}')
    seen, calls, final_flushes = [], [], []

    def invoke(name, **payload):
        if name == "transform_llm_output":
            seen.append(payload["response_text"])
            return ["Reviewed fallback partial."]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    original = agent._persist_session

    def persist(messages, history):
        if any(row.get("finish_reason") == "length" and not row.get("_length_continuation_fragment")
               for row in messages):
            final_flushes.append((len(seen), messages[-1]["content"]))
        return original(messages, history)

    monkeypatch.setattr(agent, "_persist_session", persist)
    first = response(image_item(), NS(type="function_call", id="fc", call_id="call", name="terminal",
                                      status="completed", arguments="{}"))

    def send(kwargs, **callbacks):
        calls.append((agent.api_mode, kwargs))
        assert len(calls) <= 6
        if len(calls) == 1:
            return first
        if len(calls) == 2:
            error = RuntimeError("402 synthetic insufficient credits")
            error.status_code = 402
            raise error
        return NS(id="partial", choices=[NS(message=NS(content="A partial answer.", tool_calls=[]),
                                             finish_reason="length")],
                  usage=NS(prompt_tokens=17, completion_tokens=3, total_tokens=20))

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    try:
        result = agent.run_conversation("Produce an image and explain it.")
        assert len(calls) == 6
        assert [mode for mode, _ in calls] == ["codex_responses"] * 2 + ["chat_completions"] * 4
        assert len(seen) == 1
        assert len(media_paths(seen[0])) == 1 and media_paths(seen[0])[0].is_file()
        assert result["final_response"] == "Reviewed fallback partial."
        assert result["response_transformed"] is True
        assert result["pre_transform_response"] == seen[0]
        assert final_flushes and all(count == 1 and text == result["final_response"] for count, text in final_flushes)
        assert agent._session_db.get_messages_as_conversation(agent.session_id)[-1]["content"] == result["final_response"]
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held"])
@pytest.mark.parametrize("rewrite", ["noop", "replace", "failure"])
def test_fallback_length_then_refusal_settles_only_the_final_notice(make_agent, monkeypatch, tmp_path, held, rewrite):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    url = "https://fallback.invalid/v1"
    (tmp_path / "config.yaml").write_text(json.dumps({"providers": {"fallback": {
        "base_url": url, "api_mode": "chat_completions", "api_key": "offline-fixture"}}}))
    agent = make_agent(native=True, tools=True)
    agent.max_iterations = 12
    agent._fallback_chain = [{"provider": "custom:fallback", "model": "private-model",
                              "base_url": url, "api_mode": "chat_completions", "api_key": "offline-fixture"}]
    monkeypatch.setattr("agent.auxiliary_client._create_openai_client",
                        lambda **kw: NS(base_url=url, api_key="offline-fixture", _custom_headers={}))
    monkeypatch.setattr("model_tools.handle_function_call", lambda *a, **kw: '{"ok": true}')
    seen, calls = [], []

    def transform(**kwargs):
        seen.append(kwargs["response_text"])
        if rewrite == "failure":
            raise RuntimeError("synthetic transform failure")
        return "Reviewed refusal." if rewrite == "replace" else None

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: (
        [transform(**kw)] if name == "transform_llm_output" else []
    ))
    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (transform,) if held else ())
    first = response(image_item(), NS(type="function_call", id="fc", call_id="call", name="terminal",
                                      status="completed", arguments="{}"))

    def send(kwargs, **callbacks):
        calls.append(agent.api_mode)
        assert len(calls) <= 4, "Refusal must not retry or regenerate the completed image"
        if len(calls) == 1:
            return first
        if len(calls) in (2, 4):
            error = RuntimeError("402 synthetic insufficient credits" if len(calls) == 2
                                 else "400 content_filter: REFUSAL_REASON_MARKER")
            error.status_code = 402 if len(calls) == 2 else 400
            raise error
        return NS(id="partial", choices=[NS(message=NS(content="DISCARDED_PARTIAL_MARKER", tool_calls=[]),
                                             finish_reason="length")],
                  usage=NS(prompt_tokens=17, completion_tokens=3, total_tokens=20))

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    result = agent.run_conversation("Produce an image and explain it.")
    assert calls == ["codex_responses"] * 2 + ["chat_completions"] * 2
    assert result["failure_reason"] == "content_policy_blocked"
    assert result["failed"] and not result["completed"]
    assert result["messages"] == []
    assert len(seen) == 1
    assert "REFUSAL_REASON_MARKER" in seen[0], "The refusal, not its discarded partial, must own settlement"
    assert "DISCARDED_PARTIAL_MARKER" not in seen[0]
    assert len(media_paths(seen[0])) == 1 and media_paths(seen[0])[0].is_file()
    if rewrite == "replace" or (held and rewrite == "failure"):
        expected = HELD_TRANSFORM_FAILED_REPLY if rewrite == "failure" else "Reviewed refusal."
        assert result["final_response"] == expected
        assert result["response_transformed"] is True
        assert not media_paths(result["final_response"])
    else:
        assert len(media_paths(result["final_response"])) == 1
    if held:
        assert result["pre_transform_response"] is None
