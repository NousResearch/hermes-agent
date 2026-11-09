"""Early native-image finals transform before their first durable flush (offline loop)."""
import copy
from types import SimpleNamespace as NS

import pytest

from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths
from tests.agent.native_responses_image_support import response, text_item

pytest_plugins = ("tests.agent.native_responses_image_support",)


@pytest.mark.parametrize("kind", ["text", "function"])
@pytest.mark.parametrize("artifacts", ["valid", "mixed", "none"])
@pytest.mark.parametrize("mode", ["ordinary", "ordinary_failure", "held_rewrite", "held_failure"])
def test_incomplete_native_image_final_transforms_before_persistence(
    make_agent, monkeypatch, tmp_path, kind, artifacts, mode,
):
    seen, persisted, calls = [], [], []

    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    def rewrite(**payload):
        seen.append(payload["response_text"])
        if mode in {"ordinary_failure", "held_failure"}:
            raise RuntimeError("offline held transform failure")
        return "Reviewed replacement."

    def invoke(name, **payload):
        return [rewrite(**payload)] if name == "transform_llm_output" else []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms",
                        lambda: (rewrite,) if mode.startswith("held_") else ())
    agent = make_agent(native=True, tools=kind == "function")
    agent.max_iterations = 10
    agent._session_db = SessionDB(tmp_path / "turn.db")
    agent._ensure_db_session()
    original = agent._persist_session

    def persist(messages, history):
        persisted.append((len(seen), copy.deepcopy(messages)))
        return original(messages, history)

    items = [text_item("A partial answer.")] if artifacts == "none" else [image_item()]
    if artifacts == "none":
        items[0].status = "incomplete"
    if artifacts == "mixed":
        items.append(image_item("invalid", result=None))
    if kind == "function":
        items.append(NS(type="function_call", id="fc", call_id="call", name="terminal",
                        status="completed", arguments='{"unfinished":'))
    raw = response(*items, status="incomplete")
    expected_calls = 5 if kind == "function" else 3
    raw.incomplete_details = NS(reason="max_output_tokens")

    def send(kwargs, **callbacks):
        calls.append(kwargs)
        assert len(calls) <= expected_calls
        return raw

    monkeypatch.setattr(agent, "_persist_session", persist)
    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    try:
        result = agent.run_conversation("Produce the requested image.")
        assert len(calls) == expected_calls
        assert len(seen) == 1
        paths = media_paths(seen[0])
        assert len(paths) == (0 if artifacts == "none" else 1)
        assert all(path.is_file() for path in paths)
        expected = HELD_TRANSFORM_FAILED_REPLY if mode == "held_failure" else "Reviewed replacement."
        if mode == "ordinary_failure":
            expected = seen[0]  # ordinary observers remain fail-open, not retroactive redactors
        assert result["final_response"] == expected
        assert result["response_transformed"] is (mode != "ordinary_failure")
        assert result["pre_transform_response"] == (seen[0] if mode == "ordinary" else None)
        assert result["partial"] is True
        final_flushes = [(count, rows) for count, rows in persisted
                         if any(row.get("content") == result["final_response"] for row in rows)]
        if artifacts != "none" or mode != "ordinary_failure":
            assert final_flushes and all(count == 1 for count, _ in final_flushes)
            assert persisted[-1][1][-1]["content"] == result["final_response"]
            stored = agent._session_db.get_messages_as_conversation(agent.session_id)
            assert stored[-1]["content"] == result["final_response"]
        assert len(getattr(agent, "_responses_image_artifacts", {})) == {"valid": 1, "mixed": 2, "none": 0}[artifacts]
        assert len(list((tmp_path / "cache/generated/images").glob("*.png"))) == len(paths)
        assert all(path.is_file() for path in paths)
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("kind", ["scratchpad", "repetition"])
def test_native_image_sibling_early_final_transforms_before_flush(make_agent, monkeypatch, tmp_path, kind):
    seen, calls = [], []

    def invoke(name, **payload):
        if name == "transform_llm_output":
            seen.append(payload["response_text"])
            return ["Reviewed recovery."]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    agent = make_agent(native=True)
    agent.max_iterations = 5
    agent._session_db = SessionDB(tmp_path / "sibling.db")
    agent._ensure_db_session()
    text = "<REASONING_SCRATCHPAD>unfinished" if kind == "scratchpad" else "echo " * 8000
    raw = response(text_item(text), image_item())

    def send(kwargs, **callbacks):
        calls.append(kwargs)
        assert len(calls) <= (3 if kind == "scratchpad" else 1)
        return raw

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    try:
        result = agent.run_conversation("Produce an image and an explanation.")
        assert len(calls) == (3 if kind == "scratchpad" else 1)
        assert len(seen) == 1
        assert len(media_paths(seen[0])) == 1 and media_paths(seen[0])[0].is_file()
        assert result["final_response"] == "Reviewed recovery."
        assert result["response_transformed"] is True
        assert result["pre_transform_response"] == seen[0]
        assert result["partial"] is True
        assert agent._session_db.get_messages_as_conversation(agent.session_id)[-1]["content"] == result["final_response"]
    finally:
        agent._session_db.close()



def test_ordinary_interrupted_native_image_keeps_untransformed_partial(make_agent, monkeypatch, tmp_path):
    seen, calls = [], []

    def invoke(name, **payload):
        if name == "transform_llm_output":
            seen.append(payload["response_text"])
            return ["Must not replace an ordinary interrupted partial."]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    agent = make_agent(native=True)
    agent.max_iterations = 1
    agent._session_db = SessionDB(tmp_path / "interrupted.db")
    agent._ensure_db_session()
    raw = response(image_item(), status="incomplete")

    def send(kwargs, **callbacks):
        calls.append(kwargs)
        assert len(calls) <= 2
        if len(calls) == 2:
            raise InterruptedError("offline summary interruption")
        return raw

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    try:
        result = agent.run_conversation("Produce an image.")
        assert result["interrupted"] is True and len(calls) == 2
        assert seen == []
        paths = media_paths(result["final_response"])
        assert len(paths) == 1 and paths[0].is_file()
        assert not result.get("response_transformed")
        assert agent._session_db.get_messages_as_conversation(agent.session_id)[-1]["content"] == result["final_response"]
    finally:
        agent._session_db.close()
