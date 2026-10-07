"""Native attachments obey the turn's authoritative output transform (offline real loop)."""
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths
from tests.agent.native_responses_image_support import response, run_turn, text_item

pytest_plugins = ("tests.agent.native_responses_image_support",)


@pytest.mark.parametrize("held_failure", [False, True])
def test_native_image_removal_is_authoritative(make_agent, monkeypatch, tmp_path, held_failure):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    seen = []

    def redactor(**payload):
        seen.append(payload["response_text"])
        raise RuntimeError("offline held redactor failure")

    def invoke(name, **payload):
        if name == "transform_llm_output":
            seen.append(payload["response_text"])
            return ["Replacement without the original attachment."]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    if held_failure:
        monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (redactor,))
    agent = make_agent(native=True)
    agent._session_db = SessionDB(tmp_path / "transformed.db")
    agent._ensure_db_session()
    try:
        result, requests = run_turn(agent, monkeypatch, response(text_item("Original."), image_item()))
        assert len(requests) == len(seen) == 1
        original = media_paths(seen[0])
        assert len(original) == 1 and original[0].is_file()
        expected = HELD_TRANSFORM_FAILED_REPLY if held_failure else "Replacement without the original attachment."
        assert result["final_response"] == expected
        assert not media_paths(result["final_response"])
        assert result["response_transformed"] is True
        assert result["completed"] is True
        assert agent._hold_stream is held_failure
        if held_failure:
            assert result["pre_transform_response"] is None
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert stored[-1]["content"] == expected
        assert result["messages"][-1]["content"] == expected
        assert Path(original[0]).is_file()  # withholding delivery must not delete paid output
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("held_failure", [False, True])
@pytest.mark.parametrize("summary_succeeds", [False, True])
def test_budget_images_reach_transform_before_recovery_persistence(
    make_agent, monkeypatch, tmp_path, held_failure, summary_succeeds,
):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    seen = []

    def redactor(**payload):
        seen.append(payload["response_text"])
        raise RuntimeError("offline recovery redactor failure")

    def invoke(name, **payload):
        if name == "transform_llm_output":
            seen.append(payload["response_text"])
            return ["Recovery replacement without attachments."]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    if held_failure:
        monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (redactor,))
    agent = make_agent(native=True)
    agent.max_iterations = 1
    agent._session_db = SessionDB(tmp_path / "recovery.db")
    agent._ensure_db_session()
    raw = response(image_item("valid"), image_item("invalid", result=None), status="incomplete")
    followup = [response(text_item("Budget summary."), response_id="resp_2")] if summary_succeeds else []
    try:
        result, requests = run_turn(agent, monkeypatch, raw, *followup)
        assert len(requests) == 2
        assert len(seen) == 1
        paths = media_paths(seen[0])
        assert len(paths) == 1 and paths[0].is_file()
        assert "Image output error:" in seen[0]
        expected = HELD_TRANSFORM_FAILED_REPLY if held_failure else "Recovery replacement without attachments."
        assert result["final_response"] == expected
        assert result["completed"] is False
        assert result["failed"] is False
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert stored[-1]["content"] == expected
        assert paths[0].is_file()
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("exit_path", ["image_error", "function", "text", "interrupt"])
def test_held_failure_recovery_never_persists_original_attachments(
    make_agent, monkeypatch, tmp_path, exit_path,
):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    seen = []

    def redactor(**payload):
        seen.append(payload["response_text"])
        raise RuntimeError("offline early-exit redactor failure")

    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (redactor,))
    agent = make_agent(native=True, tools=exit_path == "function")
    agent._session_db = SessionDB(tmp_path / "early-exit.db")
    agent._ensure_db_session()
    siblings = [image_item("valid"), image_item("invalid", result=None)]
    status = "completed" if exit_path == "image_error" else "incomplete"
    if exit_path == "function":
        siblings.append(NS(type="function_call", id="fc_1", call_id="call_1", name="terminal",
                           status="completed", arguments='{"unfinished":'))
    raw = response(*siblings, status=status)
    raw.incomplete_details = NS(reason="max_output_tokens")
    calls = []
    expected_calls = {"image_error": 1, "function": 5, "text": 3, "interrupt": 2}[exit_path]
    if exit_path == "interrupt":
        agent.max_iterations = 1

    def send(kwargs, **callbacks):
        calls.append(kwargs)
        assert len(calls) <= expected_calls
        if exit_path == "interrupt" and len(calls) == 2:
            raise InterruptedError("offline summary interruption")
        return raw

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", send)
    try:
        result = agent.run_conversation("Produce the requested output.")
        assert len(calls) == expected_calls
        assert len(seen) == 1
        paths = media_paths(seen[0])
        assert len(paths) == 1 and paths[0].is_file()
        assert "Image output error:" in seen[0]
        assert result["final_response"] == HELD_TRANSFORM_FAILED_REPLY
        assert result["completed"] is False
        assert result.get("pre_transform_response") is None
        assert result.get("interrupted", False) is (exit_path == "interrupt")
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert stored[-1]["content"] == HELD_TRANSFORM_FAILED_REPLY
        assert paths[0].is_file()
    finally:
        agent._session_db.close()


def test_held_continuation_ceiling_includes_prior_round_image_before_transform(
    make_agent, monkeypatch, tmp_path,
):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    seen = []

    def redactor(**payload):
        seen.append(payload["response_text"])
        raise RuntimeError("offline continuation redactor failure")

    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (redactor,))
    monkeypatch.setattr("model_tools.handle_function_call", lambda *a, **kw: '{"ok": true}')
    agent = make_agent(native=True, tools=True)
    agent.max_iterations = 10
    agent._session_db = SessionDB(tmp_path / "ceiling.db")
    agent._ensure_db_session()
    agent._hold_stream = True
    agent._held_transforms = (redactor,)
    from agent.turn_truncation import collapse_continuation_trail
    from tests.agent.native_responses_image_support import intake

    # A completed image from a prior round is absent from the later length fragments.
    intake(agent, response(image_item()))
    messages = [{"role": "user", "content": "Produce the requested output."}]
    try:
        final = collapse_continuation_trail(
            agent, messages, 0, finish_reason="length", parts=["A partial answer."],
        )
        assert len(seen) == 1
        paths = media_paths(seen[0])
        assert len(paths) == 1 and paths[0].is_file()
        assert final == HELD_TRANSFORM_FAILED_REPLY
        agent._persist_session(messages, None)
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert stored[-1]["content"] == HELD_TRANSFORM_FAILED_REPLY
        assert paths[0].is_file()
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("artifacts", [None, {}])
def test_optional_native_image_store_does_not_block_text_transform(monkeypatch, artifacts):
    from agent.turn_finalizer import apply_llm_output_transform

    agent = NS(session_id="text-only", model="fake", platform="api_server",
               _responses_image_artifacts=artifacts, _hold_stream=False)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: ["Rewritten reply."])
    result = apply_llm_output_transform(agent, "Original reply.", turn_id="text-turn")
    assert result == ("Rewritten reply.", True, "Original reply.")


def test_optional_native_image_store_is_initialized_at_intake(make_agent):
    from tests.agent.native_responses_image_support import intake

    agent = make_agent(native=True)
    agent._responses_image_artifacts = None
    intake(agent, response(image_item()))
    artifacts = vars(agent)["_responses_image_artifacts"]
    assert isinstance(artifacts, dict)
    paths = [Path(item["path"]) for item in artifacts.values()]
    assert len(paths) == 1 and paths[0].is_file()
