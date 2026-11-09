"""Native-image error finals are rewritten before their first assistant append and durable flush."""
import copy
import os
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths
from tests.agent.native_responses_image_support import response, run_turn, text_item

pytest_plugins = ("tests.agent.native_responses_image_support",)


@pytest.mark.parametrize("outcome", ["mixed", "all_invalid", "write_error", "all_write_error"])
@pytest.mark.parametrize("transform", ["ordinary", "held_rewrite", "held_failure"])
def test_image_error_final_rewrite_precedes_append_and_persistence(
    make_agent, monkeypatch, tmp_path, outcome, transform,
):
    import socket
    from agent import turn_final_response
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    def no_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in native image regressions")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    agent = make_agent(native=True)
    agent._session_db = SessionDB(tmp_path / "error-transform.db")
    agent._ensure_db_session()
    seen, appended, persisted = [], [], []
    replacement = "Replacement without original native output."

    def rewrite(**payload):
        seen.append(payload["response_text"])
        if transform == "held_failure":
            raise RuntimeError("offline held redactor failure")
        return replacement

    def invoke(name, **payload):
        return [rewrite(**payload)] if name == "transform_llm_output" else []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    if transform != "ordinary":
        monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (rewrite,))
    original_append, original_persist = turn_final_response.append_message, agent._persist_session

    def append(messages, row, *args, **kwargs):
        appended.append(copy.deepcopy(row))
        return original_append(messages, row, *args, **kwargs)

    def persist(messages, history):
        persisted.extend(copy.deepcopy([row for row in messages if row.get("role") == "assistant"]))
        return original_persist(messages, history)

    monkeypatch.setattr(turn_final_response, "append_message", append)
    monkeypatch.setattr(agent, "_persist_session", persist)
    siblings = [image_item("bad", result=None)]
    if outcome == "mixed":
        siblings.insert(0, image_item("good"))
    elif "write_error" in outcome:
        siblings = [image_item("first"), image_item("second")]
        original_replace = os.replace
        publications = []

        def replace(src, dst, *args, **kwargs):
            if Path(dst).parent.name == "images":
                publications.append(dst)
                if outcome == "all_write_error" or len(publications) == 2:
                    raise OSError("offline atomic publication failure")
            return original_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", replace)
    try:
        result, requests = run_turn(agent, monkeypatch, response(text_item("Original."), *siblings))
        assert len(requests) == len(seen) == 1
        assert "Image output error:" in seen[0]
        paths = media_paths(seen[0])
        successful = outcome in {"mixed", "write_error"}
        assert len(paths) == (1 if successful else 0)
        assert all(path.is_file() for path in paths)
        expected = HELD_TRANSFORM_FAILED_REPLY if transform == "held_failure" else replacement
        assert result["final_response"] == expected
        assert not media_paths(result["final_response"])
        assert result["response_transformed"] is True
        assert result["pre_transform_response"] == (seen[0] if transform == "ordinary" else None)
        assert result["completed"] is False
        assert bool(result.get("partial")) is successful
        assert bool(result.get("failed")) is not successful
        assert result["failure_reason"] == "image_artifact_error"
        assert result["failure_retryable"] is False
        assert "image" in result["error"].lower()
        assert len(appended) == 1
        assert appended[0]["content"] == expected
        assert appended[0].get("display_kind") != "hidden"
        assert persisted and all(row["content"] == expected for row in persisted)
        assert result["messages"][-1]["content"] == expected
        stored = agent._session_db.get_messages_as_conversation(agent.session_id)
        assert stored[-1]["content"] == expected
        assert all("MEDIA:" not in row.get("content", "") for row in stored if row.get("role") == "assistant")
        assert set((tmp_path / "cache" / "generated" / "images").glob("*")) == set(paths)
    finally:
        agent._session_db.close()
