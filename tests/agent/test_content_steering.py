"""Safe native correction boundaries and single-owner recovery."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from agent.image_routing import build_native_content_parts
from tests.agent.test_tool_call_incremental_persistence import _attach_real_session_db, _durable_messages, _mock_response

pytest_plugins = ["tests.gateway.test_native_image_steering"]


@pytest.mark.parametrize("streaming", [False, True])
def test_final_candidate_continues_one_logical_turn(setup, tmp_path, streaming):
    from tests.agent.test_run_agent import _make_chunk
    from agent.turn_finalizer import finalize_turn

    _, agent, images = setup
    db_path = tmp_path / "state.db"
    db = _attach_real_session_db(agent, db_path, "final-correction")
    content, _ = build_native_content_parts("Use the new diagram", [str(images[0])])
    requests, accepted, deltas = [], [], []
    if streaming:
        agent.stream_delta_callback = deltas.append

    def respond(**kwargs):
        requests.append(deepcopy(kwargs))
        answer = "First candidate" if len(requests) == 1 else "Revised answer"
        if len(requests) == 1:
            accepted.append(agent.steer_content(content, correction_id="correction"))
        if streaming:
            return iter([_make_chunk(content=answer), _make_chunk(finish_reason="stop")])
        return _mock_response(content=answer)

    agent.client.chat.completions.create.side_effect = respond
    with patch("agent.conversation_loop.finalize_turn", autospec=True, side_effect=finalize_turn) as finalizer:
        result = agent.run_conversation("Inspect the diagram")
    db.close()
    assert accepted == [True]
    assert result["final_response"] == "Revised answer"
    assert len(requests) == 2
    assert finalizer.call_count == 1
    assert requests[1]["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]
    assert requests[1]["messages"][-2:] == [
        {"role": "assistant", "content": "First candidate"}, {"role": "user", "content": content},
    ]
    replay = _durable_messages(db_path, agent.session_id)
    assert [m["role"] for m in replay] == ["user", "assistant", "user", "assistant"]
    assert "Use the new diagram" in replay[-2]["content"]
    assert "[screenshot]" in replay[-2]["content"]
    assert not agent.withdraw_steer("correction")
    assert not agent.steer_content(content, correction_id="late")


@pytest.mark.parametrize("outcome", ["budget", "interrupt", "persist-false", "persist-error", "persist-fatal"])
def test_unconsumed_corrections_return_once_without_durable_ghost(setup, tmp_path, monkeypatch, outcome):
    _, agent, images = setup
    db_path = tmp_path / "recovery.db"
    db = _attach_real_session_db(agent, db_path, "recoverable-correction")
    content, _ = build_native_content_parts("Use the updated sketch", [str(images[0])])
    admissions = []
    if outcome == "budget":
        agent.max_iterations = 1
    flush = agent._flush_messages_to_session_db
    def fail_correction(messages, history):
        if any((m.get("display_metadata") or {}).get("content_steer_ids") for m in messages):
            if outcome == "persist-false":
                return False
            if outcome == "persist-error":
                raise OSError("synthetic storage failure")
            if outcome == "persist-fatal":
                raise SystemExit("synthetic exceptional exit")
        return flush(messages, history)
    monkeypatch.setattr(agent, "_flush_messages_to_session_db", fail_correction)
    def respond(**kwargs):
        admissions.append(agent.steer_content(content, correction_id="recoverable"))
        admissions.append(agent.steer_content(content, correction_id="recoverable"))
        if outcome == "interrupt":
            agent.interrupt()
        return _mock_response(content="Initial candidate")
    agent.client.chat.completions.create.side_effect = respond
    if outcome == "persist-fatal":
        with pytest.raises(SystemExit, match="synthetic exceptional exit"):
            agent.run_conversation("Inspect the original sketch")
        pending = agent.withdraw_steer("recoverable")
        # A later cleanup flush must not resurrect a rolled-back correction.
        flush(agent._session_messages, [])
    else:
        result = agent.run_conversation("Inspect the original sketch")
        pending = agent.withdraw_steer("recoverable")
        if outcome.startswith("persist"):
            assert result["failed"] and "Initial candidate" not in result["final_response"]
    db.close()
    assert admissions == [True, True]
    assert pending is True
    assert not agent.withdraw_steer("recoverable")
    assert not agent.steer_content(content, correction_id="too-late")
    assert agent.client.chat.completions.create.call_count == 1
    replay = _durable_messages(db_path, agent.session_id)
    assert all(not isinstance(m["content"], list) for m in replay)



@pytest.mark.parametrize("uncertainty", ["committed", "unreadable"])
def test_uncertain_transcript_commit_is_never_replayed(setup, tmp_path, monkeypatch, uncertainty):
    _, agent, images = setup
    db = _attach_real_session_db(agent, tmp_path / "uncertain.db", "uncertain-correction")
    content, _ = build_native_content_parts("Replacement sketch", [str(images[0])])
    flush = agent._flush_messages_to_session_db
    def uncertain(messages, history):
        if any((m.get("display_metadata") or {}).get("content_steer_ids") for m in messages):
            if uncertainty == "committed":
                flush(messages, history)
            else:
                def unreadable(*a, **kw):
                    raise OSError("synthetic read failure")
                monkeypatch.setattr(db, "get_messages", unreadable)
            raise OSError("synthetic lost commit acknowledgement")
        return flush(messages, history)
    monkeypatch.setattr(agent, "_flush_messages_to_session_db", uncertain)
    def respond(**kwargs):
        assert agent.steer_content(content, correction_id="uncertain")
        return _mock_response(content="Candidate")
    agent.client.chat.completions.create.side_effect = respond
    result = agent.run_conversation("Inspect the sketch")
    assert result["failed"]
    assert agent.client.chat.completions.create.call_count == 1
    assert not agent.withdraw_steer("uncertain"), "uncertain work must settle interrupted, never auto-replay"
    if uncertainty == "committed":
        rows = db.get_messages(agent.session_id)
        assert sum("uncertain" in (m.get("display_metadata") or {}).get("content_steer_ids", []) for m in rows) == 1
    db.close()
