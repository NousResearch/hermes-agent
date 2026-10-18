"""Short reconstructed workers share cadence, never transcripts or profiles."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import codex_runtime
from hermes_state import SessionDB


def _agent(db, sid, receipts, interval=3, **extra):
    return SimpleNamespace(
        session_id=sid, _session_db=db, _skill_nudge_interval=interval,
        _iters_since_skill=0, valid_tool_names={"skill_manage"},
        _spawn_background_review=lambda **kw: receipts.append((sid, kw)),
        _sync_external_memory_for_turn=lambda **kw: None, **extra,
    )


def _turn(monkeypatch):
    monkeypatch.setattr(codex_runtime, "_record_codex_app_server_compaction", lambda *a: None)
    monkeypatch.setattr(codex_runtime, "_record_codex_app_server_usage", lambda *a, **kw: {})
    return SimpleNamespace(tool_iterations=1, final_text="done", interrupted=False, error=None)


@pytest.mark.parametrize("source", ["kanban", "desktop", "cli", "telegram"])
def test_reconstructed_short_turns_eventually_review_only_their_snapshot(tmp_path, monkeypatch, source):
    turn = _turn(monkeypatch)
    receipts = []
    path = tmp_path / "state.db"
    for index in range(3):
        db = SessionDB(path)
        sid = f"card-{index}" if source == "kanban" else "same-conversation"
        try:
            db.create_session(sid, source)
            agent = _agent(db, sid, receipts)
            snapshot = [{"role": "assistant", "content": f"task {index}"}]
            codex_runtime._finish_codex_turn(agent, turn, snapshot, original_user_message="work", should_review_memory=False)
        finally:
            db.close()
    assert len(receipts) == 1, "cadence restarted with each reconstructed agent"
    assert receipts[0][1]["messages_snapshot"] == snapshot
    other = SessionDB(tmp_path / "other-profile" / "state.db")
    try:
        other.create_session(sid, source)
        codex_runtime._finish_codex_turn(_agent(other, sid, receipts), turn, [], original_user_message="work",
                                        should_review_memory=False)
        assert len(receipts) == 1
    finally:
        other.close()


def test_atomic_claim_refund_optouts_and_compression(tmp_path, monkeypatch):
    from agent.review_cadence import schedule_turn_review, reset_skill_review_cadence
    turn = _turn(monkeypatch)
    path = tmp_path / "state.db"
    dbs = [SessionDB(path) for _ in range(4)]
    receipts = []
    try:
        def cards(i):
            db = dbs[i]
            for n in range(10):
                sid = f"{i}-{n}"
                db.create_session(sid, "kanban")
                codex_runtime._finish_codex_turn(_agent(db, sid, receipts, 10), turn, [],
                                                original_user_message="work", should_review_memory=False)
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(cards, range(4)))
        assert len(receipts) == 4
        db = dbs[0]
        db.create_session("parent", "desktop")
        agent = _agent(db, "parent", receipts, 2)
        agent._iters_since_skill = 1
        schedule_turn_review(agent, [], final_response="done", interrupted=False)
        db.end_session("parent", "compression")
        db.create_session("tip", "desktop", parent_session_id="parent")
        resumed = _agent(db, "tip", receipts, 2)
        resumed._iters_since_skill = 1
        resumed._spawn_background_review = lambda **kw: False
        schedule_turn_review(resumed, [], final_response="done", interrupted=False)
        resumed._spawn_background_review = lambda **kw: receipts.append(("tip", kw))
        schedule_turn_review(resumed, [], final_response="done", interrupted=False)
        assert len(receipts) == 5 and receipts[-1][0] == "tip", "rejected admission consumed the claim"
        for extra in ({"skip_background_review": True}, {"_delegate_depth": 1}, {"_persist_disabled": True}):
            blocked = _agent(db, "tip", receipts, 1, **extra)
            blocked._iters_since_skill = 50
            schedule_turn_review(blocked, [], final_response="done", interrupted=False, review_memory=True)
        assert len(receipts) == 5
        resumed._iters_since_skill = 2
        schedule_turn_review(resumed, [], final_response="partial", interrupted=True)
        assert len(receipts) == 5
        reset_skill_review_cadence(resumed)
        schedule_turn_review(resumed, [], final_response="done", interrupted=False)
        assert len(receipts) == 5, "foreground skill use must reset only its own session"
    finally:
        for db in dbs:
            db.close()
