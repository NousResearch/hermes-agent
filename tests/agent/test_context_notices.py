"""Durable notices follow completed attempts, not compressor admission counters."""
import time
from types import SimpleNamespace

import pytest

from agent.conversation_compression import _emit_compression_attempt_telemetry
from hermes_state import SessionDB
from tui_gateway import server


def _agent(db, monkeypatch, events, session_id="conversation"):
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: events.append((kind, sid, payload)))
    callbacks = server._agent_cbs("runtime")
    return SimpleNamespace(
        _session_db=db, session_id=session_id,
        context_compressor=SimpleNamespace(_last_compression_telemetry={}),
        **{k: v for k, v in callbacks.items() if k.startswith("notice_")},
    )


def _outcome(agent, attempt_id, failure: str | None = "summary_generation_aborted", *, committed=False, fallback=False):
    agent._compression_attempt_id = attempt_id
    agent.context_compressor._last_compression_telemetry = {
        "attempt_id": attempt_id, "session_id": agent.session_id, "fallback_used": fallback,
    }
    _emit_compression_attempt_telemetry(
        agent, started_at=time.monotonic(), commit_status="committed" if committed else "aborted",
        split_status="in_place_committed" if committed else "aborted", failure_class=failure,
        captured_telemetry=dict(agent.context_compressor._last_compression_telemetry),
    )


@pytest.mark.parametrize("failure,committed,fallback", [
    ("summary_generation_aborted", False, False),
    ("summary_auth_failure", False, False),
    ("summary_network_failure", False, False),
    ("summary_truncated_failure", False, False),
    ("summary_empty_content_failure", False, False),
    ("summary_generation_failed", True, True),
    ("aux_model_fallback", True, True),
    ("session_split_failed", False, False),
    ("stall_interrupted", False, False),
])
def test_repeated_failures_survive_database_reopen(monkeypatch, tmp_path, failure, committed, fallback):
    path = tmp_path / "state.db"
    events = []
    db = SessionDB(db_path=path)
    db.create_session("conversation", source="gui")
    db.set_session_title("conversation", "Research")
    agent = _agent(db, monkeypatch, events)
    _outcome(agent, "first", failure, committed=committed, fallback=fallback)
    assert events == []  # A routine failure retains the existing immediate status only.
    db.close()

    db = SessionDB(db_path=path)
    try:
        agent = _agent(db, monkeypatch, events)
        for skip in ("explicit_interrupt", "commit_fence_cancelled", "no_progress", "attempt_superseded",
                     "pool_saturated", "compression_lock_busy", "feasibility_skip", "unknown_failure"):
            _outcome(agent, skip, skip, fallback=True)
        assert events == []
        _outcome(agent, "second", failure, committed=committed, fallback=fallback)
        assert len(events) == 1, "a second real failure after restart must show a durable notice"
        kind, sid, payload = events.pop()
        assert (kind, sid, payload["kind"], payload["level"]) == ("notification.show", "runtime", "sticky", "warn")
        assert "Research" in payload["text"] and "conversation" in payload["text"]
        key = payload["key"]
        _outcome(agent, "second", failure, committed=committed, fallback=fallback)
        assert events == []  # Delivery/telemetry retries are not new failures.
        _outcome(agent, "fallback", failure=None, committed=True, fallback=True)
        assert events == []  # A fallback is not proof the summary route recovered.
        _outcome(agent, "third", failure="summary_network_failure")
        assert len(events) == 1, "ongoing failures update the same notice rather than freezing at two"
        assert events[0][2]["key"] == key
        assert "3" in events[0][2]["text"] and "network" in events[0][2]["text"].lower()
        events.clear()
        _outcome(agent, "recovery", failure=None, committed=True)
        assert events[0][:2] == ("notification.clear", "runtime")
        assert events[0][2] == {"key": key, "state_key": key,
                                "state_revision": db.get_context_notice_state("conversation")["revision"]}
        assert len(events) == 2 and events[1][0] == "notification.show"
        assert events[1][2]["kind"] == "ttl" and events[1][2]["ttl_ms"] > 0
    finally:
        db.close()


@pytest.mark.parametrize("entry", ["resume", "reconnect", "rebuild"])
def test_notice_replays_without_resident_agent(monkeypatch, tmp_path, entry):
    path = tmp_path / "profile" / "state.db"
    events = []
    db = SessionDB(db_path=path)
    db.create_session("conversation", source="gui")
    # A pre-feature database is reconciled on open, not through a manual migration.
    db._write_sql("ALTER TABLE sessions DROP COLUMN context_notice_state")
    db.close()
    db = SessionDB(db_path=path)
    agent = _agent(db, monkeypatch, events)
    _outcome(agent, "first")
    _outcome(agent, "second")
    key = events[-1][2]["key"]
    db.close()
    db = SessionDB(db_path=path)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_sessions", {})
    events.clear()
    try:
        if entry == "resume":
            result = server._methods["session.resume"](1, {"session_id": "conversation", "lazy": True})
            assert "error" not in result, result
            sid = result["result"]["session_id"]
            assert server._sessions[sid].get("agent") is None
        else:
            sid = "runtime"
            server._sessions[sid] = {"session_key": "conversation", "agent": None}
            if entry == "reconnect":
                server._methods["session.events.since"](1, {"session_id": sid, "last_seen": 0})
            else:
                server._wire_callbacks(sid)
        shows = [e for e in events if e[0] == "notification.show"]
        assert len(shows) == 1, f"{entry} must replay durable notices even without an agent"
        assert shows[0][1] == sid and shows[0][2]["key"] == key
    finally:
        db.close()
