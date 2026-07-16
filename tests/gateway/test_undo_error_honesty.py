"""Honesty of /undo·/redo failure reporting (2026-07-15 fix).

The false-"Nothing to undo." bug: rewind_session collapsed EVERY exception into
None → "Nothing to undo." with only a DEBUG log. These tests pin the four
distinct outcomes and the pass-1 B1 no-double-undo invariant.
"""
import sqlite3
from pathlib import Path

import pytest

import hermes_undo
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, _is_transient_db_busy
from hermes_state import SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    session_store._db = db
    hermes_undo._session_db = db
    hermes_undo.clear_state()
    yield session_store
    db.close()


def _source(chat_id="chat-1"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")


def _seed(db, sid):
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "a2")


# ── the classifier ──────────────────────────────────────────────────────────

def test_is_transient_db_busy_only_lock_class():
    assert _is_transient_db_busy(sqlite3.OperationalError("database is locked"))
    assert _is_transient_db_busy(sqlite3.OperationalError("database table is busy"))
    assert not _is_transient_db_busy(sqlite3.OperationalError("no such table: messages"))
    assert not _is_transient_db_busy(ValueError("rewind would orphan active tool row"))


# ── the four outcomes of rewind_session ─────────────────────────────────────

def test_genuine_empty_returns_none(store):
    """AC2/I1: a truly empty session → None → 'Nothing to undo.' (unchanged)."""
    entry = store.get_or_create_session(_source("empty"))
    # session exists but no messages → nothing to undo
    assert store.rewind_session(entry.session_id, 1) is None


def test_lock_error_returns_busy_sentinel(store, monkeypatch):
    """AC3/I3: a lock/busy OperationalError from the undo core → busy sentinel,
    NOT None (which would render 'Nothing to undo.')."""
    def boom(*a, **kw):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(hermes_undo, "undo", boom)
    out = store.rewind_session("any-sid", 1)
    assert out == {"status": "busy"}, out


def test_orphan_guard_is_retryable_busy_not_error(store, monkeypatch):
    """AC4/RC3: the orphan-guard RewindWouldOrphanError (the most likely trigger
    of the 2026-07-15 incident) is a TRANSIENT, self-healing condition — it must
    map to the retryable 'busy' outcome (WARNING), NOT a permanent 'error'
    (ERROR-logged). Reporting a mid-flush race as a permanent internal fault is
    both dishonest and alert-noise."""
    from hermes_state import RewindWouldOrphanError

    def boom(*a, **kw):
        raise RewindWouldOrphanError("rewind would orphan active tool row id=5 ...")
    monkeypatch.setattr(hermes_undo, "undo", boom)
    out = store.rewind_session("any-sid", 1)
    assert out == {"status": "busy"}, out


def test_non_lock_operationalerror_returns_error_not_busy(store, monkeypatch):
    """AC4/B3: a non-lock OperationalError (schema/logic bug) must NOT be
    laundered into the transient 'busy' path."""
    def boom(*a, **kw):
        raise sqlite3.OperationalError("no such table: messages")
    monkeypatch.setattr(hermes_undo, "undo", boom)
    out = store.rewind_session("any-sid", 1)
    assert out == {"status": "error"}, out


def test_generic_bug_still_returns_error(store, monkeypatch):
    """A genuine bug (not the orphan guard, not a lock) → error, ERROR-logged —
    never masked as a transient 'busy'."""
    def boom(*a, **kw):
        raise KeyError("some real bug")
    monkeypatch.setattr(hermes_undo, "undo", boom)
    assert store.rewind_session("any-sid", 1) == {"status": "error"}


def test_healthy_rewind_returns_real_result(store):
    """A genuine rewindable session → a real result dict with rewound_ids
    (never a status sentinel)."""
    entry = store.get_or_create_session(_source("live"))
    sid = entry.session_id
    _seed(store._db, sid)
    out = store.rewind_session(sid, 1)
    assert isinstance(out, dict) and out.get("rewound_ids"), out
    assert "status" not in out


# ── restore_session (redo) parity ───────────────────────────────────────────

def test_restore_lock_error_returns_busy(store, monkeypatch):
    def boom(*a, **kw):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(hermes_undo, "redo", boom)
    assert store.restore_session("any-sid", 1) == {"status": "busy"}


def test_restore_other_error_returns_error(store, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(hermes_undo, "redo", boom)
    assert store.restore_session("any-sid", 1) == {"status": "error"}


# ── B1 (pass-1 blocker): post-commit read failure must NOT double-undo ───────

def test_post_commit_head_read_failure_still_returns_committed_rewind(store, monkeypatch):
    """AC5/I2/B1: if the POST-commit head-id read raises, the rewind already
    committed — rewind_to_message must return the committed result (rewound rows
    durable) with new_head_id=None, NOT raise. A raise would reach
    rewind_session → error sentinel → user retries → DOUBLE undo."""
    db = store._db
    entry = store.get_or_create_session(_source("b1"))
    sid = entry.session_id
    _seed(db, sid)

    active_before = len(db.get_messages(sid, include_inactive=False))

    target = hermes_undo.compute_half_turn_target(
        db.get_messages(sid, include_inactive=False), 1
    )

    # Wrap the connection so ONLY the post-write MAX(id) read raises; the write
    # and validation reads pass through. (Can't monkeypatch the C-level
    # Connection.execute, so proxy the object.)
    real_conn = db._conn
    state = {"armed": False}

    class _FlakyConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a, **kw):
            if state["armed"] and "MAX(id)" in sql:
                raise sqlite3.OperationalError("database is locked")
            return self._inner.execute(sql, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    db._conn = _FlakyConn(real_conn)
    state["armed"] = True
    try:
        result = db.rewind_to_message(sid, target, require_user_role=False)
    finally:
        state["armed"] = False
        db._conn = real_conn

    # The rewind committed: rows were deactivated, result returned (no raise),
    # head id is None (fail-soft).
    assert result["rewound_ids"], "rewind must have committed"
    assert result["new_head_id"] is None
    active_after = len(db.get_messages(sid, include_inactive=False))
    assert active_after < active_before, "rows really deactivated (durable)"


def test_redo_second_op_failure_returns_partial_not_nothing_changed(store):
    """AC5 redo symmetry (pass-2 B1): redo does ONE write PER op. If a LATER op's
    write raises after an EARLIER op committed, the earlier rows are live in the
    DB — restore_session must NOT report 'nothing changed' (that would let a
    retry double-redo the earlier ops). The partial result is returned honestly.
    """
    db = store._db
    entry = store.get_or_create_session(_source("redo-partial"))
    sid = entry.session_id
    # Two separate undo ops on the stack → redo m=2 will do two restore_ids writes.
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "a2")
    r1 = store.rewind_session(sid, 1)  # op A
    r2 = store.rewind_session(sid, 1)  # op B
    assert r1 and r2

    # Make the SECOND restore_ids write raise; the first must commit.
    calls = {"n": 0}
    real_restore = db.restore_ids

    def flaky_restore(session_id, ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return real_restore(session_id, ids)

    db.restore_ids = flaky_restore
    try:
        out = store.restore_session(sid, 2)
    finally:
        db.restore_ids = real_restore

    # NOT a busy/error sentinel: the first op committed, so we report the partial
    # honestly (reactivated_count > 0), never "nothing changed".
    assert isinstance(out, dict), out
    assert out.get("status") not in ("busy", "error"), out
    assert (out.get("reactivated_count") or 0) > 0, (
        "the first (committed) redo op must be reported, not discarded"
    )
    # pass-3 B2: the FAILED op must be pushed back and NOT wiped — it stays
    # recoverable. Retrying /redo (with the DB healthy) redoes it.
    assert out.get("partial_retryable") is True, out
    out2 = store.restore_session(sid, 1)
    assert isinstance(out2, dict) and (out2.get("reactivated_count") or 0) > 0, (
        "the transient-failed redo op must be recoverable on retry (not lost)"
    )


def test_redo_nontransient_op2_failure_not_labeled_retryable(store):
    """pass-4 RC1: a NON-transient error on redo op≥2 after op1 committed must
    report the partial as NOT retryable (retrying re-fails identically). The
    redo loop must classify inside itself, not blanket-label every mid-loop
    failure transient."""
    db = store._db
    entry = store.get_or_create_session(_source("redo-hard"))
    sid = entry.session_id
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "a2")
    assert store.rewind_session(sid, 1)
    assert store.rewind_session(sid, 1)

    calls = {"n": 0}
    real_restore = db.restore_ids

    def flaky_restore(session_id, ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("no such column: bogus")  # NON-transient
        return real_restore(session_id, ids)

    db.restore_ids = flaky_restore
    try:
        out = store.restore_session(sid, 2)
    finally:
        db.restore_ids = real_restore

    assert isinstance(out, dict), out
    assert (out.get("reactivated_count") or 0) > 0, "op1 committed, must be reported"
    assert out.get("partial_retryable") is False, (
        "a non-transient bug must NOT be labeled retryable (would loop forever)"
    )


def test_empty_with_rows_incident_signature_warns(store, monkeypatch, caplog):
    """AC1/B2/RC2/RC3: a genuine-empty result with rows PRESENT and no target
    (the 2026-07-15 incident signature) is surfaced at WARNING (visible in prod)
    with the active_count — not a silent DEBUG."""
    import logging

    def fake_undo(session_id, n):
        return {
            "rewound_ids": [],
            "message": "nothing to undo",
            "active_count": 365,
            "target_id": None,
        }
    monkeypatch.setattr(hermes_undo, "undo", fake_undo)
    with caplog.at_level(logging.DEBUG, logger="gateway.session"):
        out = store.rewind_session("any-sid", 1)
    assert out is None
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    joined = " ".join(r.getMessage() for r in warns)
    assert "365 active row" in joined, joined


def test_true_healthy_empty_stays_debug(store, monkeypatch, caplog):
    """A genuinely empty session (0 active rows) stays at DEBUG — no WARNING
    noise for the normal 'nothing to undo' case."""
    import logging

    def fake_undo(session_id, n):
        return {"rewound_ids": [], "active_count": 0, "target_id": None}
    monkeypatch.setattr(hermes_undo, "undo", fake_undo)
    with caplog.at_level(logging.DEBUG, logger="gateway.session"):
        assert store.rewind_session("any-sid", 1) is None
    assert not [r for r in caplog.records if r.levelno == logging.WARNING], (
        "a true healthy empty must not WARN"
    )


