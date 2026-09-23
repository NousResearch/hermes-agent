"""Tests for handoff-owned recovery: a session whose durable row was stamped
with a NON-recoverable end_reason by the teardown that ran *after* a completed
``/handoff`` still resolves to its durable row.

Incident shape (QQ DM, single ``session_key``, 2026-09-22):

  21:33  qqbot session B starts (253 messages)
  22:29  user runs ``/handoff qqbot`` from the CLI for B -> gateway claims the
         row, ``handoff_state='completed'``, synthetic turn dispatched
  22:29  the CLI teardown then stamps B ``end_reason='cli_close'`` (the
         handed-off-id guard is missing on that one teardown path)
  22:54  user's next DM: routing still points at B, the stale-route self-heal
         finds ``cli_close`` (neither recoverable nor a reset boundary) and
         silently mints a brand-new session C — the handed-off leg is lost

The CLI-side guard is fixed separately. These tests pin the recovery-side
contract the incident exposes: a row a handoff completed may only be dropped by
an end_reason stamped AFTER the handoff, so the teardown race cannot orphan it.
"""

import time

import pytest

from hermes_state import SessionDB

PEER = dict(
    source="qqbot",
    user_id="D425866BDDB69A1305199BAE0B166AFB",
    session_key="agent:main:qqbot:dm:D425866BDDB69A1305199BAE0B166AFB",
    chat_id="D425866BDDB69A1305199BAE0B166AFB",
    chat_type="dm",
    thread_id=None,
)


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _mk_handed_off(db, session_id, *, ended_at=None, end_reason=None, handoff_state="completed",
                   msgs=3, last_activity_at=None):
    db.create_session(
        session_id, PEER["source"], user_id=PEER["user_id"],
        session_key=PEER["session_key"], chat_id=PEER["chat_id"], chat_type=PEER["chat_type"],
    )
    for i in range(msgs):
        db.append_message(session_id, "user" if i % 2 == 0 else "assistant", f"m{i}")
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET handoff_state=?, handoff_platform='qqbot', last_activity_at=? "
            "WHERE id=?",
            (handoff_state, last_activity_at or time.time(), session_id),
        )
        if end_reason is not None:
            db._conn.execute(
                "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?", (ended_at, end_reason, session_id))
        db._conn.commit()
    return session_id


def _find(db):
    return db.find_latest_gateway_session_for_peer(**PEER)


def test_cli_close_after_completed_handoff_is_recoverable(db):
    """The incident: handoff completed, then the CLI teardown stamps cli_close."""
    _mk_handed_off(db, "handoff-leg", ended_at=1000.0, end_reason="cli_close")
    assert _find(db)["id"] == "handoff-leg"


def test_plain_cli_close_without_handoff_is_not_recoverable(db):
    """Control: the widened contract is handoff-only — an ordinary CLI exit stays closed."""
    db.create_session(
        "plain-cli", PEER["source"], user_id=PEER["user_id"],
        session_key=PEER["session_key"], chat_id=PEER["chat_id"], chat_type=PEER["chat_type"],
    )
    db.append_message("plain-cli", "user", "hi")
    db.end_session("plain-cli", "cli_close")
    assert _find(db) is None


def test_explicit_reset_after_handoff_still_fences(db):
    """A deliberate boundary stamped after the handed-off row's last activity still blocks recovery."""
    _mk_handed_off(db, "handoff-then-new", ended_at=1000.0, end_reason="cli_close",
                   last_activity_at=1000.0)
    db.create_session(
        "later-new", PEER["source"], user_id=PEER["user_id"],
        session_key=PEER["session_key"], chat_id=PEER["chat_id"], chat_type=PEER["chat_type"],
    )
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET started_at=500.0, ended_at=3000.0, end_reason='session_reset' WHERE id=?",
            ("later-new",))
        db._conn.commit()
    assert _find(db) is None


def test_live_row_outranks_stale_handed_off_row(db):
    """Widening recovery must not resurrect an OLD handed-off row over the live one for the key."""
    _mk_handed_off(db, "old-handoff-leg", ended_at=1000.0, end_reason="cli_close",
                   last_activity_at=1000.0)
    db.create_session(
        "live-now", PEER["source"], user_id=PEER["user_id"],
        session_key=PEER["session_key"], chat_id=PEER["chat_id"], chat_type=PEER["chat_type"],
    )
    db.append_message("live-now", "user", "current thread")
    with db._lock:
        db._conn.execute("UPDATE sessions SET last_activity_at=? WHERE id=?", (time.time(), "live-now"))
        db._conn.commit()
    assert _find(db)["id"] == "live-now"


def test_failed_handoff_does_not_widen_recovery(db):
    """``handoff_state='failed'`` is not a handoff that owns the row."""
    _mk_handed_off(db, "failed-handoff", ended_at=1000.0, end_reason="cli_close",
                   handoff_state="failed")
    assert _find(db) is None


def test_pending_handoff_does_not_widen_recovery(db):
    """A handoff still in flight does not own the row yet."""
    _mk_handed_off(db, "pending-handoff", ended_at=1000.0, end_reason="cli_close",
                   handoff_state="pending")
    assert _find(db) is None
