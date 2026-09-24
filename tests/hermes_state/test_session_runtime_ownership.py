"""SQLite state-machine evidence for the additive handoff foundation."""
from __future__ import annotations

import time

import pytest

from hermes_state import SessionDB
from hermes_state_runtime_ownership import RuntimeOwner


def _owner(name: str) -> RuntimeOwner:
    return RuntimeOwner(f"installation-{name}", f"host-{name}", f"generation-{name}")


def _expire(db: SessionDB, session_id: str) -> None:
    db._execute_write(lambda conn: conn.execute(
        "UPDATE session_runtime_owners SET expires_at = ? WHERE session_id = ?", (time.time() - 1, session_id)
    ))


def _turn(db: SessionDB, session_id: str, turn_id: str):
    owner = db.acquire_session_runtime_ownership(session_id, _owner("one"), ttl_seconds=5)
    assert owner is not None
    assert db.begin_session_runtime_turn(owner, turn_id)
    return owner


def test_same_machine_restart_renews_stable_owner_and_turn_is_idempotent(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first = db.acquire_session_runtime_ownership("s", _owner("one"), ttl_seconds=5)
    assert first is not None and first.fence == 1
    restarted = db.acquire_session_runtime_ownership("s", _owner("one"), ttl_seconds=5)
    assert restarted is not None and restarted.fence == first.fence
    assert db.begin_session_runtime_turn(restarted, "turn-1")
    assert db.begin_session_runtime_turn(restarted, "turn-1")


def test_two_owners_contend_then_expiry_takeover_indeterminates_running_turn(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first = _turn(db, "s", "turn-1")
    assert db.acquire_session_runtime_ownership("s", _owner("two"), ttl_seconds=5) is None
    _expire(db, "s")
    second = db.acquire_session_runtime_ownership("s", _owner("two"), ttl_seconds=5)
    assert second is not None and second.fence == first.fence + 1
    with db._read_ctx() as conn:
        row = conn.execute("SELECT state FROM session_runtime_turns WHERE session_id = ? AND turn_id = ?", ("s", "turn-1")).fetchone()
    assert row["state"] == "indeterminate"


def test_stale_fence_cannot_renew_release_or_resolve_after_takeover(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first = _turn(db, "s", "turn-1")
    _expire(db, "s")
    second = db.acquire_session_runtime_ownership("s", _owner("two"), ttl_seconds=5)
    assert second is not None
    assert db.renew_session_runtime_ownership(first) is None
    assert not db.release_session_runtime_ownership(first)
    assert not db.resolve_session_runtime_turn(first, "turn-1", state="indeterminate")


def test_crash_during_turn_is_indeterminate_then_new_owner_explicitly_resolves(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first = _turn(db, "s", "turn-1")
    _expire(db, "s")
    second = db.acquire_session_runtime_ownership("s", _owner("two"), ttl_seconds=5)
    assert second is not None
    assert db.resolve_session_runtime_turn(second, "turn-1", state="settled", receipt_data={"verified": True})
    with db._read_ctx() as conn:
        assert conn.execute("SELECT state FROM session_runtime_turns WHERE session_id = ? AND turn_id = ?", ("s", "turn-1")).fetchone()["state"] == "settled"
    assert db.begin_session_runtime_turn(second, "turn-2")


def test_settled_requires_receipt_and_process_local_capabilities_are_excluded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    receipt = _turn(db, "s", "turn-1")
    with pytest.raises(ValueError, match="verified receipt"):
        db.resolve_session_runtime_turn(receipt, "turn-1", state="settled")
    assert not db.supports_session_runtime_handoff_capability("browser")
    assert not db.supports_session_runtime_handoff_capability("computer_use")
    assert db.supports_session_runtime_handoff_capability("durable_transcript")
