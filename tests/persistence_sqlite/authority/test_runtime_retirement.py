"""``SessionRuntimeRepository`` on SQLite: the viewer closes, the policy refresh and the two ledger-gated ends each open
exactly one write transaction, which a write that fails right after its first mutating statement leaves unchanged.
"""

from __future__ import annotations

from typing import Any

from persistence.authority.bindings import local_session_id
from tests.persistence_sqlite.authority._admission_support import admit, begin_epoch, claim, settle
from tests.persistence_sqlite.authority.test_binding_local import make_binding, reserved as reserving
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_write, build_seed

PROFILE = "default"
PRINCIPAL = "user-1"
OWNER = "owner-a"
SID = str(local_session_id(PROFILE, PRINCIPAL, "req-1"))


def seed(store: Any, *steps: Any) -> None:
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        for step in steps:
            step(db)
    build_seed(store, run)


def reserved(db: Any) -> None:
    """Reserve the local session ``SID`` with the policy ``_POLICY``."""
    reserving(make_binding(SID, request_json="{}"))(db)


# ── viewer closes ─────────────────────────────────────────────────────────────


def test_viewer_close_writes_are_one_write(store):
    seed(store)
    assert_one_write(store, lambda: store.runtime.store_viewer_close(1, session_id="s-1", record="record-1"))
    assert_one_write(store, lambda: store.runtime.forget_viewer_close(1, session_id="s-1", record="record-1"))


# ── the policy refresh ────────────────────────────────────────────────────────


_POLICY = {"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": "{}"}


def test_commit_refreshed_policy_is_one_write(store):
    seed(store, reserved)
    assert_one_write(store, lambda: store.runtime.commit_refreshed_policy(
        1, session_id=SID, old=_POLICY, new={**_POLICY, "model": "model-b"}, generation=0))


# ── end_unless_ledger_busy ────────────────────────────────────────────────────


def test_end_unless_ledger_busy_is_one_write(store):
    seed(store, lambda db: db.create_session("s-1", source="cli"))
    assert_one_write(store, lambda: store.runtime.end_unless_ledger_busy("s-1", session_key=None, end_reason="tui_close"))


# ── end_after_settlement ──────────────────────────────────────────────────────


def _settled(db: Any, *, pending: bool = False) -> dict:
    db.create_session("s-hook", source="webhook")
    admitted = admit(db, "s-hook", "in-1", payload={"text": "hi"}, principal_id="webhook")
    claimed = claim(db, "s-hook")
    settle(db, claimed, result={"ok": True})
    if pending:
        admit(db, "s-hook", "in-2", payload={"text": "again"}, principal_id="webhook")
    return {"admission_id": admitted["admission_id"], "generation": claimed["generation"]}


def test_end_after_settlement_is_one_write(store):
    found: dict = {}
    seed(store, lambda db: found.update(_settled(db)))
    assert_one_write(store, lambda: store.runtime.end_after_settlement(
        1, session_id="s-hook", admission_id=found["admission_id"], generation=found["generation"], admission_epoch=1,
        reason="webhook_complete"))
