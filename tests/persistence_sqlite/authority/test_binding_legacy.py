"""``SessionBindingRepository.reserve`` on SQLite, the legacy arm: binding an unrouted pre-authority row to its new receipt
is one write that a failure right after its first mutating statement leaves unchanged.
"""

from __future__ import annotations

from typing import Any

from persistence.authority.bindings import NewSessionBinding
from persistence.core.identity import SessionId
from tests.persistence_sqlite.authority._admission_support import begin_epoch
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_write, build_seed

PRINCIPAL = "user-1"
OWNER = "owner-a"
LEGACY = "20260101_120000_abcdef"
ROUTE = f"agent:main:local:dm:{LEGACY}"
STAMP = "2026-01-01T12:00:00"


def legacy_binding(*, principal: str = PRINCIPAL, legacy_session_id: str = LEGACY) -> NewSessionBinding:
    entry = {"session_key": ROUTE, "session_id": LEGACY, "created_at": STAMP, "updated_at": STAMP,
             "origin": {"chat_id": LEGACY, "chat_type": "dm", "user_id": principal, "platform": "local"}}
    return NewSessionBinding(
        profile_id="default", principal_id=principal, request_id="adopt-1", session_id=SessionId(LEGACY), route=ROUTE,
        entry=entry, policy={"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": "{}"},
        created_source="cli", legacy_session_id=legacy_session_id, lineage=(SessionId(LEGACY),))


def seed(store: Any) -> None:
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        db.create_session(LEGACY, source="cli")
    build_seed(store, run)


def test_reserve_legacy_arm_is_one_write(store):
    seed(store)
    assert_one_write(store, lambda: store.bindings.reserve(epoch=1, binding=legacy_binding(), owner=None))
