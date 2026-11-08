"""``SessionBindingRepository.reserve`` on SQLite against main: the legacy arm, which binds an unrouted pre-authority row
to its new receipt inside the reservation's write (main's hermes_state_local_migration.bind_legacy_target), refusing it
while the conversation's turn lease is live.

Each case runs main's ``commit_local_session`` and the verb on copies of one seed (same outcome, same rows), and the
reservation stays one write that a failure after its first mutating statement leaves unchanged.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from persistence.authority.bindings import NewSessionBinding
from persistence.core.identity import SessionId
from tests.persistence_sqlite.authority._admission_support import begin_epoch
from tests.persistence_sqlite.authority._support import assert_same, run_both
from tests.persistence_sqlite.authority.test_binding_local import as_record, legacy_receipt
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


def seed(store: Any, *steps: Any, legacy: bool = True) -> None:
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        db.create_session(LEGACY, source="cli")
        for step in steps:
            step(db)
    build_seed(store, run, legacy=legacy)


def raw(sql: str, *params: Any):
    def step(db: Any) -> None:
        with db._lock:
            db._conn.execute(sql, params)
            db._conn.commit()
    return step


def _cases():
    from hermes_state_local_migration import LEGACY_PREFIX
    from hermes_state_mutation_binding import BINDING_PREFIX
    from hermes_state_runtime import admit_session_input

    def compressed(db: Any) -> None:
        db.create_session("cmp-1", source="cli", parent_session_id=LEGACY)
        raw("UPDATE sessions SET ended_at=1.0, end_reason='compression' WHERE id=?", LEGACY)(db)

    return {
        "adopted": ([], legacy_binding()),
        "own_stamp": ([raw("UPDATE sessions SET user_id=? WHERE id=?", PRINCIPAL, LEGACY)], legacy_binding()),
        "foreign_owner": ([raw("UPDATE sessions SET user_id='bob' WHERE id=?", LEGACY)], legacy_binding()),
        "routed": ([raw("UPDATE sessions SET session_key='agent:main:telegram:dm:1' WHERE id=?", LEGACY)], legacy_binding()),
        "imported_by_other": ([lambda db: db.set_meta(BINDING_PREFIX + LEGACY, json.dumps(["default", "someone"]))], legacy_binding()),
        "compressed": ([compressed], legacy_binding()),
        "busy": ([lambda db: admit_session_input(db, epoch=1, principal_id=PRINCIPAL, session_id=LEGACY,
                                                  request_id="in-1", payload={"text": "hi"})], legacy_binding()),
        "lease_held": ([raw("INSERT INTO session_turn_leases(conversation_id,holder,acquired_at,expires_at) "
                            "VALUES(?, 'pid:1', 1.0, 9999999999.0)", LEGACY)], legacy_binding()),
        "already_bound": ([lambda db: db.set_meta(LEGACY_PREFIX + LEGACY, "{}")], legacy_binding()),
        "missing_row": ([raw("DELETE FROM sessions WHERE id=?", LEGACY)], legacy_binding()),
    }


@pytest.mark.parametrize("case", ["adopted", "own_stamp", "foreign_owner", "routed", "imported_by_other", "compressed",
                                  "busy", "lease_held", "already_bound", "missing_row"])
def test_reserve_legacy_arm_matches_m1(store, legacy_store, case):
    from hermes_state_local import commit_local_session

    steps, binding = _cases()[case]
    seed(store, *steps)
    legacy, ours = run_both(
        store.store_path, lambda db: as_record(commit_local_session(db, epoch=1, receipt=legacy_receipt(binding))),
        lambda: store.bindings.reserve(epoch=1, binding=binding, owner=None))
    assert_same(legacy, ours, lambda reservation: reservation.binding)
    if ours[0] == "ok":
        assert ours[1].created is True


def test_reserve_legacy_arm_is_one_write(store):
    seed(store, legacy=False)
    assert_one_write(store, lambda: store.bindings.reserve(epoch=1, binding=legacy_binding(), owner=None))
