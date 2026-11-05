"""``SessionBindingRepository.target_has_history`` on SQLite against main's ``unused_session`` (negated): a message, an
admission, a worker execution or a child session is history; the answer is one read."""

from __future__ import annotations

from typing import Any

import pytest

from tests.persistence_sqlite.authority._support import assert_same, run_both, seed_legacy
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_read


def _seed(db: Any) -> None:
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, register_worker_execution

    begin_runtime_epoch(db, instance_id="owner-a")
    for sid in ("s-empty", "s-message", "s-admitted", "s-worker", "s-parent"):
        db.create_session(sid, source="cli")
    db.append_message("s-message", "user", "hello")
    admit_session_input(db, epoch=1, principal_id="user-1", session_id="s-admitted", request_id="in-1", payload={"text": "hi"})
    register_worker_execution(db, epoch=1, execution_id="exec-1", session_id="s-worker", generation=0,
                              kind="compute", adoption_secret="secret-1")
    db.create_session("s-child", source="cli", parent_session_id="s-parent")


@pytest.mark.parametrize("sid", ["s-empty", "s-message", "s-admitted", "s-worker", "s-parent", "s-missing"])
def test_target_has_history_matches_m1(backend, scope_a, legacy_store, sid):
    from tests.persistence_sqlite._main_callers.session_local_recovery import unused_session

    store = backend.open(scope_a)
    seed_legacy(store.store_path, _seed)

    def legacy(db: Any) -> Any:
        with db._read_ctx() as conn:
            return not unused_session(conn, sid)

    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.target_has_history(sid))
    assert_same(legacy_out, ours)
    assert ours[1] is (sid not in ("s-empty", "s-missing"))
    assert_one_read(store, lambda: store.bindings.target_has_history(sid))
