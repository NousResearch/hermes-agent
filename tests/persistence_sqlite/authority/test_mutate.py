"""``SessionRuntimeRepository.mutate`` and ``prepare_mutation`` on SQLite: a value that is no command refuses
``invalid_params``, the branch publish reads a route's profile as the shared key parser does, and each command and the
prepare are one write transaction that a failure right after its first mutating statement leaves unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from persistence.authority import runtime as rt
from persistence.authority.bindings import local_session_id
from tests.persistence_sqlite.authority._support import counted_transactions
from tests.persistence_sqlite.authority.test_binding_local import make_binding, reserved
from tests.persistence_sqlite.authority.test_binding_local import seed as seed_store
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_write

PRINCIPAL = "user-1"
LOCAL = str(local_session_id("default", PRINCIPAL, "req-1"))


def mutate(store: Any, command: Any, *, session_id: str, request_id: str = "req-m", expected_revision: int = 0,
           expected_generation: int | None = None, epoch: int = 1, owner: Any = None, **kwargs: Any) -> rt.MutationOutcome:
    return store.runtime.mutate(epoch, principal_id=PRINCIPAL, session_id=session_id, request_id=request_id,
                                expected_revision=expected_revision, command=command, expected_generation=expected_generation,
                                owner=owner, **kwargs)


def plain(session_id: str = "s-plain", **kwargs: Any):
    return lambda db: db.create_session(session_id, source="cli", **kwargs)


def lineage(db: Any) -> None:
    db.create_session("s-root", source="cli")
    db.store.sessions.end("s-root", "compression")
    db.create_session("s-tip", source="cli", parent_session_id="s-root")


def transcript(session_id: str = "s-plain"):
    def run(db: Any) -> None:
        db.create_session(session_id, source="cli")
        for role, content in (("user", "one"), ("assistant", "reply"), ("user", "two"), ("assistant", "reply two")):
            db.append_message(session_id, role=role, content=content)
    return run


# ── command validation ────────────────────────────────────────────────────────


def test_a_value_that_is_no_command_refuses_invalid_params(store):
    from persistence import AuthorityRefusal

    seed_store(store, plain())
    for command in (SimpleNamespace(operation="sidebar", payload=lambda: {"color": "red"}), "sidebar", None):
        with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
            mutate(store, command, session_id="s-plain")


# ── branch route profile ──────────────────────────────────────────────────────


@pytest.mark.parametrize("key", [None, "", "agent", "agent:main:x", "agent::x", "agent:main~:y", "agent:work:z",
                                 "foo:bar", 12, "agent:default:q"])
def test_the_branch_route_profile_matches_m1(key):
    from persistence.sqlite._state.keys import profile_from_session_key
    from persistence.sqlite.authority._mutation_publish import _profile_from_session_key

    assert _profile_from_session_key(key) == profile_from_session_key(key)


# ── rewind ────────────────────────────────────────────────────────────────────


def _user_ids(store: Any) -> list[int]:
    import sqlite3

    with sqlite3.connect(store.store_path) as conn:
        return [row[0] for row in conn.execute("SELECT id FROM messages WHERE role='user' ORDER BY id")]


# ── prepared policy commits ───────────────────────────────────────────────────


def _policy(**changes: Any) -> dict:
    base = {"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": '{"kind":"launch","n":1}'}
    return {**base, **changes}


# ── one transaction ───────────────────────────────────────────────────────────


def test_each_command_is_one_write_that_rolls_back_whole(store, frozen):
    seed_store(store, lineage, transcript(), reserved(make_binding()))
    second = _user_ids(store)[1]
    assert_one_write(store, lambda: mutate(store, rt.Sidebar(archived=True, pinned=True), session_id="s-tip"))
    assert_one_write(store, lambda: mutate(store, rt.Rewind(second), session_id="s-plain", expected_generation=0))
    assert_one_write(store, lambda: mutate(store, rt.Reset(), session_id=LOCAL, expected_generation=0))
    assert_one_write(store, lambda: mutate(store, rt.Delete(), session_id="s-plain", request_id="del", expected_revision=1, expected_generation=1))


def test_prepare_and_prepared_commit_are_one_write_each(store, frozen):
    seed_store(store, reserved(make_binding()))
    with counted_transactions(store.holder()) as counts:
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0,
                                                  command=rt.EditPolicy("session"))
    assert counts == {"write": 1, "read": 0}
    values = {"policy": _policy(edit_approval_policy="session")}
    assert_one_write(store, lambda: mutate(store, rt.EditPolicy("session"), session_id=LOCAL, prepared=rt.PreparedMutation(snapshot, values)))
