"""``mutate(Import)`` on SQLite against main's ``mutate_runtime_session``: the import command runs the housekeeping
family's import bodies (payload validation, the session row and transcript insert, the parent links) inside the
receipt write, with ``history=ImportControl`` binding the import's new ids where main runs ``import_history_control`` as
its authorize hook.

Each case runs main's function and the verb on copies of one seed: the same result, receipt and rows. An import may name
a session that does not exist yet; a deleted id refuses by name.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from persistence.authority import runtime as rt
from tests.persistence_sqlite.authority.test_binding_local import seed as seed_store
from tests.persistence_sqlite.authority.test_mutate import (
    PRINCIPAL, SUBJECT, compare, expect, legacy_mutate, mutate, plain,
)

NEW = {"id": "s-new", "source": "import", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
CHILD = {"id": "s-child", "source": "import", "parent_session_id": "s-new", "messages": []}
ORPHAN = {"id": "s-orphan", "source": "import", "parent_session_id": "s-plain", "messages": []}


@pytest.fixture
def minted(frozen, monkeypatch):
    """``frozen``, and both runs mint the same message uids for the imported rows."""
    import itertools
    import uuid

    import agent.message_metadata
    import session_rules.turns

    counter = [itertools.count(1)]
    for module in (agent.message_metadata, session_rules.turns):
        monkeypatch.setattr(module, "uuid4", lambda: uuid.UUID(int=next(counter[0])))

    def restart() -> None:
        frozen()
        counter[0] = itertools.count(1)
    return restart


def _actors() -> tuple[Any, rt.HistoryActor]:
    return (SimpleNamespace(profile_id="default", subject=SUBJECT),
            rt.HistoryActor(profile_id="default", subject=SUBJECT, login_label=None, may_create=True))


def _retired(session_id: str):
    def step(db: Any) -> None:
        from hermes_state_mutation_retirement import RETIRED_PREFIX

        db.set_meta(RETIRED_PREFIX + session_id, json.dumps({"retired_at": 1.0}))
    return step


IMPORTS = {
    "absent_session": ([], (NEW,), "s-new", None),
    "skips_existing": ([plain("s-plain")], (NEW, {"id": "s-plain", "source": "import", "messages": []}), "s-new", None),
    "links_fresh_parent": ([], (NEW, CHILD), "s-new", None),
    "detaches_foreign_parent": ([plain("s-plain")], (NEW, ORPHAN), "s-new", None),
    "deleted_id": ([_retired("s-new")], (NEW,), "s-new", "admission_conflict"),
    "first_is_not_the_session": ([], (NEW,), "s-other", "invalid_params"),
    "invalid_rows": ([], ({"id": "s-new", "messages": "x"},), "s-new", "invalid_params"),
}


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("case", sorted(IMPORTS))
def test_import_matches_m1(store, minted, emitted, case, bound):
    from hermes_state_mutation_binding import import_history_control

    steps, sessions, sid, expected = IMPORTS[case]
    seed_store(store, *steps)
    legacy_actor, actor = _actors()
    ids = tuple(item["id"] for item in sessions)
    hooks = {"_authorize_write": lambda conn: import_history_control(conn, legacy_actor, ids)} if bound else {}
    kwargs = {"history": rt.ImportControl(actor, ids)} if bound else {}
    command = rt.Import(sessions)
    new = compare(store, minted, emitted, lambda db: legacy_mutate(db, command, session_id=sid, **hooks),
                  lambda: mutate(store, command, session_id=sid, **kwargs))
    expect(new, expected)


def test_import_receipt_replays(store, minted, emitted):
    command = rt.Import((NEW,))
    seed_store(store, lambda db: legacy_mutate(db, command, session_id="s-new"))
    new = compare(store, minted, emitted, lambda db: legacy_mutate(db, command, session_id="s-new"),
                  lambda: mutate(store, command, session_id="s-new"), replayed=True)
    expect(new, None)
