"""Contract cases for the runtime retirement reads."""

from __future__ import annotations

from persistence.authority.bindings import api_generation_id, generation_request_id, local_session_id, room_session_id
from persistence.authority.runtime import ApiGenerations, Delete, GenerationRow, LocalGenerations, PinnedIdentity, RoomGenerations
from persistence.conversation.sessions import NewSession


def _retire_with_turn(store, epoch, session_id, *, principal_id="api", request_id="turn-1"):
    store.sessions.create(NewSession(session_id=session_id, source="api_server"))
    record, _ = store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id, request_id=request_id, payload={"text": "x"})
    store.admissions.cancel(epoch=epoch, admission_id=record.admission_id)
    row = store.sessions.get(session_id)
    store.runtime.mutate(epoch, principal_id=principal_id, session_id=session_id, request_id="delete", expected_revision=row.runtime_revision,
                         expected_generation=row.runtime_generation, command=Delete(), owner=None)


def test_retired_session_reads_the_marker(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-reads")
    assert store.runtime.retired_session("s-x") is False
    _retire_with_turn(store, epoch, "s-x")
    assert store.runtime.retired_session("s-x") is True


def test_first_generation_skips_retired_ids_unless_pinned(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-reads")
    chain = ApiGenerations("api-conv")
    assert store.runtime.first_generation(chain) == (pick := store.runtime.first_generation(chain))
    assert (pick.generation, pick.session_id, pick.row) == (0, "api-conv", None)
    _retire_with_turn(store, epoch, "api-conv")
    moved = store.runtime.first_generation(chain)
    assert (moved.generation, moved.session_id, moved.row) == (1, str(api_generation_id("api-conv", 1)), None)
    # CONTRACT: an exact retry holding its identity on the retired generation stays there.
    pinned = store.runtime.first_generation(chain, pinned=PinnedIdentity("api", "turn-1"))
    assert (pinned.generation, pinned.session_id) == (0, "api-conv")
    assert store.runtime.first_generation(chain, pinned=PinnedIdentity("api", "other")).generation == 1
    store.sessions.create(NewSession(session_id=moved.session_id, source="api_server"))
    store.sessions.set_title(moved.session_id, "Named")
    found = store.runtime.first_generation(chain)
    assert found.row == GenerationRow(title="Named", source="api_server")
    # CONTRACT: start resumes the walk past a generation the caller judged spent.
    assert store.runtime.first_generation(chain, start=2).session_id == str(api_generation_id("api-conv", 2))


def test_first_generation_derives_room_and_local_chains(backend, scope_a):
    store = backend.open(scope_a)
    identity = ("install", "room", "member", "profile")
    assert store.runtime.first_generation(RoomGenerations(identity)).session_id == str(room_session_id(identity))
    assert store.runtime.first_generation(RoomGenerations(identity), start=1).session_id == str(room_session_id(identity, 1))
    local = LocalGenerations("default", "bot", "bot-chat")
    assert store.runtime.first_generation(local, start=3).session_id == str(local_session_id("default", "bot", generation_request_id("bot-chat", 3)))


def test_expiring_retirements_names_sweep_markers_only(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-reads")
    _retire_with_turn(store, epoch, "s-deleted")
    store.runtime.seed_retirement("s-swept", expires=True)
    assert store.runtime.expiring_retirements(["s-deleted", "s-swept", "s-live"]) == {"s-swept"}
    assert store.runtime.retired_session("s-swept")
