"""Contract suite for ``SessionBindingRepository.bind_api``, ``api_binding`` and ``declared_session``.

The session row, its route columns, the routing entry, the declared key and the receipt publish in one write; every refusal writes nothing. The broken-writer case arms the reference backend's publication failpoint.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from persistence import AuthorityRefusal, FenceStale, ProofRejected, SessionId
from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import NewApiBinding, api_generation_id
from persistence.conversation.sessions import NewSession
from tests.contracts._reference._authority import AuthorityState, InjectedFailure, authority_state
from tests.contracts.authority.test_bindings_contract import assert_refusal, begin
from tests.contracts.conftest import require_failpoints
from tests.contracts._scopes import MEDIA_ROOT

STAMP = "2026-09-25T10:00:00"
STARTED = 1790330400.0


def api_binding(
    session_id: str = "api-1",
    *,
    route: Optional[str] = None,
    storage_source: str = "api_server",
    title: Optional[str] = None,
    declared_key: Optional[str] = None,
    caller_key: Optional[str] = None,
    receipt: Optional[dict[str, Any]] = None,
) -> NewApiBinding:
    route = route or f"agent:main:api_server:dm:{session_id}"
    origin: dict[str, Any] = {"platform": "api_server", "chat_id": session_id, "user_id": "api", "chat_type": "dm"}
    entry = {"session_key": route, "session_id": session_id, "created_at": STAMP, "updated_at": STAMP, "origin": origin}
    stored = {"profile_id": "default", "session_id": session_id, "route": route, "entry": entry}
    if declared_key:
        stored["declared_key"] = declared_key
    if storage_source == "bot_room":
        stored.update(storage_source="bot_room", room_identity=["home", "room", "member", "default"])
    return NewApiBinding(
        session_id=SessionId(session_id), route=route, storage_source=storage_source, title=title, declared_key=declared_key,
        caller_key=caller_key, origin=origin, entry=entry, receipt=receipt or stored, started_at=STARTED)


def _kwargs(binding: NewApiBinding) -> dict[str, Any]:
    return {name: getattr(binding, name) for name in NewApiBinding.__dataclass_fields__}


def test_command_rejects_malformed_fields():
    good = api_binding()
    for change in ({"route": ""}, {"storage_source": ""}, {"title": ""}, {"declared_key": ""}, {"caller_key": ""},
                   {"origin": {"chat_id": "api-1"}}, {"entry": {**dict(good.entry), "session_id": "other"}},
                   {"entry": {**dict(good.entry), "created_at": "nope"}}, {"receipt": {"session_id": "other"}}):
        with pytest.raises(ValueError):
            NewApiBinding(**{**_kwargs(good), **change})


def test_bind_publishes_row_route_declared_key_and_receipt(backend, scope_a):
    # CONTRACT: the row (source, started_at), its route columns (session_key = route, chat_id = id, user_id, dm, origin), the raw routing entry, the declared key and the receipt.
    store_a = backend.open(scope_a)
    binding = api_binding("api-1", declared_key="conv-1")
    assert store_a.bindings.bind_api(epoch=begin(store_a), binding=binding) is None
    record = store_a.sessions.get(SessionId("api-1"))
    assert (record.source, record.session_key, record.chat_id, record.user_id, record.chat_type) == ("api_server", binding.route, "api-1", "api", "dm")
    assert (record.started_at, record.hidden, record.origin) == (STARTED, False, dict(binding.origin))
    assert json.loads(store_a.routing.load_routing_entries()[binding.route]) == dict(binding.entry)
    assert store_a.bindings.declared_session("conv-1") == "api-1"
    stored = store_a.bindings.api_binding(SessionId("api-1"))
    assert (stored.profile_id, stored.session_id, stored.route, stored.declared_key, stored.storage_source, stored.room_identity, stored.room_generation) == (
        "default", "api-1", binding.route, "conv-1", "api_server", None, 0)
    assert store_a.bindings.api_binding(SessionId("api-none")) is None
    assert store_a.bindings.declared_session("nope") is None


def test_existing_receipt_is_kept_and_checked(backend, scope_a):
    # CONTRACT: a stored binding writes nothing again; another storage source is permission_denied and another declared key admission_conflict.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1", declared_key="conv-1"))
    first = store_a.bindings.api_binding(SessionId("api-1"))
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1", receipt={"profile_id": "x", "session_id": "api-1", "route": "r", "entry": {}}))
    assert store_a.bindings.api_binding(SessionId("api-1")) == first
    with pytest.raises(ProofRejected):
        store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1", storage_source="bot_room", title="Group: x"))
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1", declared_key="conv-2")), "admission_conflict")
    assert store_a.bindings.declared_session("conv-2") is None


def test_room_binding_owns_its_title_and_hides_the_row(backend, scope_a):
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("room_1", storage_source="bot_room", title="Group: r"))
    record = store_a.sessions.get(SessionId("room_1"))
    assert (record.source, record.title, record.hidden) == ("bot_room", "Group: r", True)
    stored = store_a.bindings.api_binding(SessionId("room_1"))
    assert (stored.storage_source, stored.room_identity) == ("bot_room", ("home", "room", "member", "default"))
    # CONTRACT: another session cannot bind the same title.
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("room_2", storage_source="bot_room", title="Group: r")), "admission_conflict")
    assert store_a.sessions.get(SessionId("room_2")) is None


def test_refusals_write_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.sessions.create(NewSession(session_id=SessionId("native"), source="cli"))
    # CONTRACT: a row of another source is permission_denied (after the declared key is reserved inside the same write, so nothing remains).
    with pytest.raises(ProofRejected):
        store_a.bindings.bind_api(epoch=epoch, binding=api_binding("native", declared_key="conv-n"))
    assert store_a.bindings.declared_session("conv-n") is None
    store_a.sessions.create(NewSession(session_id=SessionId("titled"), source="bot_room"))
    store_a.sessions.set_title(SessionId("titled"), "Other")
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("titled", storage_source="bot_room", title="Group: t")), "admission_conflict")
    store_a.sessions.create(NewSession(session_id=SessionId("routed"), source="api_server", session_key="elsewhere"))
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("routed")), "admission_conflict")
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-a", route="shared-route"))
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-b", route="shared-route")), "admission_conflict")
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-c", declared_key="conv-c"))
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-d", declared_key="conv-c")), "admission_conflict")
    assert store_a.bindings.declared_session("conv-c") == "api-c"
    for sid in ("native", "titled", "routed", "api-b", "api-d"):
        assert store_a.bindings.api_binding(SessionId(sid)) is None


def test_legacy_session_key_is_adopted_with_the_callers_key(backend, scope_a):
    # CONTRACT: a row an earlier release stamped with the declared (or caller) key moves onto the route; the key is registered and stamped into the receipt.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.sessions.create(NewSession(session_id=SessionId("old-api"), source="api_server", session_key="my-key"))
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("old-api", caller_key="my-key"))
    assert store_a.sessions.get(SessionId("old-api")).session_key == api_binding("old-api").route
    assert store_a.bindings.declared_session("my-key") == "old-api"
    assert store_a.bindings.api_binding(SessionId("old-api")).declared_key == "my-key"


def test_route_held_by_an_unbound_heal_replacement_is_taken_back(backend, scope_a):
    # CONTRACT: a route naming a session without its own binding goes back to the caller.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    route = api_binding("api-1").route
    store_a.routing.upsert_routing_entry(route, json.dumps({"session_key": route, "session_id": "heal-1"}))
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1"))
    assert json.loads(store_a.routing.load_routing_entries()[route])["session_id"] == "api-1"


def test_gate_is_raised_after_the_epoch_and_before_any_write(backend, scope_a):
    # CONTRACT: the owner's admission gate refuses right after the epoch check, with its reason, field and detail.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    gate = GateRefusal("runtime_draining", field="session", detail="draining")
    with pytest.raises(FenceStale):
        store_a.bindings.bind_api(epoch=0, binding=api_binding("api-g"), gate=gate)
    with pytest.raises(AuthorityRefusal) as caught:
        store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-g"), gate=gate)
    assert (caught.value.reason, caught.value.field, caught.value.detail) == ("runtime_draining", "session", "draining")
    assert store_a.sessions.get(SessionId("api-g")) is None


def test_retired_id_and_a_moved_chain_refuse(backend, scope_a):
    # CONTRACT: a retired id is not_found; a walk from ``walked_from`` needs every passed generation still retired, else chain_moved.
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-r"))
    assert store_a.sessions.delete(SessionId("api-r"), media_root=MEDIA_ROOT)
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-r")), "not_found")
    successor = api_generation_id("api-r", 1)
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding(successor), walked_from=SessionId("api-r"))
    assert store_a.bindings.api_binding(SessionId(successor)) is not None
    skipped = api_generation_id("api-s", 1)
    assert_refusal(lambda: store_a.bindings.bind_api(epoch=epoch, binding=api_binding(skipped), walked_from=SessionId("api-s")), "chain_moved")


def test_stale_epoch_precedes_the_stored_receipt(backend, scope_a):
    store_a = backend.open(scope_a)
    stale = begin(store_a)
    store_a.bindings.bind_api(epoch=stale, binding=api_binding("api-1"))
    begin(store_a, "instance-b")
    for command in (api_binding("api-1"), api_binding("api-2", declared_key="conv-2")):
        with pytest.raises(FenceStale):
            store_a.bindings.bind_api(epoch=stale, binding=command)
    assert store_a.sessions.get(SessionId("api-2")) is None
    assert store_a.bindings.declared_session("conv-2") is None


def test_failed_publication_leaves_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    epoch = begin(store_a)
    authority_state(backend, scope_a).failpoints.arm(AuthorityState.PUBLISH)
    with pytest.raises(InjectedFailure):
        store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1", declared_key="conv-1"))
    assert store_a.sessions.get(SessionId("api-1")) is None
    assert store_a.routing.load_routing_entries() == {}
    assert store_a.bindings.declared_session("conv-1") is None
    store_a.bindings.bind_api(epoch=epoch, binding=api_binding("api-1", declared_key="conv-1"))
    assert store_a.bindings.api_binding(SessionId("api-1")) is not None


def test_bind_keeps_the_whole_routing_entry_as_given(backend, scope_a):
    # CONTRACT: the routing index stores the caller's entry JSON as it is, naive stamps included.
    store_a = backend.open(scope_a)
    base = api_binding("api-rich")
    naive = "2026-09-25T10:00:00.123456"
    entry = {**dict(base.entry), "created_at": naive, "updated_at": naive, "platform": "api_server", "display_name": "Caller",
             "chat_type": "dm", "input_tokens": 3, "was_auto_reset": False}
    store_a.bindings.bind_api(epoch=begin(store_a), binding=NewApiBinding(**{**_kwargs(base), "entry": entry}))
    assert json.loads(store_a.routing.load_routing_entries()[base.route]) == entry
    recovered = store_a.routing.latest_session_for_peer(source="api_server", session_key=base.route)
    assert recovered is not None and recovered.session.session_id == "api-rich"
    assert recovered.session.origin == dict(base.origin)
