"""A store serves a hosted activity only through the pin of the hosted scope bound when each verb runs, checked by the
backend at execution, not only when a handle is looked up: a repository or ``aio`` view kept from a handle is refused
under any other bound hosted scope. A local (unpinned) store serves no hosted activity at all, on every backend. A
backend declaring ``scope_pinning`` keeps each pin's ``'shared'`` hosted-room placement its own, and the claimed-turn
fence covers that placement too, on a local store as on a pinned one. A local store's kept ``'shared'`` hosted-room
verbs and health service serve no hosted activity either: they are refused like every other verb, whatever file the
backend keeps them in. Every case runs on every registered backend, never a skip: the pinned cases assert
the refusal of the pin on a backend that does not declare ``scope_pinning``.
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from agent.runtime_scope import bind_turn, claim_fence
from hermes_scope import Principal, Scope, push_state
from persistence import ProfileScope, ScopePinRefused, SessionId, TurnFenceSuperseded
from persistence.conversation.sessions import NewSession
from tests.contracts._scopes import pinned_scope
from tests.contracts.conftest import require_capability

SESSION = SessionId("hosted-activity-s1")


def _hosted_scope(tenant: str, agent: str) -> Scope:
    return Scope(tenant, agent, Principal("member"), owner=Principal("member"), can_administer=False)


@contextmanager
def _hosted(tenant: str, agent: str) -> Iterator[None]:
    """The hosted activity of ``(tenant, agent)`` bound, as a gateway binds it."""
    with push_state(scope=_hosted_scope(tenant, agent)):
        yield


def _elsewhere(fn, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` as another writer would: in a fresh context, with nothing bound."""
    return contextvars.Context().run(lambda: fn(*args, **kwargs))


def _pinned_or_refused(backend: Any, scope: ProfileScope) -> Any:
    """``scope``'s pinned handle when the backend declares ``scope_pinning``; otherwise assert the open refuses."""
    if backend.capabilities.scope_pinning:
        return backend.open(scope)
    with pytest.raises(ScopePinRefused):
        backend.open(scope)
    return None


def test_a_local_store_serves_no_hosted_activity(backend_name: str, backend, scope_a) -> None:
    """While a hosted scope is bound, every read and write through a local store's kept repository and ``aio`` view is
    refused with ScopePinRefused and changes nothing; with no hosted scope bound the same views serve as before."""
    local = backend.open(scope_a)
    sessions, aio = local.sessions, local.aio
    sessions.create(NewSession(session_id=SessionId("local-secret"), source="cli", cwd="/local"))
    with _hosted("tenant-b", "agent-1"):
        with pytest.raises(ScopePinRefused):
            sessions.get(SessionId("local-secret"))
        with pytest.raises(ScopePinRefused):
            sessions.update_cwd(SessionId("local-secret"), "/intruder")
        with pytest.raises(ScopePinRefused):
            sessions.create(NewSession(session_id=SessionId("hosted-row"), source="cli"))
        with pytest.raises(ScopePinRefused):
            asyncio.run(aio.sessions.get(SessionId("local-secret")))
    assert sessions.get(SessionId("local-secret")).cwd == "/local"
    assert sessions.get(SessionId("hosted-row")) is None
    assert asyncio.run(aio.sessions.get(SessionId("local-secret"))).cwd == "/local"


def test_a_pinned_handle_serves_only_its_bound_pin(backend_name: str, backend, scope_a) -> None:
    """A pinned handle's kept repository and ``aio`` view serve under its own bound hosted scope and with none bound,
    and are refused (ScopePinRefused, nothing changed) while another pin's hosted scope is bound."""
    store = _pinned_or_refused(backend, pinned_scope("tenant-a", "agent-1", profile_dir=scope_a.profile_dir))
    if store is None:
        return
    sessions, aio = store.sessions, store.aio
    with _hosted("tenant-a", "agent-1"):
        sessions.create(NewSession(session_id=SessionId("private-a"), source="cli", cwd="/a"))
        assert asyncio.run(aio.sessions.get(SessionId("private-a"))).cwd == "/a"
    for tenant, agent in (("tenant-b", "agent-1"), ("tenant-a", "agent-2")):
        with _hosted(tenant, agent):
            with pytest.raises(ScopePinRefused):
                sessions.get(SessionId("private-a"))
            with pytest.raises(ScopePinRefused):
                sessions.update_cwd(SessionId("private-a"), "/intruder")
            with pytest.raises(ScopePinRefused):
                asyncio.run(aio.sessions.get(SessionId("private-a")))
    assert sessions.get(SessionId("private-a")).cwd == "/a"


def test_pinned_shared_rooms_are_never_visible_across_pins(backend_name: str, backend, scope_a) -> None:
    """Each pin's ``'shared'`` hosted-room placement is its own: a room one pin creates there is not listed, found or
    writable through another pin built in the same folder."""
    store_a = _pinned_or_refused(backend, pinned_scope("tenant-a", "agent-1", profile_dir=scope_a.profile_dir))
    if store_a is None:
        return
    store_b = backend.open(pinned_scope("tenant-b", "agent-1", profile_dir=scope_a.profile_dir))
    store_a.hosted_rooms.create_room(placement="shared", room_id="secret-room", name="a private room", members=[],
                                     authority_gateway_id="gateway-a", now=100.0)
    assert store_b.hosted_rooms.list_rooms(placement="shared") == []
    store_b.hosted_rooms.create_room(placement="shared", room_id="secret-room", name="b's own room", members=[],
                                     authority_gateway_id="gateway-b", now=200.0)
    assert [room["name"] for room in store_a.hosted_rooms.list_rooms(placement="shared")] == ["a private room"]
    assert [room["name"] for room in store_b.hosted_rooms.list_rooms(placement="shared")] == ["b's own room"]


def test_a_shared_placement_write_is_fenced_after_takeover(backend_name: str, backend, scope_a) -> None:
    """After another owner's takeover, a superseded turn's hosted-room write to the ``'shared'`` placement is refused
    with TurnFenceSuperseded and leaves nothing, exactly as its write to the ``'profile'`` placement does."""
    scope = pinned_scope("tenant-a", "agent-1", profile_dir=scope_a.profile_dir)
    store = _pinned_or_refused(backend, scope)
    if store is None:
        return
    require_capability(backend, "authority_fences")
    require_capability(backend, "durable_admissions")
    with _hosted("tenant-a", "agent-1"):
        store.sessions.create(NewSession(session_id=SESSION, source="contract"))
        epoch = store.runtime.begin_epoch(instance_id="owner-1")
        store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={})
        claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    _elsewhere(store.runtime.begin_epoch, instance_id="owner-2")
    with _hosted("tenant-a", "agent-1"), bind_turn(SESSION, claimed.admission_id, claim_fence(claimed), store=scope):
        for placement in ("profile", "shared"):
            with pytest.raises(TurnFenceSuperseded):
                store.hosted_rooms.create_room(placement=placement, room_id=f"late-{placement}", name="stale room",
                                               members=[], authority_gateway_id="gateway-a", now=100.0)
    for placement in ("profile", "shared"):
        assert store.hosted_rooms.list_rooms(placement=placement) == []


def _create_shared(rooms: Any, room_id: str) -> Any:
    return rooms.create_room(placement="shared", room_id=room_id, name="private room", members=[],
                             authority_gateway_id="gateway-a", now=100.0)


def _shared_ids(rooms: Any) -> list[str]:
    return sorted(room["room_id"] for room in rooms.list_rooms(placement="shared"))


def test_a_local_stores_kept_shared_hosted_rooms_serve_no_hosted_activity(backend_name: str, backend, scope_a) -> None:
    """While a hosted scope is bound, a local store's kept hosted-room repository refuses its ``'shared'`` placement's
    read, probe and write with ScopePinRefused and changes nothing; unbound it serves as before."""
    rooms = backend.open(scope_a).hosted_rooms
    _create_shared(rooms, "local-room")
    with _hosted("tenant-b", "agent-1"):
        with pytest.raises(ScopePinRefused):
            rooms.list_rooms(placement="shared")
        with pytest.raises(ScopePinRefused):
            rooms.probe_hosted_room(placement="shared", room_id="local-room")
        with pytest.raises(ScopePinRefused):
            _create_shared(rooms, "hosted-room")
    assert _shared_ids(rooms) == ["local-room"]
    assert rooms.probe_hosted_room(placement="shared", room_id="local-room")


def test_a_local_stores_kept_health_serves_no_hosted_activity(backend_name: str, backend, scope_a) -> None:
    """While a hosted scope is bound, a local store's kept ``health`` refuses its probe and its latch with
    ScopePinRefused; unbound it answers as before."""
    health = backend.open(scope_a).health
    before = (health.probe(), health.storage_state())
    with _hosted("tenant-b", "agent-1"):
        with pytest.raises(ScopePinRefused):
            health.probe()
        with pytest.raises(ScopePinRefused):
            health.storage_state()
    assert (health.probe(), health.storage_state()) == before


def _claimed_local(store: Any) -> Any:
    store.sessions.create(NewSession(session_id=SESSION, source="contract"))
    _create_shared(store.hosted_rooms, "existing-room")  # the shared placement is set up before the claim
    epoch = store.runtime.begin_epoch(instance_id="owner-1")
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={})
    return store.admissions.claim(epoch=epoch, session_id=SESSION)


def test_a_superseded_turns_shared_placement_write_publishes_nothing(backend_name: str, backend, scope_a) -> None:
    """On a local store, after another owner's takeover, a claimed turn's write to the ``'shared'`` hosted-room
    placement is refused with TurnFenceSuperseded and publishes nothing, as its session write is refused."""
    require_capability(backend, "authority_fences")
    require_capability(backend, "durable_admissions")
    store = backend.open(scope_a)
    claimed = _claimed_local(store)
    _elsewhere(store.runtime.begin_epoch, instance_id="owner-2")
    with bind_turn(SESSION, claimed.admission_id, claim_fence(claimed), store=scope_a):
        with pytest.raises(TurnFenceSuperseded):
            store.sessions.update_cwd(SESSION, "/stale")
        with pytest.raises(TurnFenceSuperseded):
            _create_shared(store.hosted_rooms, "stale-room")
    assert _shared_ids(store.hosted_rooms) == ["existing-room"]


def test_a_live_turns_shared_placement_write_lands(backend_name: str, backend, scope_a) -> None:
    """On a local store, a claimed turn that still holds its lease writes to the ``'shared'`` placement."""
    require_capability(backend, "authority_fences")
    require_capability(backend, "durable_admissions")
    store = backend.open(scope_a)
    claimed = _claimed_local(store)
    with bind_turn(SESSION, claimed.admission_id, claim_fence(claimed), store=scope_a):
        _create_shared(store.hosted_rooms, "live-room")
    assert _shared_ids(store.hosted_rooms) == ["existing-room", "live-room"]
