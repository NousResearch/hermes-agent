"""Hosted scope binding in the persistence assembly: while a hosted scope is bound, every store a handle reaches is
pinned to that scope's ``(tenant, agent)``; outside hosted mode nothing is pinned and nothing pinned opens.

A fake pinning backend stands in for the configured provider by replacing ``persistence_assembly._assemble``; the
hosted scopes are bound with ``agent.runtime_scope.bind_state`` as a gateway binds them.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

import persistence_assembly as pa
from agent.runtime_scope import bind_state
from hermes_scope import Principal, Scope
from persistence import ProfileScope, ScopePinRefused, StorePin


class _FakeStore:
    def __init__(self, backend: _FakePinningBackend, scope: ProfileScope) -> None:
        self.backend = backend
        self.scope = scope
        self.sessions = ("sessions-of", scope.store_key)

    def ready(self) -> None:
        return None

    def serves_same_store(self, other: object) -> bool:
        return isinstance(other, _FakeStore) and other.scope.store_key == self.scope.store_key


class _FakePinningBackend:
    """Serves pinned scopes only, as a tenant-isolating provider would; records every open."""

    def __init__(self, stores: Any) -> None:
        self.stores = dict(stores)
        self.opened: list[ProfileScope] = []
        self.closed = 0

    def open(self, scope: ProfileScope) -> _FakeStore:
        if scope.pin is None:
            raise AssertionError("the fake pinning backend was asked for a local store")
        self.opened.append(scope)
        return _FakeStore(self, scope)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def assembled(monkeypatch: pytest.MonkeyPatch) -> list[_FakePinningBackend]:
    """Every backend the assembly builds, in order; the process caches start empty."""
    built: list[_FakePinningBackend] = []

    def assemble(provider: str, stores: Any, **_: Any) -> _FakePinningBackend:
        built.append(_FakePinningBackend(stores))
        return built[-1]

    monkeypatch.setattr(pa, "_assemble", assemble)
    monkeypatch.setattr(pa, "_shared", {})
    monkeypatch.setattr(pa, "_pinned_backends", {})
    return built


@contextmanager
def hosted(tenant: str, agent: str = "agent-1") -> Iterator[Scope]:
    scope = Scope(tenant, agent, Principal("member"), owner=Principal("member"), can_administer=False)
    with bind_state(scope):
        yield scope


def test_hosted_profile_scope_is_pinned_to_the_bound_scope(assembled) -> None:
    from hermes_constants import get_hermes_home

    with hosted("tenant-a"):
        scope = pa.profile_scope()
        handle = pa.open_persistence(scope)
        assert handle.sessions == ("sessions-of", StorePin("tenant-a", "agent-1").key)
        handle.close()
    assert scope.pin == StorePin("tenant-a", "agent-1")
    assert (scope.name, scope.profile_dir) == ("hosted", get_hermes_home().resolve())
    assert [opened.pin for opened in assembled[0].opened] == [StorePin("tenant-a", "agent-1")]
    assert assembled[0].stores == {}


def test_two_pins_never_share_a_cache_entry_and_share_one_backend(assembled) -> None:
    with hosted("tenant-a"):
        handle_a = pa.open_persistence(pa.profile_scope())
        store_a = handle_a.sessions
    with hosted("tenant-b"):
        handle_b = pa.open_persistence(pa.profile_scope())
        store_b = handle_b.sessions
        assert set(pa._shared) == {StorePin("tenant-a", "agent-1").key, StorePin("tenant-b", "agent-1").key}
    assert store_a != store_b
    assert len(assembled) == 1
    handle_a.close()
    handle_b.close()
    assert pa._shared == {} and assembled[0].closed == 0
    pa.close_all_stores()
    assert assembled[0].closed == 1 and pa._pinned_backends == {}


def test_a_handle_opened_under_one_tenant_refuses_its_first_use_under_another(assembled) -> None:
    with hosted("tenant-a"):
        handle = pa.open_persistence(pa.profile_scope())
    with hosted("tenant-b"), pytest.raises(ScopePinRefused):
        handle.sessions
    with hosted("tenant-a", agent="agent-2"), pytest.raises(ScopePinRefused):
        handle.sessions
    assert assembled == [] or assembled[0].opened == []
    handle.close()


def test_a_handle_used_under_one_tenant_refuses_a_later_use_under_another(assembled) -> None:
    with hosted("tenant-a"):
        handle = pa.open_persistence(pa.profile_scope())
        handle.sessions
    with hosted("tenant-b"), pytest.raises(ScopePinRefused):
        handle.sessions
    # A worker thread that did not inherit the turn's context binds nothing: the bound store still serves only its pin.
    assert handle.sessions == ("sessions-of", StorePin("tenant-a", "agent-1").key)
    handle.close()


def test_open_refuses_any_scope_but_the_bound_pin(assembled, tmp_path: Path) -> None:
    local = ProfileScope(profile_dir=tmp_path, name="default")
    foreign = ProfileScope(profile_dir=tmp_path, name="hosted", pin=StorePin("tenant-b", "agent-1"))
    with hosted("tenant-a"):
        for scope in (local, foreign):
            with pytest.raises(ScopePinRefused):
                pa.open_persistence(scope)
    assert pa._shared == {} and assembled == []


def test_a_pinned_scope_never_opens_outside_hosted_mode(assembled, tmp_path: Path) -> None:
    pinned = ProfileScope(profile_dir=tmp_path, name="hosted", pin=StorePin("tenant-a", "agent-1"))
    with pytest.raises(ScopePinRefused):
        pa.open_persistence(pinned)
    with hosted("tenant-a"):
        handle = pa.open_persistence(pinned)
    with pytest.raises(ScopePinRefused):
        handle.sessions
    handle.close()
    assert assembled == []


def test_a_hosted_activity_cannot_name_a_profile_or_home(assembled, tmp_path: Path) -> None:
    with hosted("tenant-a"):
        for call in (lambda: pa.profile_scope("work"), lambda: pa.profile_scope("default"),
                     lambda: pa.profile_scope(home=tmp_path)):
            with pytest.raises(ScopePinRefused):
                call()


def test_local_profile_scope_is_unpinned_and_unchanged() -> None:
    from hermes_constants import get_hermes_home

    scope = pa.profile_scope()
    assert scope.pin is None
    assert scope == pa.profile_scope(home=get_hermes_home())
    assert scope.store_key == scope.hermes_home_key
    assert (scope.profile_dir, scope.name) == (get_hermes_home().resolve(), "default")


def test_hosted_doors_never_touch_a_local_store(assembled, tmp_path: Path) -> None:
    home = tmp_path / "local-profile"
    home.mkdir()
    local = ProfileScope(profile_dir=home, name="default")
    with hosted("tenant-a"):
        bound = pa.profile_scope()
        assert pa.store_location(bound) is None
        refused = (
            lambda: pa.store_location(local),
            lambda: pa.storage_health(local), lambda: pa.storage_health(bound),
            lambda: pa.store_exists(local), lambda: pa.store_exists(bound),
            lambda: pa.open_store_admin(local), lambda: pa.open_store_admin(home / "state.db"),
            lambda: pa.open_profile_repair(local, read_only=True),
            lambda: pa.close_stores_under(home),
            lambda: pa.open_persistence_reader("work").__enter__(),
            lambda: pa.open_persistence_reader(home).__enter__(),
        )
        for call in refused:
            with pytest.raises(ScopePinRefused):
                call()
        with pa.live_stores() as handles:
            assert handles == []
    assert not list(home.iterdir())


def test_local_doors_refuse_a_pinned_scope(tmp_path: Path) -> None:
    pinned = ProfileScope(profile_dir=tmp_path, name="hosted", pin=StorePin("tenant-a", "agent-1"))
    local = ProfileScope(profile_dir=tmp_path, name="default")
    assert pa.store_location(local) == tmp_path.resolve() / "state.db"
    for call in (lambda: pa.store_location(pinned), lambda: pa.storage_health(pinned),
                 lambda: pa.store_exists(pinned), lambda: pa.open_store_admin(pinned),
                 lambda: pa.open_profile_repair(pinned, read_only=True)):
        with pytest.raises(ScopePinRefused):
            call()


def test_the_built_in_sqlite_refuses_a_hosted_turns_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pa, "_shared", {})
    monkeypatch.setattr(pa, "_pinned_backends", {})
    with hosted("tenant-a"):
        with pytest.raises(ScopePinRefused):
            pa.open_persistence(pa.profile_scope(), eager=True)
        assert pa._shared == {}
    pa.close_all_stores()
