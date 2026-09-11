"""Regression tests for the advertised compatibility entry points."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "compat_manifest.json").exists(), reason="compat layer removed (scheduled revert)"
)


def test_gateway_startup_watchdog_matches_the_startup_watchdog_api() -> None:
    import gateway.startup_watchdog as compat
    import hermes_startup_watchdog as base

    base_public = {name for name in vars(base) if not name.startswith("_")}
    compat_public = {name for name in vars(compat) if not name.startswith("_")}
    assert compat_public == base_public
    assert compat.arm_startup_watchdog is base.arm_startup_watchdog
    assert not hasattr(compat, "arm_shutdown_watchdog")


def test_historical_session_registry_functions_are_facade_aliases() -> None:
    import hermes_state as facade
    import hermes_state_registry as registry

    for name in ("get_shared_session_db", "release_shared_session_db", "close_shared_session_dbs"):
        assert getattr(facade, name) is getattr(registry, name)


class _FakeScope:
    def __init__(self) -> None:
        self.events: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.pops: list[Any] = []

    def push(self, name: str, scope_type: Any, **kwargs: Any) -> object:
        return object()

    def event(self, *args: Any, **kwargs: Any) -> None:
        self.events.append((args, kwargs))

    def pop(self, handle: Any, **kwargs: Any) -> None:
        self.pops.append(handle)


class _FakeRelay:
    ScopeType = SimpleNamespace(Agent="agent", Function="function")

    def __init__(self) -> None:
        self.scope = _FakeScope()
        self.plugin = SimpleNamespace(report=lambda: None)
        self.subscribers = SimpleNamespace(flush_async=lambda: None)

    def get_scope_stack(self) -> None:
        return None


def test_relay_compat_wrappers_delegate_to_runtime_and_noop_is_inert(monkeypatch) -> None:
    from agent import relay_runtime

    relay = _FakeRelay()
    runtime = relay_runtime.RelayRuntime(relay=relay, profile_key="compat-test")
    monkeypatch.setattr(relay_runtime, "get_runtime", lambda **_: runtime)
    try:
        assert relay_runtime.emit_mark(
            "compat.mark", session_id="session", data={"value": 1}, metadata={"source": "test"}
        ) is True
        session = runtime.get_session("session")
        assert session is not None
        assert relay_runtime.get_session_handle("session") is session.handle
        assert relay.scope.events == [
            (
                ("compat.mark",),
                {"handle": session.handle, "data": {"value": 1}, "metadata": {"source": "test"}},
            )
        ]
    finally:
        runtime.shutdown()

    noop = relay_runtime.NoopRelayRuntime(profile_key="compat-test", reason="unavailable")
    assert noop.emit_mark("compat.mark", {"session_id": "session"}) is False
    assert noop.get_session_handle("session") is None
