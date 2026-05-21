"""Tests for the Phase 4 s6 hooks in hermes_cli.profiles.

Specifically: _allocate_gateway_port, _maybe_register_gateway_service,
_maybe_unregister_gateway_service. The integration with
create_profile and delete_profile is covered indirectly by the
existing TestCreateProfile and TestDeleteProfile classes in
tests/hermes_cli/test_profiles.py; here we only exercise the new
helper surface that doesn't touch the filesystem.
"""
from __future__ import annotations

from typing import Any

import pytest

from hermes_cli.profiles import (
    _allocate_gateway_port,
    _maybe_register_gateway_service,
    _maybe_unregister_gateway_service,
)


# ---------------------------------------------------------------------------
# _allocate_gateway_port
# ---------------------------------------------------------------------------


def test_allocate_gateway_port_is_deterministic() -> None:
    """Same profile name → same port across calls. This matters because
    a profile's gateway must come back up on the same port across
    container restarts."""
    a = _allocate_gateway_port("coder")
    b = _allocate_gateway_port("coder")
    assert a == b


def test_allocate_gateway_port_in_advertised_range() -> None:
    """[9200, 9800) — the window the helper's docstring promises."""
    for name in ("a", "b", "coder", "assistant", "very-long-profile-name-here"):
        port = _allocate_gateway_port(name)
        assert 9200 <= port < 9800, f"{name} got {port}"


def test_allocate_gateway_port_distributes_across_range() -> None:
    """Sanity check: ports for ~100 random-ish names should land in
    enough distinct buckets that the distribution is plausibly uniform.
    Catches accidental hash truncation that would collapse the range."""
    ports = {_allocate_gateway_port(f"profile-{i}") for i in range(100)}
    # 100 inputs mapped into 600 slots — expect at least ~60 distinct.
    assert len(ports) >= 60, f"Only {len(ports)} distinct ports across 100 names"


# ---------------------------------------------------------------------------
# _maybe_register_gateway_service / _maybe_unregister_gateway_service
# ---------------------------------------------------------------------------


class _HostManager:
    """Mimics a host backend that doesn't support runtime registration."""
    kind = "systemd"

    def supports_runtime_registration(self) -> bool:
        return False

    def register_profile_gateway(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("host backend register_profile_gateway should not be called")

    def unregister_profile_gateway(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("host backend unregister_profile_gateway should not be called")


class _S6Manager:
    """Mimics S6ServiceManager just enough for the hooks."""
    kind = "s6"

    def __init__(self) -> None:
        self.registered: list[tuple[str, int]] = []
        self.unregistered: list[str] = []
        self.raise_on_register: Exception | None = None
        self.raise_on_unregister: Exception | None = None

    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(
        self, profile: str, *, port: int,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        if self.raise_on_register is not None:
            raise self.raise_on_register
        self.registered.append((profile, port))

    def unregister_profile_gateway(self, profile: str) -> None:
        if self.raise_on_unregister is not None:
            raise self.raise_on_unregister
        self.unregistered.append(profile)


def test_register_noop_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: _HostManager(),
    )
    # Should NOT raise the AssertionError from _HostManager.register
    _maybe_register_gateway_service("hostprof")


def test_register_calls_through_on_s6(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _S6Manager()
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: mgr,
    )
    _maybe_register_gateway_service("coder")
    assert len(mgr.registered) == 1
    profile, port = mgr.registered[0]
    assert profile == "coder"
    assert 9200 <= port < 9800


def test_register_swallows_duplicate_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing s6 registration (from container-boot reconcile)
    is a benign condition — register must not propagate ValueError."""
    mgr = _S6Manager()
    mgr.raise_on_register = ValueError("already registered")
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: mgr,
    )
    # Should NOT raise
    _maybe_register_gateway_service("coder")


def test_register_swallows_arbitrary_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Even an unexpected exception from the manager must not bring
    down `hermes profile create` — print and continue."""
    mgr = _S6Manager()
    mgr.raise_on_register = RuntimeError("svscanctl exploded")
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: mgr,
    )
    _maybe_register_gateway_service("coder")
    captured = capsys.readouterr()
    assert "Could not register" in captured.out


def test_register_swallows_no_backend_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `get_service_manager()` raises RuntimeError (no backend
    detected), the hook must silently no-op."""
    def _no_backend() -> None:
        raise RuntimeError("no supported service manager detected")
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", _no_backend,
    )
    # Should NOT raise
    _maybe_register_gateway_service("anywhere")


def test_unregister_noop_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: _HostManager(),
    )
    _maybe_unregister_gateway_service("hostprof")


def test_unregister_calls_through_on_s6(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _S6Manager()
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: mgr,
    )
    _maybe_unregister_gateway_service("coder")
    assert mgr.unregistered == ["coder"]


def test_unregister_swallows_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    mgr = _S6Manager()
    mgr.raise_on_unregister = RuntimeError("svc gone weird")
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: mgr,
    )
    _maybe_unregister_gateway_service("coder")
    captured = capsys.readouterr()
    assert "Could not unregister" in captured.out
