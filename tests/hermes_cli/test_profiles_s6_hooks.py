"""Tests for the Phase 4 s6 hooks in hermes_cli.profiles.

Specifically: _maybe_register_gateway_service,
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
    _maybe_register_gateway_service,
    _maybe_unregister_gateway_service,
)


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
        self.registered: list[str] = []
        self.unregistered: list[str] = []
        self.raise_on_register: Exception | None = None
        self.raise_on_unregister: Exception | None = None

    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(
        self, profile: str, *,
        extra_env: dict[str, str] | None = None,
        start_now: bool = True,
    ) -> None:
        if self.raise_on_register is not None:
            raise self.raise_on_register
        self.registered.append(profile)
        self.last_start_now = start_now

    def unregister_profile_gateway(self, profile: str) -> None:
        if self.raise_on_unregister is not None:
            raise self.raise_on_unregister
        self.unregistered.append(profile)


def _patch_detect_s6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend we're inside an s6 container so the host short-circuit
    in :func:`_maybe_register_gateway_service` /
    :func:`_maybe_unregister_gateway_service` doesn't fire.

    Without this, ``detect_service_manager()`` runs its real
    implementation (host Linux/macOS in CI), returns ``"systemd"`` or
    ``"launchd"``, and the hooks return early before reaching the
    patched ``get_service_manager``. Each s6-call-through test
    explicitly opts into this so the host-no-op tests can still
    exercise the early-return path.
    """
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager",
        lambda: "s6",
    )


def test_register_noop_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # NOTE: deliberately DO NOT patch detect_service_manager — we want
    # the real host detection to kick in and short-circuit before
    # get_service_manager is ever called. The lambda below is a
    # defense-in-depth assertion that get_service_manager is never
    # reached on host.
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: _HostManager(),
    )
    # Should NOT raise the AssertionError from _HostManager.register
    _maybe_register_gateway_service("hostprof")


def test_register_passes_start_now_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """_maybe_register_gateway_service must register with start_now=False
    so that profile creation does not auto-start a gateway that may
    conflict with the main gateway's bot-token lock."""
    _patch_detect_s6(monkeypatch)
    mgr = _S6Manager()
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: mgr,
    )
    _maybe_register_gateway_service("coder")
    assert mgr.last_start_now is False, (
        "profile creation must not auto-start the gateway service"
    )


def test_register_silent_when_detect_throws(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """If detect_service_manager itself raises (e.g. a partial s6
    install on a host machine), the hook must stay silent — no
    confusing s6 warning printed to a user who has never touched a
    container."""
    def _broken_detect() -> str:
        raise RuntimeError("detection blew up")
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", _broken_detect,
    )
    # If get_service_manager is reached, the test will assert via
    # _HostManager.register. It must NOT be reached.
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: _HostManager(),
    )
    _maybe_register_gateway_service("anywhere")
    captured = capsys.readouterr()
    assert "Could not register" not in captured.out
    assert captured.out == ""


