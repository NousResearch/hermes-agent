"""Tests for the Phase 4 s6 dispatch helper in hermes_cli.gateway.

`_dispatch_via_service_manager_if_s6` decides whether a
`hermes gateway start/stop/restart` invocation should be routed to
the in-container S6ServiceManager instead of falling through to the
host systemd/launchd/windows code path.
"""
from __future__ import annotations

from typing import Any

import pytest


class _CallRecorder:
    """Minimal stand-in for S6ServiceManager."""
    kind = "s6"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start(self, name: str) -> None:
        self.calls.append(("start", name))

    def stop(self, name: str) -> None:
        self.calls.append(("stop", name))

    def restart(self, name: str) -> None:
        self.calls.append(("restart", name))


def test_dispatch_returns_false_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the environment isn't s6 (host run), the helper must
    return False and not invoke a manager — callers continue with
    their existing systemd/launchd/windows path."""
    from hermes_cli import gateway as gw
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "systemd",
    )
    # Should not even attempt to construct a manager.
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: pytest.fail("manager should not be constructed on host"),
    )
    assert gw._dispatch_via_service_manager_if_s6("start", profile="x") is False


def test_dispatch_returns_true_and_calls_start_on_s6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import gateway as gw
    rec = _CallRecorder()
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_via_service_manager_if_s6("start", profile="coder") is True
    assert rec.calls == [("start", "gateway-coder")]


@pytest.mark.parametrize("action,expected", [
    ("start", "start"),
    ("stop", "stop"),
    ("restart", "restart"),
])
def test_dispatch_translates_action_to_manager_method(
    monkeypatch: pytest.MonkeyPatch, action: str, expected: str,
) -> None:
    from hermes_cli import gateway as gw
    rec = _CallRecorder()
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_via_service_manager_if_s6(action, profile="x") is True
    assert rec.calls == [(expected, "gateway-x")]


def test_dispatch_unknown_action_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized action (e.g. 'install') must not silently
    succeed — return False so the host code path handles it."""
    from hermes_cli import gateway as gw
    rec = _CallRecorder()
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_via_service_manager_if_s6("install", profile="x") is False
    assert rec.calls == []


def test_dispatch_defaults_profile_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When profile is None, the helper resolves it via _profile_arg().
    With no profile context set anywhere, that resolves to "default"."""
    from hermes_cli import gateway as gw
    rec = _CallRecorder()
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._profile_suffix", lambda: "",
    )
    assert gw._dispatch_via_service_manager_if_s6("start") is True
    assert rec.calls == [("start", "gateway-default")]
