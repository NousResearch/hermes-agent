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


# ---------------------------------------------------------------------------
# _dispatch_all_via_service_manager_if_s6 — --all under s6
# ---------------------------------------------------------------------------


class _ListingRecorder(_CallRecorder):
    """_CallRecorder that also exposes a profile list."""

    def __init__(self, profiles: list[str]) -> None:
        super().__init__()
        self._profiles = profiles

    def list_profile_gateways(self) -> list[str]:
        return list(self._profiles)


def test_dispatch_all_returns_false_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import gateway as gw
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "systemd",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: pytest.fail("manager should not be constructed on host"),
    )
    assert gw._dispatch_all_via_service_manager_if_s6("stop") is False


def test_dispatch_all_iterates_every_profile_on_stop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from hermes_cli import gateway as gw
    rec = _ListingRecorder(["coder", "writer", "assistant"])
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_all_via_service_manager_if_s6("stop") is True
    assert rec.calls == [
        ("stop", "gateway-coder"),
        ("stop", "gateway-writer"),
        ("stop", "gateway-assistant"),
    ]
    out = capsys.readouterr().out
    assert "Stopped 3 profile gateway(s)" in out


def test_dispatch_all_iterates_every_profile_on_restart(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from hermes_cli import gateway as gw
    rec = _ListingRecorder(["coder", "writer"])
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_all_via_service_manager_if_s6("restart") is True
    assert rec.calls == [
        ("restart", "gateway-coder"),
        ("restart", "gateway-writer"),
    ]
    out = capsys.readouterr().out
    assert "Restarted 2 profile gateway(s)" in out


def test_dispatch_all_handles_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """A failure on one profile must not skip the others; the helper
    reports each failure and the success count."""
    from hermes_cli import gateway as gw

    class _FailOnWriter(_ListingRecorder):
        def stop(self, name: str) -> None:
            if name == "gateway-writer":
                raise RuntimeError("supervise FIFO permission denied")
            super().stop(name)

    rec = _FailOnWriter(["coder", "writer", "assistant"])
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_all_via_service_manager_if_s6("stop") is True
    # The two successful ones were called; writer raised before recording.
    assert ("stop", "gateway-coder") in rec.calls
    assert ("stop", "gateway-assistant") in rec.calls
    assert ("stop", "gateway-writer") not in rec.calls
    out = capsys.readouterr().out
    assert "Stopped 2 profile gateway(s)" in out
    assert "Could not stop gateway-writer" in out
    assert "supervise FIFO permission denied" in out


def test_dispatch_all_empty_list_reports_and_returns_true(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """With no profile gateways registered the helper still claims the
    dispatch (returns True) and prints a friendly message — the host
    fallback would just pkill nothing, which isn't useful inside a
    container."""
    from hermes_cli import gateway as gw
    rec = _ListingRecorder([])
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    assert gw._dispatch_all_via_service_manager_if_s6("stop") is True
    assert rec.calls == []
    assert "No profile gateways" in capsys.readouterr().out


def test_dispatch_all_unknown_action_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`start --all` is not a supported CLI surface; the helper must
    fall through to the host code path rather than no-op."""
    from hermes_cli import gateway as gw
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: pytest.fail(
            "manager should not be constructed for unsupported --all action",
        ),
    )
    assert gw._dispatch_all_via_service_manager_if_s6("start") is False
