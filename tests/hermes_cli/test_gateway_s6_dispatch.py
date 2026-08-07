"""Tests for the Phase 4 s6 dispatch helper in hermes_cli.gateway.

`_dispatch_via_service_manager_if_s6` decides whether a
`hermes gateway start/stop/restart` invocation should be routed to
the in-container S6ServiceManager instead of falling through to the
host systemd/launchd/windows code path.
"""
from __future__ import annotations


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


# ---------------------------------------------------------------------------
# Friendly error rendering — GatewayNotRegisteredError / S6CommandError
# (PR #30136 review item I2)
# ---------------------------------------------------------------------------




# =============================================================================
# `_maybe_redirect_run_to_s6_supervision`: the "upgrade old `gateway run`
# invocation to supervised semantics inside an s6 container" helper.
# =============================================================================


class _Args:
    """Lightweight argparse-like namespace for the helper."""

    def __init__(self, no_supervise: bool = False) -> None:
        self.no_supervise = no_supervise


def _stub_s6(monkeypatch: pytest.MonkeyPatch, *, on_s6: bool) -> _CallRecorder:
    """Wire up service-manager stubs so the underlying dispatcher will
    fire (on_s6=True) or return False (on_s6=False)."""
    rec = _CallRecorder()
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager",
        lambda: "s6" if on_s6 else "systemd",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: rec,
    )
    return rec




def test_redirect_falls_back_when_sleep_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression guard for issue #36208: when ``os.execvp("sleep", ...)``
    raises (no `sleep` on a clobbered/empty PATH, or a minimal image
    without it), the redirect must NOT crash the container — it falls
    back to the in-process ``_block_until_terminated`` heartbeat so the
    container keeps running.
    """
    from hermes_cli import gateway as gw

    rec = _stub_s6(monkeypatch, on_s6=True)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")

    def missing_sleep(file: str, args: list[str]) -> None:
        raise FileNotFoundError(2, "No such file or directory", file)

    monkeypatch.setattr("hermes_cli.gateway.os.execvp", missing_sleep)
    block_calls: list[bool] = []
    monkeypatch.setattr(
        "hermes_cli.gateway._block_until_terminated",
        lambda: block_calls.append(True),
    )
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)

    # Must not raise FileNotFoundError — that was the #36208 crash.
    result = gw._maybe_redirect_run_to_s6_supervision(_Args())

    assert result is True
    assert rec.calls == [("start", "gateway-default")]
    # Fell back to the in-process heartbeat instead of crashing.
    assert block_calls == [True]
    err = capsys.readouterr().err
    assert "`sleep` is unavailable" in err


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", " On "])
def test_no_supervise_truthy_env_skips_s6_redirect(
    monkeypatch: pytest.MonkeyPatch, raw: str,
) -> None:
    """HERMES_GATEWAY_NO_SUPERVISE must honor shared truthy aliases (incl. on)."""
    from hermes_cli import gateway as gw

    rec = _stub_s6(monkeypatch, on_s6=True)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", raw)

    assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is False
    assert rec.calls == []


def test_no_supervise_falsy_env_still_redirects_on_s6(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli import gateway as gw

    rec = _stub_s6(monkeypatch, on_s6=True)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp",
        lambda *_a, **_k: None,
    )
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", "off")

    assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is True
    assert rec.calls == [("start", "gateway-default")]


