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




def _raise_missing_sleep(file: str, args: list[str]) -> None:
    raise FileNotFoundError(2, "No such file or directory", file)


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

    monkeypatch.setattr("hermes_cli.gateway.os.execvp", _raise_missing_sleep)
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


def _armed_watchdog(monkeypatch: pytest.MonkeyPatch):
    """Arm the real watchdog as hermes_cli.main's argv fast-path does; the long timeout keeps the
    deadline out of the test, only the handle state at handoff is under test."""
    import hermes_startup_watchdog as sw

    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    sw._reset_for_tests()
    handle = sw.arm_startup_watchdog(timeout_s=3600)
    assert handle is not None and handle.is_alive()
    return sw, handle


def test_redirect_disarms_startup_watchdog_before_parking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #102000: the CMD process never reaches a GatewayRunner, so the #36208 in-process
    heartbeat must not park under an armed watchdog (it would os._exit(75) a healthy container)."""
    from hermes_cli import gateway as gw

    sw, handle = _armed_watchdog(monkeypatch)
    try:
        _stub_s6(monkeypatch, on_s6=True)
        monkeypatch.setattr("hermes_cli.gateway.os.execvp", _raise_missing_sleep)
        disarmed_at_park: list[bool] = []
        monkeypatch.setattr(
            "hermes_cli.gateway._block_until_terminated",
            lambda: disarmed_at_park.append(handle.disarmed),
        )

        assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is True

        assert disarmed_at_park == [True]
        assert sw._handle is None
    finally:
        sw._reset_for_tests()
    capsys.readouterr()


def test_redirect_not_taken_leaves_startup_watchdog_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside s6 the gateway still boots in-process, so GatewayRunner's own disarm must govern."""
    from hermes_cli import gateway as gw

    sw, handle = _armed_watchdog(monkeypatch)
    try:
        _stub_s6(monkeypatch, on_s6=False)

        assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is False

        assert sw._handle is handle and handle.is_alive() and not handle.disarmed
    finally:
        sw._reset_for_tests()


# ---------------------------------------------------------------------------
# On-demand slot registration — a profile dir that exists but was never registered (#111720)
# ---------------------------------------------------------------------------


class _UnregisteredRecorder(_CallRecorder):
    """Recorder whose slot is missing until ``register_profile_gateway`` runs."""

    def __init__(self) -> None:
        super().__init__()
        self.registered: list[tuple[str, bool]] = []
        self._slots: set[str] = set()

    def _svc(self, action: str, name: str) -> None:
        from hermes_cli.service_manager import GatewayNotRegisteredError
        if name not in self._slots:
            raise GatewayNotRegisteredError(name.removeprefix("gateway-"))
        self.calls.append((action, name))

    def start(self, name: str) -> None:
        self._svc("start", name)

    def stop(self, name: str) -> None:
        self._svc("stop", name)

    def register_profile_gateway(self, profile: str, *, start_now: bool = True) -> None:
        self.registered.append((profile, start_now))
        self._slots.add(f"gateway-{profile}")


def _arrange(monkeypatch, tmp_path, mgr, *, profile: str, seed_soul: bool):
    """Force the s6 branch and make ``tmp_path`` the shared HERMES_HOME the slot maps back to."""
    from hermes_cli import gateway as gw
    from hermes_cli import service_manager as sm

    monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
    monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile_dir = tmp_path / "profiles" / profile
    profile_dir.mkdir(parents=True)
    if seed_soul:
        (profile_dir / "SOUL.md").write_text("# soul\n", encoding="utf-8")
    return gw


def test_start_registers_a_missing_slot_for_a_real_profile(monkeypatch, tmp_path, capsys):
    """A profile created from the host against a bind-mounted home has a directory (SOUL.md) but
    no s6 slot. ``gateway start`` must register it and come up instead of demanding a container
    restart; the registration is ``down`` so the ordinary ``start`` owns the desired-state write."""
    mgr = _UnregisteredRecorder()
    gw = _arrange(monkeypatch, tmp_path, mgr, profile="coder", seed_soul=True)

    assert gw._dispatch_via_service_manager_if_s6("start", "coder") is True

    assert mgr.registered == [("coder", False)]
    assert mgr.calls == [("start", "gateway-coder")]
    assert "registered the s6 gateway slot" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("action", "seed_soul"),
    [
        pytest.param("stop", True, id="stop-never-registers"),
        pytest.param("start", False, id="no-soul-marker-mints-nothing"),
    ],
)
def test_missing_slot_stays_an_error_outside_the_repair_case(
    monkeypatch, tmp_path, capsys, action, seed_soul
):
    """Only ``start`` on a real profile self-heals: stopping an unregistered profile and starting a
    mistyped/stray directory (no SOUL.md) keep the original ✗ + exit 1 and mint no slot."""
    mgr = _UnregisteredRecorder()
    gw = _arrange(monkeypatch, tmp_path, mgr, profile="coder", seed_soul=seed_soul)

    with pytest.raises(SystemExit) as excinfo:
        gw._dispatch_via_service_manager_if_s6(action, "coder")

    assert excinfo.value.code == 1
    assert mgr.registered == []
    assert mgr.calls == []
    assert "no such gateway 'coder'" in capsys.readouterr().out
