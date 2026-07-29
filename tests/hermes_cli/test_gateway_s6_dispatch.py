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


def test_dispatch_renders_s6_command_error_friendly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """An s6-svc failure (e.g. EACCES on the supervise FIFO) should
    surface the stderr inline, not as an opaque traceback."""
    from hermes_cli import gateway as gw
    from hermes_cli.service_manager import S6CommandError

    class _RaisesS6Error:
        kind = "s6"

        def start(self, name: str) -> None:
            raise S6CommandError(
                service=name,
                action="start",
                returncode=111,
                stderr="s6-svc: fatal: Permission denied",
            )

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: _RaisesS6Error(),
    )

    with pytest.raises(SystemExit) as excinfo:
        gw._dispatch_via_service_manager_if_s6("start", profile="coder")
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "rc=111" in out
    assert "Permission denied" in out
    assert "Traceback" not in out


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


def test_redirect_noop_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host runs (non-s6) must not redirect. Returns False; caller
    continues to the foreground gateway code path unchanged."""
    from hermes_cli import gateway as gw

    _stub_s6(monkeypatch, on_s6=False)
    # If execvp got called we'd raise — keep it bound so test fails loudly.
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp",
        lambda *a, **kw: pytest.fail("execvp should not be called on host"),
    )
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)

    assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is False


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


def test_block_until_terminated_installs_sigterm_handler_and_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_block_until_terminated`` must register a SIGTERM handler (so
    `docker stop` exits cleanly) and then block on signal.pause() — never
    touching an external binary. Regression guard for issue #36208, where
    os.execvp("sleep", ...) crashed the container with FileNotFoundError
    when PATH lacked a directory containing `sleep`.
    """
    import signal as _signal
    from hermes_cli import gateway as gw

    registered: dict[int, object] = {}
    monkeypatch.setattr(
        "hermes_cli.gateway.signal.signal",
        lambda signum, handler: registered.__setitem__(signum, handler),
    )

    # Make signal.pause() raise after the first call so the infinite loop
    # terminates deterministically instead of hanging the test.
    pause_calls = {"n": 0}

    def fake_pause() -> None:
        pause_calls["n"] += 1
        raise KeyboardInterrupt  # break out of the `while True: pause()` loop

    monkeypatch.setattr("hermes_cli.gateway.signal.pause", fake_pause)

    with pytest.raises(KeyboardInterrupt):
        gw._block_until_terminated()

    # A SIGTERM handler was installed...
    assert _signal.SIGTERM in registered
    # ...and it exits with the conventional 128+signum code.
    handler = registered[_signal.SIGTERM]
    with pytest.raises(SystemExit) as exc:
        handler(_signal.SIGTERM, None)  # type: ignore[operator]
    assert exc.value.code == 128 + _signal.SIGTERM
    # ...and we actually blocked on pause().
    assert pause_calls["n"] == 1


def test_redirect_no_supervise_env_falsy_values_dont_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falsy / unrecognized values of HERMES_GATEWAY_NO_SUPERVISE must
    NOT opt out. We're strict about what counts as "yes" so a typo
    like `HERMES_GATEWAY_NO_SUPERVISE=0` doesn't silently enable the
    historical foreground behavior."""
    from hermes_cli import gateway as gw

    _stub_s6(monkeypatch, on_s6=True)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")

    # The redirect reaching its `sleep` heartbeat means it did NOT opt
    # out. Stub execvp to record + raise (so it doesn't replace the test
    # process) rather than actually exec.
    class _ExecvpCalled(BaseException):
        pass

    execvp_calls: list[str] = []

    def fake_execvp(file: str, args: list[str]) -> None:
        execvp_calls.append(file)
        raise _ExecvpCalled

    monkeypatch.setattr("hermes_cli.gateway.os.execvp", fake_execvp)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    for falsy in ("", "0", "false", "no", "off", "garbage"):
        execvp_calls.clear()
        monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", falsy)
        with pytest.raises(_ExecvpCalled):
            gw._maybe_redirect_run_to_s6_supervision(_Args())
        assert execvp_calls == ["sleep"], f"redirect should fire for {falsy!r}"
