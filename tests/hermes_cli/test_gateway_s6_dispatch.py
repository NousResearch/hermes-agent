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


# ---------------------------------------------------------------------------
# Friendly error rendering — GatewayNotRegisteredError / S6CommandError
# (PR #30136 review item I2)
# ---------------------------------------------------------------------------


def test_dispatch_renders_gateway_not_registered_friendly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """`hermes -p typo gateway start` should print a clear message and
    exit 1 — not dump a traceback at the user."""
    from hermes_cli import gateway as gw
    from hermes_cli.service_manager import GatewayNotRegisteredError

    class _RaisesMissing:
        kind = "s6"

        def start(self, name: str) -> None:
            raise GatewayNotRegisteredError("typo")

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager", lambda: _RaisesMissing(),
    )

    with pytest.raises(SystemExit) as excinfo:
        gw._dispatch_via_service_manager_if_s6("start", profile="typo")
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "no such gateway 'typo'" in out
    assert "hermes profile create typo" in out
    # And critically: no traceback prefix.
    assert "Traceback" not in out


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


class _ExecvpCalled(BaseException):
    """Sentinel raised by the os.execvp stub so tests can assert on it
    without actually replacing the test runner process. Inherits from
    BaseException so it bypasses generic ``except Exception`` blocks in
    the code under test (just like a real exec would)."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv


def _stub_execvp(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace os.execvp with a recorder that raises _ExecvpCalled."""
    calls: list[list[str]] = []

    def fake_execvp(file: str, args: list[str]) -> None:  # noqa: ANN401
        calls.append([file, *args])
        raise _ExecvpCalled([file, *args])

    monkeypatch.setattr("hermes_cli.gateway.os.execvp", fake_execvp)
    return calls


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


def test_redirect_fires_inside_s6_container(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Inside an s6 container, `gateway run` should:

    1. Dispatch `start` to the service manager.
    2. Print the loud breadcrumb to stderr.
    3. exec `sleep infinity` to keep the CMD alive without binding
       container lifetime to gateway PID lifetime.
    """
    from hermes_cli import gateway as gw

    rec = _stub_s6(monkeypatch, on_s6=True)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    execvp_calls = _stub_execvp(monkeypatch)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)

    with pytest.raises(_ExecvpCalled) as excinfo:
        gw._maybe_redirect_run_to_s6_supervision(_Args())

    # 1. Dispatcher fired.
    assert rec.calls == [("start", "gateway-default")]
    # 2. Breadcrumb went to stderr and mentions the opt-out path.
    err = capsys.readouterr().err
    assert "s6 supervision" in err
    assert "--no-supervise" in err
    assert "HERMES_GATEWAY_NO_SUPERVISE" in err
    # 3. exec'd `sleep infinity`.
    assert execvp_calls == [["sleep", "sleep", "infinity"]]
    assert excinfo.value.argv == ["sleep", "sleep", "infinity"]


def test_redirect_short_circuits_supervised_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recursion guard: when the supervised gateway s6-supervise is
    running execs `hermes gateway run --replace`, the
    HERMES_S6_SUPERVISED_CHILD sentinel must short-circuit the redirect
    so the gateway actually starts foreground. Without this guard the
    supervised process would re-dispatch `start` → re-exec `run` → ...
    in an infinite loop.
    """
    from hermes_cli import gateway as gw

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager",
        lambda: pytest.fail("dispatcher should not run when sentinel is set"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp",
        lambda *a, **kw: pytest.fail("execvp should not run when sentinel is set"),
    )
    monkeypatch.setenv("HERMES_S6_SUPERVISED_CHILD", "1")
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)

    assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is False


def test_redirect_respects_no_supervise_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--no-supervise` (CLI flag) must skip the redirect even inside
    an s6 container, restoring pre-s6 foreground semantics."""
    from hermes_cli import gateway as gw

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager",
        lambda: pytest.fail("dispatcher should not run when --no-supervise is set"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp",
        lambda *a, **kw: pytest.fail("execvp should not run when --no-supervise is set"),
    )
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)

    assert gw._maybe_redirect_run_to_s6_supervision(_Args(no_supervise=True)) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
def test_redirect_respects_no_supervise_env(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    """`HERMES_GATEWAY_NO_SUPERVISE=1` (env var) must skip the redirect.

    Truthiness mirrors the dashboard service's own env var parsing —
    1/true/yes are all accepted, case-insensitively.
    """
    from hermes_cli import gateway as gw

    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager",
        lambda: pytest.fail("dispatcher should not run when env opt-out is set"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp",
        lambda *a, **kw: pytest.fail("execvp should not run when env opt-out is set"),
    )
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", value)

    assert gw._maybe_redirect_run_to_s6_supervision(_Args()) is False


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
    _stub_execvp(monkeypatch)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    for falsy in ("", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", falsy)
        with pytest.raises(_ExecvpCalled):
            gw._maybe_redirect_run_to_s6_supervision(_Args())
