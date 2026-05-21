"""Tests for hermes_cli.service_manager — the abstract ServiceManager
protocol, the detect_service_manager() entry point, and the host-side
adapter wrappers (Systemd / Launchd / Windows).

The s6 backend is added in Phase 3; its tests live alongside the
implementation in this same file once that phase ships.
"""
from __future__ import annotations

import pytest

from hermes_cli.service_manager import (
    LaunchdServiceManager,
    S6ServiceManager,
    ServiceManager,
    ServiceManagerKind,
    SystemdServiceManager,
    WindowsServiceManager,
    detect_service_manager,
    get_service_manager,
    validate_profile_name,
)


# ---------------------------------------------------------------------------
# validate_profile_name
# ---------------------------------------------------------------------------


def test_validate_profile_name_accepts_valid_names() -> None:
    # Smoke: known-good names should not raise.
    validate_profile_name("coder")
    validate_profile_name("my-profile")
    validate_profile_name("assistant_v2")
    validate_profile_name("a")
    validate_profile_name("0")
    validate_profile_name("0abc")


@pytest.mark.parametrize(
    "bad",
    [
        "",                  # empty
        "Coder",             # uppercase
        "foo/bar",           # path traversal
        "../escape",         # path traversal
        "-leading-dash",     # leading dash (s6 reads as a flag)
        "_leading_underscore",  # leading underscore
        "name with spaces",  # whitespace
        "name.with.dots",    # punctuation
        "a" * 252,           # too long
    ],
)
def test_validate_profile_name_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_profile_name(bad)


# ---------------------------------------------------------------------------
# detect_service_manager
# ---------------------------------------------------------------------------


def test_detect_service_manager_returns_known_value() -> None:
    """Without mocking, the function must still return one of the
    advertised literals — anything else means a new platform branch
    was added without updating ServiceManagerKind."""
    result = detect_service_manager()
    assert result in ("systemd", "launchd", "windows", "s6", "none")


# ---------------------------------------------------------------------------
# Backend wrappers — kind + registration unsupported on hosts
# ---------------------------------------------------------------------------


def test_systemd_manager_kind_and_registration_unsupported() -> None:
    mgr = SystemdServiceManager()
    assert mgr.kind == "systemd"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    with pytest.raises(NotImplementedError):
        mgr.unregister_profile_gateway("foo")
    assert mgr.list_profile_gateways() == []
    # Protocol conformance — runtime_checkable lets us assert this.
    assert isinstance(mgr, ServiceManager)


def test_launchd_manager_kind_and_registration_unsupported() -> None:
    mgr = LaunchdServiceManager()
    assert mgr.kind == "launchd"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    assert mgr.list_profile_gateways() == []
    assert isinstance(mgr, ServiceManager)


def test_windows_manager_kind_and_registration_unsupported() -> None:
    mgr = WindowsServiceManager()
    assert mgr.kind == "windows"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    assert isinstance(mgr, ServiceManager)


# ---------------------------------------------------------------------------
# Lifecycle delegation — wrappers must call through to module-level fns
# ---------------------------------------------------------------------------


def test_systemd_manager_lifecycle_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.systemd_start", lambda: called.append("start"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.systemd_stop", lambda: called.append("stop"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.systemd_restart", lambda: called.append("restart"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._probe_systemd_service_running",
        lambda *a, **kw: (False, True),
    )
    mgr = SystemdServiceManager()
    mgr.start("ignored")
    mgr.stop("ignored")
    mgr.restart("ignored")
    assert called == ["start", "stop", "restart"]
    assert mgr.is_running("ignored") is True


def test_launchd_manager_lifecycle_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.launchd_start", lambda: called.append("start"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.launchd_stop", lambda: called.append("stop"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.launchd_restart", lambda: called.append("restart"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._probe_launchd_service_running", lambda: False,
    )
    mgr = LaunchdServiceManager()
    mgr.start("ignored")
    mgr.stop("ignored")
    mgr.restart("ignored")
    assert called == ["start", "stop", "restart"]
    assert mgr.is_running("ignored") is False


def test_windows_manager_lifecycle_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    # Force-import the submodule so monkeypatch's attribute lookup
    # against the `hermes_cli` package succeeds — gateway_windows is
    # imported lazily inside the wrapper and may not yet be loaded.
    import hermes_cli.gateway_windows  # noqa: F401

    class _FakeWindowsModule:
        @staticmethod
        def start() -> None: called.append("start")
        @staticmethod
        def stop() -> None: called.append("stop")
        @staticmethod
        def restart() -> None: called.append("restart")
        @staticmethod
        def is_installed() -> bool: return True

    monkeypatch.setattr("hermes_cli.gateway_windows", _FakeWindowsModule)
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids",
        lambda **kw: [12345],
    )
    mgr = WindowsServiceManager()
    mgr.start("ignored")
    mgr.stop("ignored")
    mgr.restart("ignored")
    assert called == ["start", "stop", "restart"]
    assert mgr.is_running("ignored") is True


def test_windows_manager_is_running_false_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.gateway_windows  # noqa: F401

    class _FakeWindowsModule:
        @staticmethod
        def is_installed() -> bool: return False

    monkeypatch.setattr("hermes_cli.gateway_windows", _FakeWindowsModule)
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids",
        lambda **kw: [12345],  # PIDs would otherwise vote "running"
    )
    assert WindowsServiceManager().is_running("ignored") is False


def test_windows_manager_install_forwards_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    import hermes_cli.gateway_windows  # noqa: F401

    class _FakeWindowsModule:
        @staticmethod
        def install(*, force, start_now, start_on_login, elevated_handoff) -> None:
            captured["force"] = force
            captured["start_now"] = start_now
            captured["start_on_login"] = start_on_login
            captured["elevated_handoff"] = elevated_handoff

    monkeypatch.setattr("hermes_cli.gateway_windows", _FakeWindowsModule)
    WindowsServiceManager().install(
        force=True, start_now=True, start_on_login=False, elevated_handoff=True,
    )
    assert captured == {
        "force": True,
        "start_now": True,
        "start_on_login": False,
        "elevated_handoff": True,
    }


# ---------------------------------------------------------------------------
# get_service_manager factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,cls",
    [
        ("systemd", SystemdServiceManager),
        ("launchd", LaunchdServiceManager),
        ("windows", WindowsServiceManager),
    ],
)
def test_get_service_manager_returns_correct_backend(
    monkeypatch: pytest.MonkeyPatch,
    kind: ServiceManagerKind,
    cls: type,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: kind,
    )
    assert isinstance(get_service_manager(), cls)


def test_get_service_manager_raises_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "none",
    )
    with pytest.raises(RuntimeError, match="no supported service manager"):
        get_service_manager()


def test_get_service_manager_returns_s6_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The s6 backend ships in Phase 3 — the factory must return an
    S6ServiceManager when running inside a container."""
    from hermes_cli.service_manager import S6ServiceManager
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    assert isinstance(get_service_manager(), S6ServiceManager)


# ---------------------------------------------------------------------------
# S6ServiceManager — unit tests against a tmp-path scandir (no real s6)
# ---------------------------------------------------------------------------


@pytest.fixture
def s6_scandir(tmp_path):
    """Empty scandir for the S6ServiceManager tests."""
    d = tmp_path / "service"
    d.mkdir()
    return d


@pytest.fixture
def fake_subprocess_run(monkeypatch: pytest.MonkeyPatch):
    """Capture subprocess.run calls + always return success. Lets the
    S6ServiceManager tests run on hosts that don't have s6-svc /
    s6-svscanctl installed.

    Records are normalized: leading ``/command/`` is stripped from
    cmd[0] so assertions can match on the bare s6-svc / s6-svstat /
    s6-svscanctl name regardless of whether the manager calls them
    via absolute path or bare name."""
    calls: list[list[str]] = []

    def _fake(cmd, **kw):
        import subprocess as _sp
        seq = list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)]
        if seq and seq[0].startswith("/command/"):
            seq[0] = seq[0][len("/command/"):]
        calls.append(seq)
        return _sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("subprocess.run", _fake)
    return calls


def test_s6_manager_kind_and_supports_registration() -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager()
    assert mgr.kind == "s6"
    assert mgr.supports_runtime_registration() is True


def test_s6_register_creates_service_dir_and_triggers_scan(
    s6_scandir, fake_subprocess_run,
) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.register_profile_gateway("coder", port=9150)

    svc_dir = s6_scandir / "gateway-coder"
    assert svc_dir.is_dir()
    assert (svc_dir / "type").read_text().strip() == "longrun"

    run_path = svc_dir / "run"
    assert run_path.is_file()
    assert run_path.stat().st_mode & 0o111  # executable
    run_text = run_path.read_text()
    assert "hermes -p coder gateway run" in run_text
    assert "s6-setuidgid hermes" in run_text

    log_run = svc_dir / "log" / "run"
    assert log_run.is_file()
    log_text = log_run.read_text()
    # CRITICAL: HERMES_HOME must be a runtime env-var expansion, NOT
    # a Python-substituted absolute path. Negative-assert the wrong
    # form so future regressions are caught.
    assert "$HERMES_HOME" in log_text
    assert "logs/gateways/coder" in log_text
    assert "/opt/data/logs/gateways/coder" not in log_text, (
        "log_dir was hard-coded; must use ${HERMES_HOME} at run time"
    )

    # s6-svscanctl -a was invoked against the scandir
    assert any(
        cmd[0] == "s6-svscanctl" and "-a" in cmd
        and str(s6_scandir) in cmd
        for cmd in fake_subprocess_run
    ), f"s6-svscanctl -a not invoked; saw: {fake_subprocess_run}"


def test_s6_register_extra_env_is_quoted(s6_scandir, fake_subprocess_run) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.register_profile_gateway(
        "x", port=9300, extra_env={"FOO": "bar baz", "QUOTED": "a'b"},
    )
    run_text = (s6_scandir / "gateway-x" / "run").read_text()
    # shlex.quote should have wrapped both values
    assert "export FOO='bar baz'" in run_text
    assert "export QUOTED='a'\"'\"'b'" in run_text


def test_s6_register_rejects_invalid_profile_name(s6_scandir) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    with pytest.raises(ValueError):
        mgr.register_profile_gateway("Bad/Name", port=9100)


def test_s6_register_rejects_duplicate(s6_scandir, fake_subprocess_run) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    (s6_scandir / "gateway-coder").mkdir(parents=True)
    with pytest.raises(ValueError, match="already registered"):
        mgr.register_profile_gateway("coder", port=9150)


def test_s6_register_rolls_back_on_svscanctl_failure(
    s6_scandir, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If s6-svscanctl fails the service dir must be cleaned up so the
    next register call doesn't see a stale duplicate."""
    import subprocess as _sp
    from hermes_cli.service_manager import S6ServiceManager

    def _fail_scanctl(cmd, **kw):
        # Manager calls s6-svscanctl by absolute path; match on basename.
        if cmd[0].endswith("/s6-svscanctl"):
            return _sp.CompletedProcess(cmd, 1, "", "rescan failed")
        return _sp.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("subprocess.run", _fail_scanctl)

    mgr = S6ServiceManager(scandir=s6_scandir)
    with pytest.raises(RuntimeError, match="s6-svscanctl failed"):
        mgr.register_profile_gateway("coder", port=9150)
    assert not (s6_scandir / "gateway-coder").exists()


def test_s6_unregister_removes_service_dir(
    s6_scandir, fake_subprocess_run,
) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    svc_dir = s6_scandir / "gateway-coder"
    svc_dir.mkdir(parents=True)
    (svc_dir / "type").write_text("longrun\n")

    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.unregister_profile_gateway("coder")

    # s6-svc -d was issued
    assert any(
        cmd[0] == "s6-svc" and "-d" in cmd
        for cmd in fake_subprocess_run
    )
    # Service dir was removed
    assert not svc_dir.exists()
    # Rescan was triggered
    assert any(cmd[0] == "s6-svscanctl" for cmd in fake_subprocess_run)


def test_s6_unregister_absent_profile_is_noop(s6_scandir) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    # Should NOT raise even though "ghost" doesn't exist
    S6ServiceManager(scandir=s6_scandir).unregister_profile_gateway("ghost")


def test_s6_list_profile_gateways(s6_scandir) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    # Three gateway profiles + one unrelated service + one hidden dir
    (s6_scandir / "gateway-coder").mkdir()
    (s6_scandir / "gateway-assistant").mkdir()
    (s6_scandir / "gateway-writer").mkdir()
    (s6_scandir / "s6-linux-init-shutdownd").mkdir()  # filtered out
    (s6_scandir / ".lock").mkdir()  # filtered out (hidden)

    profiles = sorted(S6ServiceManager(scandir=s6_scandir).list_profile_gateways())
    assert profiles == ["assistant", "coder", "writer"]


def test_s6_list_profile_gateways_empty_when_scandir_missing(tmp_path) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    missing = tmp_path / "does-not-exist"
    assert S6ServiceManager(scandir=missing).list_profile_gateways() == []


def test_s6_lifecycle_dispatches_to_s6_svc(
    s6_scandir, fake_subprocess_run,
) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.start("gateway-coder")
    mgr.stop("gateway-coder")
    mgr.restart("gateway-coder")

    flags = [c[1] for c in fake_subprocess_run if c[0] == "s6-svc"]
    assert flags == ["-u", "-d", "-t"]


def test_s6_is_running_parses_svstat(
    s6_scandir, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess as _sp
    from hermes_cli.service_manager import S6ServiceManager

    def _svstat(cmd, **kw):
        if cmd[0].endswith("/s6-svstat"):
            return _sp.CompletedProcess(cmd, 0, "up (pid 42) 17 seconds\n", "")
        return _sp.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("subprocess.run", _svstat)
    assert S6ServiceManager(scandir=s6_scandir).is_running("gateway-coder") is True

    def _svstat_down(cmd, **kw):
        if cmd[0].endswith("/s6-svstat"):
            return _sp.CompletedProcess(cmd, 0, "down 5 seconds\n", "")
        return _sp.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("subprocess.run", _svstat_down)
    assert S6ServiceManager(scandir=s6_scandir).is_running("gateway-coder") is False
