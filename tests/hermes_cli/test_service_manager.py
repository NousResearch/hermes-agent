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
# _s6_running — must work for unprivileged users, not just root
# ---------------------------------------------------------------------------


def _patch_s6_paths(
    monkeypatch: pytest.MonkeyPatch,
    *,
    comm: str | OSError | None,
    basedir_is_dir: bool,
) -> None:
    """Stub /proc/1/comm and /run/s6/basedir for _s6_running tests."""
    from pathlib import Path as _Path

    real_read_text = _Path.read_text
    real_is_dir = _Path.is_dir

    def fake_read_text(self, *args, **kwargs):  # type: ignore[override]
        if str(self) == "/proc/1/comm":
            if isinstance(comm, OSError):
                raise comm
            if comm is None:
                raise FileNotFoundError(2, "No such file or directory")
            return comm + "\n"
        return real_read_text(self, *args, **kwargs)

    def fake_is_dir(self):  # type: ignore[override]
        if str(self) == "/run/s6/basedir":
            return basedir_is_dir
        return real_is_dir(self)

    monkeypatch.setattr(_Path, "read_text", fake_read_text)
    monkeypatch.setattr(_Path, "is_dir", fake_is_dir)


def test_s6_running_true_when_comm_and_basedir_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.service_manager import _s6_running

    _patch_s6_paths(monkeypatch, comm="s6-svscan", basedir_is_dir=True)
    assert _s6_running() is True


def test_s6_running_false_when_comm_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.service_manager import _s6_running

    # systemd as PID 1, basedir present from some stray s6 install
    _patch_s6_paths(monkeypatch, comm="systemd", basedir_is_dir=True)
    assert _s6_running() is False


def test_s6_running_false_when_basedir_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.service_manager import _s6_running

    # The comm matches but the basedir is missing — e.g. an unrelated
    # process happens to be named "s6-svscan"
    _patch_s6_paths(monkeypatch, comm="s6-svscan", basedir_is_dir=False)
    assert _s6_running() is False


def test_s6_running_false_when_comm_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: /proc/1/exe was unreadable to UID 10000 and
    resolve() silently returned the unresolved path, making detection
    always-False inside the container under the hermes user. The new
    probe must FAIL CLOSED — not raise — when /proc/1/comm can't be
    read.
    """
    from hermes_cli.service_manager import _s6_running

    _patch_s6_paths(
        monkeypatch,
        comm=PermissionError(13, "Permission denied"),
        basedir_is_dir=True,
    )
    assert _s6_running() is False


def test_s6_running_handles_missing_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On macOS / Windows / WSL-without-procfs, /proc/1/comm doesn't
    exist. Must return False, not raise."""
    from hermes_cli.service_manager import _s6_running

    _patch_s6_paths(monkeypatch, comm=None, basedir_is_dir=False)
    assert _s6_running() is False


# ---------------------------------------------------------------------------
# Backend wrappers — kind + registration unsupported on hosts
# ---------------------------------------------------------------------------


def test_systemd_manager_kind_and_registration_unsupported() -> None:
    mgr = SystemdServiceManager()
    assert mgr.kind == "systemd"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo")
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
        mgr.register_profile_gateway("foo")
    assert mgr.list_profile_gateways() == []
    assert isinstance(mgr, ServiceManager)


def test_windows_manager_kind_and_registration_unsupported() -> None:
    mgr = WindowsServiceManager()
    assert mgr.kind == "windows"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo")
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


# ---------------------------------------------------------------------------
# _seed_supervise_skeleton — unit tests
# ---------------------------------------------------------------------------
#
# The skeleton helper pre-creates the dirs and FIFOs that s6-supervise
# would otherwise create as root mode 0700, locking out the
# unprivileged hermes user from every lifecycle op. These tests run
# against tmp_path and assert the produced layout — the live-container
# verification (against real s6-svc / s6-svstat) lives in
# tests/docker/test_s6_profile_gateway_integration.py.


def test_seed_supervise_skeleton_creates_expected_layout(tmp_path) -> None:
    """Verifies the dirs + FIFO + modes the helper lays down."""
    import stat

    from hermes_cli.service_manager import _seed_supervise_skeleton

    svc_dir = tmp_path / "gateway-foo"
    svc_dir.mkdir()

    _seed_supervise_skeleton(svc_dir)

    # Top-level event/ — s6-svlisten1 event subscription dir.
    event = svc_dir / "event"
    assert event.is_dir(), "missing top-level event/"
    assert stat.S_IMODE(event.stat().st_mode) == 0o3730, (
        f"event/ mode = {oct(event.stat().st_mode)}, want 03730"
    )

    # supervise/ dir.
    supervise = svc_dir / "supervise"
    assert supervise.is_dir(), "missing supervise/"
    assert stat.S_IMODE(supervise.stat().st_mode) == 0o755

    # supervise/event/.
    supervise_event = supervise / "event"
    assert supervise_event.is_dir(), "missing supervise/event/"
    assert stat.S_IMODE(supervise_event.stat().st_mode) == 0o3730

    # supervise/control FIFO.
    control = supervise / "control"
    assert control.exists(), "missing supervise/control FIFO"
    assert stat.S_ISFIFO(control.stat().st_mode), (
        "supervise/control must be a FIFO"
    )
    assert stat.S_IMODE(control.stat().st_mode) == 0o660


def test_seed_supervise_skeleton_handles_log_subservice(tmp_path) -> None:
    """When a log/ subdir exists, its supervise tree also gets seeded.

    Without this, ``unregister_profile_gateway``'s rmtree would EACCES
    on the logger's root-owned supervise dir even after the parent
    slot's supervise/ was hermes-owned.
    """
    import stat

    from hermes_cli.service_manager import _seed_supervise_skeleton

    svc_dir = tmp_path / "gateway-foo"
    svc_dir.mkdir()
    (svc_dir / "log").mkdir()  # logger subdir present

    _seed_supervise_skeleton(svc_dir)

    # Logger's own supervise tree is seeded the same way.
    log_event = svc_dir / "log" / "event"
    log_supervise = svc_dir / "log" / "supervise"
    log_supervise_event = log_supervise / "event"
    log_control = log_supervise / "control"

    assert log_event.is_dir()
    assert stat.S_IMODE(log_event.stat().st_mode) == 0o3730
    assert log_supervise.is_dir()
    assert log_supervise_event.is_dir()
    assert log_control.exists() and stat.S_ISFIFO(log_control.stat().st_mode)


def test_seed_supervise_skeleton_skips_when_no_log_subservice(tmp_path) -> None:
    """If log/ isn't present, no logger skeleton is created."""
    from hermes_cli.service_manager import _seed_supervise_skeleton

    svc_dir = tmp_path / "gateway-foo"
    svc_dir.mkdir()

    _seed_supervise_skeleton(svc_dir)

    assert not (svc_dir / "log").exists(), (
        "helper must not synthesize a log/ subdir on its own"
    )


def test_seed_supervise_skeleton_is_idempotent(tmp_path) -> None:
    """Calling the helper twice on the same dir is a no-op the second time.

    Important because s6-supervise may have already opened the FIFO
    when a re-register / reconcile happens; double-creation would
    error out. The helper short-circuits on existence.
    """
    from hermes_cli.service_manager import _seed_supervise_skeleton

    svc_dir = tmp_path / "gateway-foo"
    svc_dir.mkdir()

    _seed_supervise_skeleton(svc_dir)
    _seed_supervise_skeleton(svc_dir)  # must not raise


def test_s6_register_creates_service_dir_and_triggers_scan(
    s6_scandir, fake_subprocess_run,
) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.register_profile_gateway("coder")

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
        "x", extra_env={"FOO": "bar baz", "QUOTED": "a'b"},
    )
    run_text = (s6_scandir / "gateway-x" / "run").read_text()
    # shlex.quote should have wrapped both values
    assert "export FOO='bar baz'" in run_text
    assert "export QUOTED='a'\"'\"'b'" in run_text


def test_s6_register_rejects_invalid_profile_name(s6_scandir) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    with pytest.raises(ValueError):
        mgr.register_profile_gateway("Bad/Name")


def test_s6_register_rejects_duplicate(s6_scandir, fake_subprocess_run) -> None:
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager(scandir=s6_scandir)
    (s6_scandir / "gateway-coder").mkdir(parents=True)
    with pytest.raises(ValueError, match="already registered"):
        mgr.register_profile_gateway("coder")


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
        mgr.register_profile_gateway("coder")
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
    # _run_svc now verifies the slot exists before invoking s6-svc, so
    # we have to pre-seed the dir. In real use the slot is created by
    # register_profile_gateway or the cont-init.d reconciler.
    (s6_scandir / "gateway-coder").mkdir()
    mgr.start("gateway-coder")
    mgr.stop("gateway-coder")
    mgr.restart("gateway-coder")

    flags = [c[1] for c in fake_subprocess_run if c[0] == "s6-svc"]
    assert flags == ["-u", "-d", "-t"]


# ---------------------------------------------------------------------------
# Lifecycle errors — friendly messages, not raw CalledProcessError
# ---------------------------------------------------------------------------


def test_lifecycle_raises_gateway_not_registered_for_missing_slot(
    s6_scandir, fake_subprocess_run,
) -> None:
    """When the service slot doesn't exist, the lifecycle methods
    must raise GatewayNotRegisteredError BEFORE invoking s6-svc, so
    the user sees a clear 'no such gateway' message instead of an
    opaque CalledProcessError stacktrace."""
    from hermes_cli.service_manager import (
        GatewayNotRegisteredError,
        S6ServiceManager,
    )

    mgr = S6ServiceManager(scandir=s6_scandir)
    # No gateway-typo/ directory exists — slot is missing.
    with pytest.raises(GatewayNotRegisteredError) as excinfo:
        mgr.start("gateway-typo")
    assert excinfo.value.profile == "typo"
    assert excinfo.value.service == "gateway-typo"
    msg = str(excinfo.value)
    assert "'typo'" in msg
    assert "hermes profile create typo" in msg
    # And critically: s6-svc was NOT invoked.
    assert not any(c[0] == "s6-svc" for c in fake_subprocess_run)


@pytest.mark.parametrize("action,method_name", [
    ("start", "start"),
    ("stop", "stop"),
    ("restart", "restart"),
])
def test_all_lifecycle_methods_check_for_missing_slot(
    s6_scandir,
    fake_subprocess_run,
    action: str,
    method_name: str,
) -> None:
    """start/stop/restart all check for missing slots the same way."""
    from hermes_cli.service_manager import (
        GatewayNotRegisteredError,
        S6ServiceManager,
    )

    mgr = S6ServiceManager(scandir=s6_scandir)
    with pytest.raises(GatewayNotRegisteredError):
        getattr(mgr, method_name)("gateway-absent")


def test_gateway_not_registered_unprefixed_service_name(s6_scandir) -> None:
    """If the caller passes a name without the 'gateway-' prefix (the
    Protocol allows arbitrary service names), the error still carries
    that name verbatim as the 'profile' so error messages don't
    accidentally strip user-provided text."""
    from hermes_cli.service_manager import (
        GatewayNotRegisteredError,
        S6ServiceManager,
    )

    mgr = S6ServiceManager(scandir=s6_scandir)
    with pytest.raises(GatewayNotRegisteredError) as excinfo:
        mgr.start("not-prefixed")
    assert excinfo.value.profile == "not-prefixed"


def test_lifecycle_raises_s6_command_error_on_subprocess_failure(
    s6_scandir, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When s6-svc itself fails (non-zero exit) — e.g. EACCES on the
    supervise control FIFO — the lifecycle methods translate the
    CalledProcessError into a named S6CommandError carrying the
    return code and stderr."""
    import subprocess as _sp
    from hermes_cli.service_manager import S6CommandError, S6ServiceManager

    # Pre-create the slot so we reach the s6-svc call.
    (s6_scandir / "gateway-coder").mkdir()

    def _fail(cmd, **kw):
        raise _sp.CalledProcessError(
            returncode=111,
            cmd=cmd,
            stderr="s6-svc: fatal: unable to control supervise/control: "
                   "Permission denied\n",
        )
    monkeypatch.setattr("subprocess.run", _fail)

    mgr = S6ServiceManager(scandir=s6_scandir)
    with pytest.raises(S6CommandError) as excinfo:
        mgr.start("gateway-coder")
    assert excinfo.value.service == "gateway-coder"
    assert excinfo.value.action == "start"
    assert excinfo.value.returncode == 111
    assert "Permission denied" in excinfo.value.stderr
    assert "Permission denied" in str(excinfo.value)
    assert "rc=111" in str(excinfo.value)


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
