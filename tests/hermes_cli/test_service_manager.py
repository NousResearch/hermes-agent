"""Tests for hermes_cli.service_manager — the abstract ServiceManager
protocol, the detect_service_manager() entry point, and the host-side
adapter wrappers (Systemd / Launchd / Windows).

The s6 backend is added in Phase 3; its tests live alongside the
implementation in this same file once that phase ships.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import service_manager as service_manager_module
from hermes_cli.service_manager import (
    LaunchdServiceManager,
    S6Error,
    S6ServiceManager,
    S6TestIsolationError,
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






# ---------------------------------------------------------------------------
# detect_service_manager
# ---------------------------------------------------------------------------






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


# ---------------------------------------------------------------------------
# Lifecycle delegation — wrappers must call through to module-level fns
# ---------------------------------------------------------------------------




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




# ---------------------------------------------------------------------------
# get_service_manager factory
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# S6ServiceManager — unit tests against a tmp-path scandir (no real s6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolution_error", [OSError("io"), RuntimeError("loop")])
def test_s6_test_isolation_denies_when_scandir_identity_cannot_be_resolved(
    monkeypatch, tmp_path, fake_subprocess_run, resolution_error
):
    scandir = tmp_path / "uncertain-scandir"
    scandir.mkdir()
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test")
    monkeypatch.setattr(
        service_manager_module, "S6_DYNAMIC_SCANDIR", tmp_path / "canonical"
    )

    def fail_resolve(self, *args, **kwargs):
        raise resolution_error

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    manager = S6ServiceManager(scandir=scandir)

    with pytest.raises(S6TestIsolationError, match="could not prove"):
        manager.register_profile_gateway("coder", start_now=False)

    assert not (scandir / "gateway-coder").exists()
    assert fake_subprocess_run == []


def test_s6_test_isolation_binds_safe_resolved_scandir_before_mutation(
    monkeypatch, tmp_path, fake_subprocess_run
):
    canonical = tmp_path / "canonical-service"
    canonical.mkdir()
    sentinel = canonical / "sentinel"
    sentinel.write_text("untouched")
    safe = tmp_path / "safe-scandir"
    safe.mkdir()
    alias = tmp_path / "mutable-alias"
    alias.symlink_to(safe, target_is_directory=True)
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test")
    monkeypatch.setattr(service_manager_module, "S6_DYNAMIC_SCANDIR", canonical)
    manager = S6ServiceManager(scandir=alias)
    original_service_dir = manager._service_dir
    retargeted = False

    def retarget_then_resolve(profile):
        nonlocal retargeted
        if not retargeted:
            alias.unlink()
            alias.symlink_to(canonical, target_is_directory=True)
            retargeted = True
        return original_service_dir(profile)

    monkeypatch.setattr(manager, "_service_dir", retarget_then_resolve)

    manager.register_profile_gateway("coder", start_now=False)

    assert sentinel.read_text() == "untouched"
    assert not (canonical / "gateway-coder").exists()
    assert (safe / "gateway-coder").is_dir()
    assert manager.scandir == safe.resolve()


def test_s6_test_isolation_binds_safe_scandir_before_unregister(
    monkeypatch, tmp_path, fake_subprocess_run
):
    canonical = tmp_path / "canonical-service"
    canonical_service = canonical / "gateway-coder"
    canonical_service.mkdir(parents=True)
    canonical_sentinel = canonical_service / "sentinel"
    canonical_sentinel.write_text("untouched")
    safe = tmp_path / "safe-scandir"
    safe_service = safe / "gateway-coder"
    safe_service.mkdir(parents=True)
    alias = tmp_path / "mutable-alias"
    alias.symlink_to(safe, target_is_directory=True)
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test")
    monkeypatch.setattr(service_manager_module, "S6_DYNAMIC_SCANDIR", canonical)
    manager = S6ServiceManager(scandir=alias)
    original_service_dir = manager._service_dir
    retargeted = False

    def retarget_then_resolve(profile):
        nonlocal retargeted
        if not retargeted:
            alias.unlink()
            alias.symlink_to(canonical, target_is_directory=True)
            retargeted = True
        return original_service_dir(profile)

    monkeypatch.setattr(manager, "_service_dir", retarget_then_resolve)

    manager.unregister_profile_gateway("coder")

    assert canonical_sentinel.read_text() == "untouched"
    assert canonical_service.is_dir()
    assert not safe_service.exists()
    assert manager.scandir == safe.resolve()


def test_s6_test_isolation_binds_open_directory_when_resolved_referent_is_retargeted(
    monkeypatch, tmp_path, fake_subprocess_run
):
    canonical = tmp_path / "canonical-service"
    canonical.mkdir()
    sentinel = canonical / "sentinel"
    sentinel.write_text("untouched")
    safe = tmp_path / "safe-scandir"
    safe.mkdir()
    moved = tmp_path / "safe-scandir-moved"
    alias = tmp_path / "mutable-alias"
    alias.symlink_to(safe, target_is_directory=True)
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test")
    monkeypatch.setattr(service_manager_module, "S6_DYNAMIC_SCANDIR", canonical)
    manager = S6ServiceManager(scandir=alias)
    original_service_dir = manager._service_dir
    retargeted = False

    def retarget_referent_then_resolve(profile):
        nonlocal retargeted
        if not retargeted:
            safe.rename(moved)
            safe.symlink_to(canonical, target_is_directory=True)
            retargeted = True
        return original_service_dir(profile)

    monkeypatch.setattr(manager, "_service_dir", retarget_referent_then_resolve)

    manager.register_profile_gateway("coder", start_now=False)

    assert sentinel.read_text() == "untouched"
    assert not (canonical / "gateway-coder").exists()
    assert (moved / "gateway-coder").is_dir()


def test_s6_test_isolation_unregister_uses_open_directory_when_referent_is_retargeted(
    monkeypatch, tmp_path, fake_subprocess_run
):
    canonical = tmp_path / "canonical-service"
    canonical_service = canonical / "gateway-coder"
    canonical_service.mkdir(parents=True)
    sentinel = canonical_service / "sentinel"
    sentinel.write_text("untouched")
    safe = tmp_path / "safe-scandir"
    safe_service = safe / "gateway-coder"
    safe_service.mkdir(parents=True)
    moved = tmp_path / "safe-scandir-moved"
    alias = tmp_path / "mutable-alias"
    alias.symlink_to(safe, target_is_directory=True)
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test")
    monkeypatch.setattr(service_manager_module, "S6_DYNAMIC_SCANDIR", canonical)
    manager = S6ServiceManager(scandir=alias)
    original_service_dir = manager._service_dir
    retargeted = False

    def retarget_referent_then_resolve(profile):
        nonlocal retargeted
        if not retargeted:
            safe.rename(moved)
            safe.symlink_to(canonical, target_is_directory=True)
            retargeted = True
        return original_service_dir(profile)

    monkeypatch.setattr(manager, "_service_dir", retarget_referent_then_resolve)

    manager.unregister_profile_gateway("coder")

    assert sentinel.read_text() == "untouched"
    assert canonical_service.is_dir()
    assert not (moved / "gateway-coder").exists()


def test_s6_test_isolation_denies_resolved_alias_of_canonical_scandir(
    monkeypatch, tmp_path, fake_subprocess_run
):
    canonical = tmp_path / "canonical-service"
    canonical.mkdir()
    sentinel = canonical / "sentinel"
    sentinel.write_text("untouched")
    alias = tmp_path / "canonical-alias"
    alias.symlink_to(canonical, target_is_directory=True)
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test")
    monkeypatch.setattr(service_manager_module, "S6_DYNAMIC_SCANDIR", canonical)
    manager = S6ServiceManager(scandir=alias)

    with pytest.raises(S6TestIsolationError):
        manager.unregister_profile_gateway("coder")

    assert sentinel.read_text() == "untouched"
    assert fake_subprocess_run == []


@pytest.fixture
def s6_scandir(tmp_path):
    """Empty scandir for the S6ServiceManager tests."""
    d = tmp_path / "service"
    d.mkdir()
    return d


def test_s6_register_refuses_canonical_scandir_under_test_isolation_before_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_subprocess_run
) -> None:
    canonical_scandir = tmp_path / "canonical-service"
    canonical_scandir.mkdir()
    monkeypatch.setattr(
        service_manager_module, "S6_DYNAMIC_SCANDIR", canonical_scandir
    )
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test-register")
    manager = S6ServiceManager(scandir=canonical_scandir)

    with pytest.raises(S6TestIsolationError, match="test isolation"):
        manager.register_profile_gateway("coder", start_now=False)

    assert not (canonical_scandir / "gateway-coder").exists()
    assert fake_subprocess_run == []


def test_s6_unregister_refuses_canonical_scandir_under_test_isolation_before_effect(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_subprocess_run
) -> None:
    canonical_scandir = tmp_path / "canonical-service"
    service_dir = canonical_scandir / "gateway-coder"
    service_dir.mkdir(parents=True)
    sentinel = service_dir / "sentinel"
    sentinel.write_bytes(b"owned-before-test")
    monkeypatch.setattr(
        service_manager_module, "S6_DYNAMIC_SCANDIR", canonical_scandir
    )
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test-unregister")
    manager = S6ServiceManager(scandir=canonical_scandir)

    with pytest.raises(S6TestIsolationError, match="test isolation"):
        manager.unregister_profile_gateway("coder")

    assert sentinel.read_bytes() == b"owned-before-test"
    assert fake_subprocess_run == []


def test_s6_test_isolation_guard_allows_noncanonical_tmp_scandir(
    tmp_path, s6_scandir, monkeypatch: pytest.MonkeyPatch, fake_subprocess_run
) -> None:
    canonical_scandir = tmp_path / "canonical-service"
    canonical_scandir.mkdir()
    monkeypatch.setattr(
        service_manager_module, "S6_DYNAMIC_SCANDIR", canonical_scandir
    )
    monkeypatch.setenv("HERMES_TEST_ISOLATION", "run:test-temp")
    manager = S6ServiceManager(scandir=s6_scandir)

    manager.register_profile_gateway("coder", start_now=False)

    assert (s6_scandir / "gateway-coder").is_dir()
    assert len(fake_subprocess_run) == 1
    assert fake_subprocess_run[0][:2] == ["s6-svscanctl", "-a"]
    assert "/fd/" in fake_subprocess_run[0][2]


def test_s6_live_guard_is_inactive_outside_test_context(
    s6_scandir, monkeypatch: pytest.MonkeyPatch, fake_subprocess_run
) -> None:
    monkeypatch.setattr(service_manager_module, "S6_DYNAMIC_SCANDIR", s6_scandir)
    monkeypatch.delenv("HERMES_TEST_ISOLATION", raising=False)
    manager = S6ServiceManager(scandir=s6_scandir)

    manager.register_profile_gateway("coder", start_now=False)

    assert (s6_scandir / "gateway-coder").is_dir()
    assert fake_subprocess_run == [["s6-svscanctl", "-a", str(s6_scandir)]]


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
    """Verifies the dirs + FIFO the helper lays down."""
    import stat

    from hermes_cli.service_manager import _seed_supervise_skeleton

    svc_dir = tmp_path / "gateway-foo"
    svc_dir.mkdir()

    _seed_supervise_skeleton(svc_dir)

    # Top-level event/ — s6-svlisten1 event subscription dir.
    event = svc_dir / "event"
    assert event.is_dir(), "missing top-level event/"

    # supervise/ dir.
    supervise = svc_dir / "supervise"
    assert supervise.is_dir(), "missing supervise/"
    assert stat.S_IMODE(supervise.stat().st_mode) == 0o755

    # supervise/event/.
    supervise_event = supervise / "event"
    assert supervise_event.is_dir(), "missing supervise/event/"

    # supervise/control FIFO.
    control = supervise / "control"
    assert control.exists(), "missing supervise/control FIFO"
    assert stat.S_ISFIFO(control.stat().st_mode), (
        "supervise/control must be a FIFO"
    )
    assert stat.S_IMODE(control.stat().st_mode) == 0o660


@pytest.mark.linux_only
def test_seed_supervise_skeleton_sets_setgid_on_event_dirs(tmp_path) -> None:
    """The event dirs carry setgid so s6-supervise's EEXIST path leaves them alone.

    Linux-only because the assertion is about what ``chmod`` does, and that
    differs by kernel: BSD (macOS) silently drops ``S_ISGID`` from a directory
    unless the caller is root or a member of the directory's group, so the same
    correct helper produces 01730 there. s6 only ever runs on Linux — inside
    s6-overlay's stage2 as root with umask 0 — so Linux is the host whose
    answer matters.
    """
    import stat

    from hermes_cli.service_manager import _seed_supervise_skeleton

    svc_dir = tmp_path / "gateway-foo"
    svc_dir.mkdir()

    _seed_supervise_skeleton(svc_dir)

    for rel in ("event", "supervise/event"):
        mode = stat.S_IMODE((svc_dir / rel).stat().st_mode)
        assert mode == 0o3730, f"{rel}/ mode = {oct(mode)}, want 0o3730"










def test_render_run_script_uses_replace_to_take_over_stale_holder() -> None:
    """NS-505: the supervised gateway must exec ``gateway run --replace``.

    Without ``--replace`` a gateway started OUTSIDE s6 (a stray shell
    ``hermes gateway run``, an agent action, the Open WebUI helper) holds
    the per-HERMES_HOME PID lock; the supervised slot then execs a bare
    ``gateway run``, hits the "Another gateway instance is already
    running" guard, exits non-zero, and s6 restarts it — a restart loop
    that never binds. ``--replace`` makes the supervised gateway reap the
    stale holder and win, so s6 is authoritative for the slot.

    Covers both the default (root HERMES_HOME, no ``-p``) and named-profile
    render paths.
    """
    default_text = S6ServiceManager._render_run_script("default", {})
    # Root profile: bare `hermes gateway run --replace` (no -p flag).
    assert "hermes gateway run --replace" in default_text
    assert "hermes -p default" not in default_text
    # Every exec line that launches the gateway must carry --replace, so
    # neither the non-root nor the privilege-drop branch can spin.
    gateway_execs = [
        line for line in default_text.splitlines()
        if "gateway run" in line
    ]
    assert gateway_execs, "no gateway run exec line rendered"
    assert all("--replace" in line for line in gateway_execs), (
        f"a gateway run line is missing --replace: {gateway_execs}"
    )

    named_text = S6ServiceManager._render_run_script("coder", {})
    named_execs = [
        line for line in named_text.splitlines() if "gateway run" in line
    ]
    assert named_execs
    assert all("--replace" in line for line in named_execs), (
        f"a named-profile gateway run line is missing --replace: {named_execs}"
    )


def test_render_finish_script_exits_125_on_ex_config() -> None:
    """The finish script must translate exit 78 (EX_CONFIG) into exit 125
    (permanent failure) so s6 stops restarting on fatal config errors.
    See #51228."""
    text = S6ServiceManager._render_finish_script()
    assert '[ "$1" = "78" ]' in text
    assert "exit 125" in text
    assert "exit 0" in text


def test_render_finish_script_does_not_restart_on_clean_exit(tmp_path) -> None:
    """Behavioral: the rendered finish script, executed for each run-exit
    code, must exit 125 (no restart) for clean exit 0 and EX_CONFIG 78,
    and exit 0 (restart) for genuine crashes (#76435 — restart-on-normal-
    exit turned a supervised gateway into a reconnect storm)."""
    import subprocess

    script = tmp_path / "finish"
    script.write_text(S6ServiceManager._render_finish_script())
    script.chmod(0o755)

    def finish_exit(run_exit_code: int) -> int:
        proc = subprocess.run(["sh", str(script), str(run_exit_code)],
                              capture_output=True)
        return proc.returncode

    assert finish_exit(0) == 125   # clean stop — no restart
    assert finish_exit(78) == 125  # fatal config — no restart
    assert finish_exit(1) == 0     # crash — s6 restarts
    assert finish_exit(137) == 0   # SIGKILL crash — s6 restarts






# ---------------------------------------------------------------------------
# Lifecycle errors — friendly messages, not raw CalledProcessError
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# S6 stop writes a planned-stop marker (issue #42675)
#
# `hermes gateway stop` inside a container dispatches through
# S6ServiceManager.stop() -> `s6-svc -d`, which SIGTERMs the gateway.
# That SIGTERM is indistinguishable from the one s6/Docker sends on a
# container restart unless we mark the intentional stop first. Without
# the marker, the gateway's shutdown handler can't tell an operator
# stop from a restart kill, and the gateway_state=stopped suppression
# (run.py) would never engage for explicit stops.
# ---------------------------------------------------------------------------


def _log_run_setup_fragment(rendered: str) -> str:
    """Keep mkdir/rm setup from ``_render_log_run``; stop before ``s6-log``."""
    keep: list[str] = []
    for line in rendered.splitlines(keepends=True):
        if line.startswith("#!/") or "shellcheck" in line:
            continue
        if "s6-log" in line:
            break
        keep.append(line)
    return "#!/bin/sh\n" + "".join(keep)


def test_s6_log_run_creates_leaf_as_hermes_without_chown(
    s6_scandir, fake_subprocess_run,
) -> None:
    """log/run must not root-chown/unlink volume paths; create leaf as hermes.

    #45258 parent ownership is stage2's job (``logs/gateways`` seeded as
    hermes). Restartable log/run must not pathname-chown or pathname-rm a
    hermes-writable tree from root — that is a symlink TOCTOU hole.
    """
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.register_profile_gateway("coder")

    log_text = (s6_scandir / "gateway-coder" / "log" / "run").read_text()

    assert not any(line.lstrip().startswith("chown ") for line in log_text.splitlines()), (
        "restartable log/run must not invoke chown on hermes-writable paths; "
        f"saw: {log_text!r}"
    )
    assert 's6-setuidgid hermes mkdir -p "$log_dir"' in log_text
    assert 's6-setuidgid hermes rm -f "$log_dir/lock"' in log_text
    assert 'else\n  mkdir -p "$log_dir"\n  rm -f "$log_dir/lock"\nfi\n' in log_text
    # Lock cleanup must not remain a bare root-context pathname op after fi.
    after_fi = log_text.split("fi\n", 1)[-1]
    assert 'rm -f "$log_dir/lock"' not in after_fi

    mkdir_as_hermes_idx = log_text.index('s6-setuidgid hermes mkdir -p "$log_dir"')
    rm_as_hermes_idx = log_text.index('s6-setuidgid hermes rm -f "$log_dir/lock"')
    exec_idx = log_text.index("s6-log 1 ")
    assert mkdir_as_hermes_idx < rm_as_hermes_idx < exec_idx

    # Runtime path expansion, never a baked-in absolute path.
    assert '/opt/data/logs/gateways"' not in log_text


def test_s6_log_run_never_invokes_chown_with_symlinked_log_dir(tmp_path) -> None:
    """Symlinked ``$log_dir`` must not redirect root chown/rm to the referent."""
    import os
    import stat
    import subprocess
    import threading
    import time

    import pytest

    if os.name == "nt":
        pytest.skip("POSIX symlink + /bin/sh required")

    hermes_home = tmp_path / "hermes"
    gateways = hermes_home / "logs" / "gateways"
    gateways.mkdir(parents=True)
    leaf = gateways / "coder"
    leaf.mkdir()

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "marker").write_text("keep", encoding="utf-8")
    (victim / "lock").write_text("keep-lock", encoding="utf-8")
    before = victim.stat()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = tmp_path / "chown_calls.txt"
    (bin_dir / "chown").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{recorder.as_posix()}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    # Pretend we are root so the script takes the s6-setuidgid setup path.
    # Mark the drop so fake rm can refuse unlink outside HERMES_HOME the way
    # a real hermes uid cannot delete a foreign root-owned lock.
    (bin_dir / "id").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-u" ]; then echo 0; exit 0; fi\n'
        "exit 1\n",
        encoding="utf-8",
    )
    (bin_dir / "s6-setuidgid").write_text(
        "#!/bin/sh\n"
        "shift\n"
        'HERMES_TEST_DROPPED=1 exec "$@"\n',
        encoding="utf-8",
    )
    real_rm = "/bin/rm"
    (bin_dir / "rm").write_text(
        "#!/bin/sh\n"
        # Privilege-dropped: no-op. Models that hermes cannot unlink a foreign
        # root-owned lock outside the volume; avoids a realpath/rm TOCTOU in
        # the test double itself. Root-context: real rm — a residual bare
        # ``rm -f "$log_dir/lock"`` would delete victim/lock via the symlink.
        'if [ -n "$HERMES_TEST_DROPPED" ]; then\n'
        "  exit 0\n"
        "fi\n"
        f'exec {real_rm} "$@"\n',
        encoding="utf-8",
    )
    for name in ("chown", "id", "s6-setuidgid", "rm"):
        p = bin_dir / name
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    script_path = tmp_path / "log_run_setup.sh"
    script_path.write_text(
        _log_run_setup_fragment(S6ServiceManager._render_log_run("coder")),
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    stop = threading.Event()

    def _clear_leaf() -> None:
        if leaf.is_symlink():
            leaf.unlink()
        elif leaf.is_dir():
            leaf.rmdir()
        elif leaf.exists():
            leaf.unlink()

    def _swap_race() -> None:
        # Alternate leaf between a real dir and a symlink to the victim while
        # the setup fragment runs — proves there is no privileged chown/rm
        # window to win, unlike a check-then-use preflight.
        while not stop.is_set():
            try:
                _clear_leaf()
                leaf.symlink_to(victim)
                time.sleep(0.001)
                _clear_leaf()
                leaf.mkdir()
            except OSError:
                pass
            time.sleep(0.001)

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PATH"] = f"{bin_dir.as_posix()}{os.pathsep}{env.get('PATH', '')}"

    racer = threading.Thread(target=_swap_race, daemon=True)
    racer.start()
    try:
        for _ in range(40):
            proc = subprocess.run(
                ["/bin/sh", str(script_path)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, (proc.stdout, proc.stderr)
            assert (victim / "lock").is_file(), "symlinked leaf must not let root unlink victim/lock"
    finally:
        stop.set()
        racer.join(timeout=2)

    assert not recorder.exists() or recorder.read_text(encoding="utf-8").strip() == ""
    after = victim.stat()
    assert after.st_uid == before.st_uid
    assert after.st_gid == before.st_gid
    assert (victim / "marker").read_text(encoding="utf-8") == "keep"
    assert (victim / "lock").read_text(encoding="utf-8") == "keep-lock"


