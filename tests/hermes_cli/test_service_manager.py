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


def test_detect_service_manager_s6_keys_off_s6_running_not_is_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Fly runs s6-overlay as PID 1 in a Firecracker microVM, which
    is not a Docker/Podman container. Gating s6 detection on is_container() made
    the dispatch path inert on Fly, so `hermes gateway restart` spawned a
    foreground gateway that fought the supervised one. Detection must key off
    s6 being PID 1 (`_s6_running`) alone."""
    monkeypatch.setattr(
        "hermes_cli.service_manager._s6_running", lambda: True,
    )
    assert detect_service_manager() == "s6"


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


def test_get_service_manager_returns_s6_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The s6 backend ships in Phase 3 — the factory must return an
    S6ServiceManager when running inside a container."""
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
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.register_profile_gateway("coder")

    svc_dir = s6_scandir / "gateway-coder"
    assert svc_dir.is_dir()
    assert (svc_dir / "type").read_text().strip() == "longrun"

    run_path = svc_dir / "run"
    assert run_path.is_file()
    assert run_path.stat().st_mode & 0o111  # executable
    run_text = run_path.read_text()
    assert "export HOME=/opt/data" in run_text
    assert "hermes -p coder gateway run" in run_text
    assert "s6-setuidgid hermes" in run_text
    # Sentinel marking this as the supervised-child invocation. Without
    # it, the supervised `gateway run` would re-enter the s6 redirect
    # in `_gateway_command_inner` and recurse. See the matching guard
    # in hermes_cli/gateway.py::_gateway_command_inner.
    assert "export HERMES_S6_SUPERVISED_CHILD=1" in run_text

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
    # `1` action directive forwards lines to stdout BEFORE the file
    # destination so the supervised gateway's stdout (including the
    # rich-console banner and plain print() output) reaches docker
    # logs, not just the rotated file. See _render_log_run's docstring
    # for the full output-routing rationale.
    assert "s6-log 1 " in log_text, (
        "log/run must include the `1` action directive before the file "
        "destination so supervised stdout reaches docker logs. Saw: "
        f"{log_text!r}"
    )

    # s6-svscanctl -a was invoked against the scandir
    assert any(
        cmd[0] == "s6-svscanctl" and "-a" in cmd
        and str(s6_scandir) in cmd
        for cmd in fake_subprocess_run
    ), f"s6-svscanctl -a not invoked; saw: {fake_subprocess_run}"


def test_render_run_script_resets_home_before_exec() -> None:

    run_text = S6ServiceManager._render_run_script("coder", {})

    assert "export HOME=/opt/data" in run_text
    assert "exec s6-setuidgid hermes hermes -p coder gateway run --replace" in run_text


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


def test_s6_register_writes_finish_script(
    s6_scandir, fake_subprocess_run,
) -> None:
    """The finish script must be written alongside the run script."""
    mgr = S6ServiceManager(scandir=s6_scandir)
    mgr.register_profile_gateway("coder")

    finish_path = s6_scandir / "gateway-coder" / "finish"
    assert finish_path.is_file()
    assert finish_path.stat().st_mode & 0o111  # executable
    assert "78" in finish_path.read_text()
    assert "125" in finish_path.read_text()


def test_s6_lifecycle_dispatches_to_s6_svc(
    s6_scandir, fake_subprocess_run,
) -> None:
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
    from hermes_cli.service_manager import S6CommandError

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


def test_s6_log_run_mkdir_as_hermes_on_real_dirs(tmp_path) -> None:
    """Root-context setup creates ``$log_dir`` via ``s6-setuidgid hermes mkdir``."""
    import os
    import stat
    import subprocess

    import pytest

    if os.name == "nt":
        pytest.skip("POSIX /bin/sh required")

    hermes_home = tmp_path / "hermes"
    (hermes_home / "logs" / "gateways").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    setuid_recorder = tmp_path / "setuidgid_calls.txt"
    chown_recorder = tmp_path / "chown_calls.txt"

    (bin_dir / "id").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-u" ]; then echo 0; exit 0; fi\n'
        "exit 1\n",
        encoding="utf-8",
    )
    (bin_dir / "s6-setuidgid").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{setuid_recorder.as_posix()}"\n'
        "shift\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    (bin_dir / "chown").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{chown_recorder.as_posix()}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    for name in ("id", "s6-setuidgid", "chown"):
        p = bin_dir / name
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    script_path = tmp_path / "log_run_setup.sh"
    script_path.write_text(
        _log_run_setup_fragment(S6ServiceManager._render_log_run("coder")),
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PATH"] = f"{bin_dir.as_posix()}{os.pathsep}{env.get('PATH', '')}"

    proc = subprocess.run(
        ["/bin/sh", str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    setuid_calls = setuid_recorder.read_text(encoding="utf-8").strip().splitlines()
    assert any(
        c.split()[:3] == ["hermes", "mkdir", "-p"] and "gateways/coder" in c
        for c in setuid_calls
    ), setuid_calls
    assert any(
        c.split()[:3] == ["hermes", "rm", "-f"] and c.rstrip().endswith("/lock")
        for c in setuid_calls
    ), setuid_calls
    assert (hermes_home / "logs" / "gateways" / "coder").is_dir()
    assert (
        not chown_recorder.exists()
        or chown_recorder.read_text(encoding="utf-8").strip() == ""
    )
