"""After ``hermes update``: a notice when this home still runs an old ``hermes dashboard`` / ``hermes serve``
backend, unless a backend migration retired it.

The notice looks only when the update's own inventory or restart phase saw such a backend, and then reuses
the ``hermes gateway migrate --backends`` report, so a real old backend is started in a temp home and the
notice is read from a child process of that home. It changes nothing, and a failing detection never fails
the update. (An update that saw no such backend: ``test_update_verify_without_old_backend.py``.)
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
from tests.gateway.fixtures.local_recovery_probe import child_env

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = "gateway migrate --backends --apply --dry-run"  # after the profile-qualified `hermes`
# The child prints the notice for a plan that saw a manual serve. On stderr it lists every process it
# started and every file it opened for writing, renamed, removed or created (audit events).
_NOTICE = """
import json, os, sys
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_WRITES = ("os.rename", "os.replace", "os.remove", "os.rmdir", "os.mkdir", "os.truncate", "sqlite3.connect")
def _hook(event, args):
    if event == "subprocess.Popen":
        print("RUN " + json.dumps([str(a) for a in args[1]]), file=sys.stderr)
    elif event in _WRITES or (event == "open" and isinstance(args[2], int) and args[2] & _WRITE_FLAGS):
        print("WRITE " + json.dumps([event, str(args[0])]), file=sys.stderr)
sys.addaudithook(_hook)
from hermes_cli.update_backend_notice import print_old_backend_notice
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
print_old_backend_notice(UpdatePlan(runtimes=[RuntimeRecord("serve", "default", int(sys.argv[1]),
                                                             "manual-serve")]), None)
"""
# The only systemctl verbs the report may use: it asks systemd, never tells it anything.
_SYSTEMCTL_READS = {"show", "list-unit-files"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _home(tmp_path):
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    config = {"model": {"provider": "custom", "default": "no-inference", "base_url": "http://127.0.0.1:1/v1"}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(REPO_ROOT),
               PYTHONUNBUFFERED="1", HERMES_DISABLE_LAZY_INSTALLS="1",
               PYTHONDONTWRITEBYTECODE="1")  # the write audit below must see only what Hermes writes
    return home, env


def _notice(env, pid=0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _NOTICE, str(pid)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def _audit(result, kind: str) -> list[list[str]]:
    return [json.loads(line.split(" ", 1)[1]) for line in result.stderr.splitlines() if line.startswith(kind + " ")]


@pytest.mark.platforms("linux")
def test_an_old_backend_of_this_home_gets_one_paragraph_with_the_plan_command_and_nothing_changes(tmp_path):
    home, env = _home(tmp_path)
    port = _free_port()
    backend = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "serve", "--port", str(port)],
                               cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready = threading.Event()

    def pump():
        for line in backend.stdout:
            if "HERMES_BACKEND_READY" in line:
                ready.set()

    threading.Thread(target=pump, daemon=True).start()
    try:
        assert ready.wait(120)
        config_before = (home / "config.yaml").read_bytes()
        result = _notice(env, backend.pid)
        assert backend.poll() is None  # nothing was stopped or restarted
        assert (home / "config.yaml").read_bytes() == config_before
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=30)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(timeout=5)
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert out.startswith("\nFound an old Hermes backend for this profile: `hermes serve` ("), out
    assert f"PID {backend.pid}" in out and f"port {port}" in out
    assert "second copy of Hermes" in out
    # HERMES_HOME is a root of its own (not HOME's default), so the command pins it.
    assert out.endswith(f"would do (this changes nothing): HERMES_HOME={shlex.quote(str(home))} hermes -p default "
                        f"{PLAN}\n"), out
    assert len([line for line in out.splitlines() if line.strip()]) == 1  # one paragraph
    assert _audit(result, "WRITE") == []  # no file written, moved, removed or created
    started = _audit(result, "RUN")
    assert started  # the report ran
    for argv in started:
        name = os.path.basename(argv[0])
        assert name == "ps" or (name == "systemctl" and set(argv) & _SYSTEMCTL_READS), argv


@pytest.mark.platforms("linux")
def test_a_plan_that_saw_a_backend_of_another_home_prints_nothing(tmp_path):
    _, env = _home(tmp_path)
    result = _notice(env)
    assert (result.returncode, result.stdout) == (0, ""), result.stderr


@pytest.mark.parametrize("plan,restarted,saw", [
    (UpdatePlan(runtimes=[RuntimeRecord("dashboard", "default", 7, "systemd")]), [], True),
    (UpdatePlan(runtimes=[RuntimeRecord("serve", "work", 7, "manual-serve")]), [], True),
    (UpdatePlan(runtimes=[RuntimeRecord("serve", "default", 7, "desktop")]), [], False),
    (UpdatePlan(runtimes=[RuntimeRecord("serve", "default", 7, "desktop-ssh")]), [], False),
    (UpdatePlan(runtimes=[RuntimeRecord("gateway", "default", 7, "systemd")]), ["hermes-gateway"], False),
    (None, ["hermes-dashboard.service"], True),
    (None, ["user/hermes-serve-work"], True),  # no inventory: "work" is not known to be another profile
    (None, [], False),
    # This home is the default profile; "work" is a sibling the inventory knows.
    (UpdatePlan(profiles=["default", "work"], runtimes=[RuntimeRecord("serve", "work", 7, "manual-serve")]), [], False),
    (UpdatePlan(profiles=["default", "work"]), ["user/hermes-dashboard-work"], False),
    (UpdatePlan(profiles=["default", "work"]), ["hermes-dashboard"], True),
    (UpdatePlan(profiles=["default", "work"], runtimes=[RuntimeRecord("serve", "work", 7, "manual-serve"),
                                                        RuntimeRecord("dashboard", "default", 8, "systemd")]), [], True),
])
def test_only_an_update_that_met_a_backend_no_desktop_owns_looks_for_one(plan, restarted, saw):
    from hermes_cli.profiles import get_active_profile_name
    from hermes_cli.update_backend_notice import update_saw_old_backend
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome
    assert get_active_profile_name() == "default"  # the test home is the default profile
    restart = _GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=restarted,
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set())
    assert update_saw_old_backend(plan, restart) is saw


def test_a_failing_detection_prints_nothing_and_never_raises(monkeypatch, capsys):
    import hermes_cli.gateway_migrate_backends as backends
    from hermes_cli.update_backend_notice import print_old_backend_notice

    def broken():
        raise OSError("scan failed")

    monkeypatch.setattr(backends, "collect_backends", broken)
    print_old_backend_notice(UpdatePlan(runtimes=[RuntimeRecord("serve", "default", 7, "manual-serve")]), None)
    assert capsys.readouterr().out == ""


def _notice_for(home, monkeypatch) -> str:
    import hermes_cli.gateway_migrate_backends as backends
    from hermes_cli.gateway_migrate_backends import BackendItem
    from hermes_cli.update_backend_notice import old_backend_notice
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(backends, "collect_backends", lambda: [
        BackendItem("manual", True, pid=7, argv=["hermes", "serve", "--port", "9119"], this_home=True)])
    notice = old_backend_notice(home)
    assert notice.count(PLAN) == 1 and notice.endswith(PLAN)
    return notice.split("would do (this changes nothing): ", 1)[1]


def _standard_root(tmp_path, monkeypatch):
    """Point HOME (and LOCALAPPDATA) at *tmp_path* and HERMES_HOME at its platform default root."""
    from hermes_constants import _get_platform_default_hermes_home
    for name in ("HOME", "USERPROFILE", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path))
    root = _get_platform_default_hermes_home()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@pytest.mark.parametrize("profile,command", [("default", "hermes -p default"), ("work", "hermes -p work")])
def test_the_plan_command_names_the_profile_the_way_the_migration_prints_its_steps(
        profile, command, tmp_path, monkeypatch):
    """A bare `hermes` would follow a sticky `hermes profile use` choice, not the profile that was updated."""
    root = _standard_root(tmp_path, monkeypatch)
    home = root if profile == "default" else root / "profiles" / profile
    assert _notice_for(home, monkeypatch) == f"{command} {PLAN}"


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("profile", ["default", "work"])
def test_a_root_other_than_the_default_is_pinned_and_the_printed_line_runs_as_one_command(
        profile, tmp_path, monkeypatch):
    """`hermes -p NAME` resolves against the shell's own HERMES_HOME (the platform default when unset): a home
    under another root (HERMES_HOME=/srv/team-hermes) is named with that root, quoted for the shell."""
    _standard_root(tmp_path / "account", monkeypatch)
    root = tmp_path / "srv" / "team hermes's root"
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").parent.mkdir(parents=True)
    (root / "config.yaml").write_text("{}", encoding="utf-8")
    line = _notice_for(root if profile == "default" else root / "profiles" / profile, monkeypatch)
    assert line.startswith("HERMES_HOME=") and line.endswith(f" hermes -p {profile} {PLAN}"), line
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "hermes"  # records what the shell hands `hermes`
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$HERMES_HOME" "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}  # the shell that types it has no HERMES_HOME of its own
    ran = subprocess.run(["/bin/sh", "-c", line], env=env, capture_output=True, text=True, timeout=30, check=True)
    assert ran.stdout.splitlines() == [str(root), "-p", profile, *PLAN.split()]


@pytest.mark.platforms("windows")
def test_a_root_other_than_the_default_is_pinned_for_powershell(tmp_path, monkeypatch):
    _standard_root(tmp_path / "account", monkeypatch)
    root = tmp_path / "team hermes's root"
    monkeypatch.setenv("HERMES_HOME", str(root))
    root.mkdir()
    (root / "config.yaml").write_text("{}", encoding="utf-8")
    line = _notice_for(root, monkeypatch)
    assert line.startswith("$env:HERMES_HOME='") and line.endswith(f"'; hermes -p default {PLAN}"), line
    script = "function hermes { [Console]::Out.WriteLine($env:HERMES_HOME); $args | ForEach-Object { " \
             "[Console]::Out.WriteLine($_) } }; " + line
    ran = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True, timeout=60, check=True)
    assert ran.stdout.splitlines() == [str(root), "-p", "default", *PLAN.split()]


@pytest.mark.parametrize("state,stopped_unit_named", [
    (None, True), ("running", False), ("done", False), ("failed", False), ("rolled_back", True), ("restored", True),
    ("unreadable", False),
])
def test_a_stopped_unit_a_migration_retired_is_not_named_but_a_running_backend_always_is(
        state, stopped_unit_named, tmp_path, monkeypatch):
    """The manifest is the one the migration writes: while it says the old backend stays down (or cannot be
    read), a stopped unit it left `static` is not a leftover; a running backend is named in every state."""
    import hermes_cli.gateway_migrate_backends as backends
    import hermes_cli.gateway_migrate_backends_host as backends_host
    from hermes_cli.gateway_migrate_backends import BackendItem
    from hermes_cli.gateway_migrate_backends_apply import MANIFEST_NAME, write_manifest
    from hermes_cli.update_backend_notice import old_backend_notice
    if state == "unreadable":
        (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    elif state is not None:
        write_manifest(tmp_path, {"state": state})
    stopped = BackendItem("systemd", False, unit="hermes-dashboard.service", scope="user", state="static",
                          argv=["hermes", "dashboard"], this_home=True)
    running = BackendItem("manual", True, pid=7, argv=["hermes", "serve", "--port", "9119"], this_home=True)
    monkeypatch.setattr(backends_host.SystemHost, "unit_environment",
                        lambda self, unit, scope: ({"HERMES_HOME": str(tmp_path)}, None))
    monkeypatch.setattr(backends, "collect_backends", lambda: [stopped])
    assert (old_backend_notice(tmp_path) is not None) is stopped_unit_named
    monkeypatch.setattr(backends, "collect_backends", lambda: [stopped, running])
    assert "PID 7" in old_backend_notice(tmp_path)


def test_an_update_with_no_serve_or_dashboard_candidate_resolves_nothing_else(monkeypatch):
    """The gate answers from the update's own lists first: with no candidate it never asks which profile
    is active (nor anything else)."""
    from hermes_cli import profiles
    from hermes_cli.update_backend_notice import update_saw_old_backend

    def unexpected():
        raise AssertionError("resolved the active profile without a candidate")

    monkeypatch.setattr(profiles, "get_active_profile_name", unexpected)
    plan = UpdatePlan(profiles=["default", "work"], runtimes=[RuntimeRecord("gateway", "default", 7, "systemd"),
                                                             RuntimeRecord("serve", "default", 8, "desktop")])
    assert update_saw_old_backend(plan, _restart(["hermes-gateway", "user/hermes-gateway-work"])) is False


def _restart(units):
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome
    return _GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[1234], restarted_services=units,
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set())


@pytest.mark.parametrize("broken", ["gate", "report"])
def test_a_notice_that_raises_never_fails_the_update(broken, monkeypatch, capsys):
    """The whole notice (the gate and the report) is optional: the post-update check still finishes,
    prints its fleet lines and clears the restart marker."""
    import hermes_cli.gateway_migrate_backends as backends
    from hermes_cli import profiles
    from hermes_cli import update_cmd, update_cmd_fleet, update_cmd_maint, update_receipt
    from hermes_cli import update_cmd_fleet_verify as fleet_verify

    def boom(*args, **kwargs):
        raise RuntimeError(f"{broken} failed")

    monkeypatch.setattr(*((profiles, "get_active_profile_name", boom) if broken == "gate"
                          else (backends, "collect_backends", boom)))
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_cmd_maint, "_refresh_dashboard_after_update", lambda **kwargs: set())
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **kwargs: {})
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    monkeypatch.setattr(fleet_verify, "_collect_fleet_snapshot", lambda outcome, rows_expected: [
        {"profile": "default", "pid": 1234, "state": "current"}])
    cleared = []
    monkeypatch.setattr(update_cmd_fleet, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))
    plan = UpdatePlan(profiles=["default"], runtimes=[RuntimeRecord("gateway", "default", 1234, "systemd"),
                                                      RuntimeRecord("serve", "default", 4321, "manual-serve")])
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        fleet_verify._verify_fleet_after_update(
            _restart(["hermes-gateway"]), _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    out = capsys.readouterr().out
    assert "default (pid 1234)" in out and "old Hermes backend" not in out
    assert cleared == [True]


def test_stopped_sibling_user_unit_is_not_named_while_this_homes_unit_is(tmp_path, monkeypatch):
    """Stopped user units use the planner's ownership check: `-p/--profile` in ExecStart and a later
    `EnvironmentFile=` overriding `HERMES_HOME` exclude sibling units just as the dry run does."""
    import hermes_cli.gateway_migrate_backends as backends
    import hermes_cli.gateway_migrate_backends_host as backends_host
    from hermes_cli.gateway_migrate_backends import BackendItem
    from hermes_cli.update_backend_notice import old_backend_notice

    root = _standard_root(tmp_path, monkeypatch)
    work_home = root / "profiles" / "work"
    sibling_env = tmp_path / "work.env"
    sibling_env.write_text(f"HERMES_HOME={work_home}\n", encoding="utf-8")

    units = {
        "hermes-dashboard-work.service": (
            "Environment=\n"
            "EnvironmentFiles=\n"
            "ExecStart={ path=/usr/bin/hermes ; argv[]=hermes -p work dashboard --port 9119 ; }\n"
        ),
        "hermes-dashboard-envfile.service": (
            f"Environment=HERMES_HOME={root}\n"
            f"EnvironmentFiles={sibling_env} (ignore_errors=no)\n"
            "ExecStart={ path=/usr/bin/hermes ; argv[]=hermes dashboard --port 9120 ; }\n"
        ),
        "hermes-dashboard.service": (
            f"Environment=HERMES_HOME={root}\n"
            "EnvironmentFiles=\n"
            "ExecStart={ path=/usr/bin/hermes ; argv[]=hermes dashboard --port 9119 ; }\n"
        ),
    }

    def fake_systemctl(scope, args, **kwargs):
        if args[:1] == ["show-environment"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:1] == ["show"] and len(args) >= 2 and args[1] in units:
            return subprocess.CompletedProcess(args, 0, stdout=units[args[1]], stderr="")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="not found")

    monkeypatch.setattr(backends_host, "_systemctl", fake_systemctl)

    sibling_profile = BackendItem("systemd", False, unit="hermes-dashboard-work.service", scope="user",
                                  state="enabled", hermes_home=None, this_home=None)
    sibling_envfile = BackendItem("systemd", False, unit="hermes-dashboard-envfile.service", scope="user",
                                  state="enabled", hermes_home=str(root), this_home=True)
    own_unit = BackendItem("systemd", False, unit="hermes-dashboard.service", scope="user",
                           state="enabled", hermes_home=str(root), this_home=True)

    monkeypatch.setattr(backends, "collect_backends", lambda: [sibling_profile])
    assert old_backend_notice(root) is None

    monkeypatch.setattr(backends, "collect_backends", lambda: [sibling_envfile])
    assert old_backend_notice(root) is None

    monkeypatch.setattr(backends, "collect_backends", lambda: [sibling_profile, sibling_envfile, own_unit])
    notice = old_backend_notice(root)
    assert notice is not None
    assert "hermes-dashboard.service" in notice
    assert "hermes-dashboard-work.service" not in notice
    assert "hermes-dashboard-envfile.service" not in notice


def test_stopped_system_launchd_daemon_without_hermes_home_is_not_named(tmp_path, monkeypatch):
    """Only stopped systemd user units use the planner's `_for_this_home`; a stopped system-scope
    launchd daemon with no HERMES_HOME must not be attributed to this user's default home."""
    import types
    import hermes_cli.gateway_migrate_backends as backends
    from hermes_cli.gateway_migrate_backends import BackendItem
    from hermes_cli.update_backend_notice import old_backend_notice

    root = _standard_root(tmp_path, monkeypatch)
    try:
        import pwd
        monkeypatch.setattr(pwd, "getpwnam", lambda _user: types.SimpleNamespace(pw_dir=str(tmp_path)))
    except ImportError:
        pass
    daemon = BackendItem("launchd", False, unit="com.example.hermes-dashboard", scope="system",
                         hermes_home=None, this_home=None)
    monkeypatch.setattr(backends, "collect_backends", lambda: [daemon])
    assert old_backend_notice(root) is None


def test_notice_runs_after_clearing_restart_marker_and_still_prints_when_restart_incomplete(monkeypatch, capsys):
    """On a complete restart the notice runs only after `_clear_fleet_restart_pending_marker()`; on an
    incomplete restart the marker stays armed and the notice still prints before returning."""
    from hermes_cli import (
        update_backend_notice,
        update_cmd,
        update_cmd_fleet,
        update_cmd_fleet_verify as fleet_verify,
        update_cmd_maint,
        update_receipt,
    )
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome

    order: list[str] = []
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_cmd_maint, "_refresh_dashboard_after_update", lambda **kwargs: set())
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **kwargs: {})
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: order.append("migrate"))
    monkeypatch.setattr(fleet_verify, "_collect_fleet_snapshot", lambda outcome, rows_expected: [
        {"profile": "default", "pid": 1234, "state": "current"}])
    monkeypatch.setattr(update_cmd_fleet, "_clear_fleet_restart_pending_marker", lambda: order.append("clear"))

    def fake_notice():
        order.append("notice")
        return "Found an old Hermes backend for this profile: stub."

    monkeypatch.setattr(update_backend_notice, "old_backend_notice", fake_notice)
    plan = UpdatePlan(profiles=["default"], runtimes=[RuntimeRecord("gateway", "default", 1234, "systemd"),
                                                      RuntimeRecord("serve", "default", 4321, "manual-serve")])

    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        fleet_verify._verify_fleet_after_update(
            _restart(["hermes-gateway"]), _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    out = capsys.readouterr().out
    assert order == ["clear", "migrate", "notice"]
    assert "Found an old Hermes backend for this profile: stub." in out

    order.clear()
    incomplete = _GatewayRestartOutcome(
        incomplete=True, phase_errors=[], pre_restart_gateway_pids=[1234], restarted_services=["hermes-gateway"],
        failed_or_stale_units=["hermes-dashboard"], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set())
    monkeypatch.setattr(fleet_verify, "_record_owed_gateway_inventory", lambda *args, **kwargs: None)
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        fleet_verify._verify_fleet_after_update(
            incomplete, _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    out = capsys.readouterr().out
    assert order == ["notice"]
    assert "Found an old Hermes backend for this profile: stub." in out


def test_keyboard_interrupt_in_notice_leaves_every_required_step_done(monkeypatch):
    """Ctrl-C during the notice wait cannot skip any required post-update step on either path."""
    from hermes_cli import (
        update_backend_notice,
        update_cmd,
        update_cmd_fleet,
        update_cmd_fleet_verify as fleet_verify,
        update_cmd_maint,
        update_receipt,
    )
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome

    steps: list[str] = []
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_cmd_maint, "_refresh_dashboard_after_update", lambda **kwargs: set())
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **kwargs: {})
    monkeypatch.setattr(fleet_verify, "_collect_fleet_snapshot", lambda outcome, rows_expected: [
        {"profile": "default", "pid": 1234, "state": "current"}])
    monkeypatch.setattr(update_receipt, "finalize_update_receipt", lambda *args, **kwargs: steps.append("receipt"))
    monkeypatch.setattr(update_cmd_fleet, "_clear_fleet_restart_pending_marker", lambda: steps.append("clear"))
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: steps.append("migrate"))

    def interrupt_notice(*args, **kwargs):
        steps.append("notice")
        raise KeyboardInterrupt

    monkeypatch.setattr(update_backend_notice, "print_old_backend_notice", interrupt_notice)
    plan = UpdatePlan(profiles=["default"], runtimes=[RuntimeRecord("gateway", "default", 1234, "systemd"),
                                                      RuntimeRecord("serve", "default", 4321, "manual-serve")])

    with pytest.raises(KeyboardInterrupt):
        with update_receipt.update_receipt_scope():
            update_receipt.begin_update_receipt()
            fleet_verify._verify_fleet_after_update(
                _restart(["hermes-gateway"]), _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    assert steps == ["receipt", "clear", "migrate", "notice"]

    steps.clear()
    incomplete = _GatewayRestartOutcome(
        incomplete=True, phase_errors=[], pre_restart_gateway_pids=[1234], restarted_services=["hermes-gateway"],
        failed_or_stale_units=["hermes-dashboard"], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set())
    monkeypatch.setattr(fleet_verify, "_record_owed_gateway_inventory", lambda *args, **kwargs: steps.append("owed"))
    with pytest.raises(KeyboardInterrupt):
        with update_receipt.update_receipt_scope():
            update_receipt.begin_update_receipt()
            fleet_verify._verify_fleet_after_update(
                incomplete, _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    assert steps == ["owed", "receipt", "notice"]


def test_a_hanging_scan_times_out_at_the_deadline_and_prints_nothing(monkeypatch, capsys):
    """The caller waits at most the deadline for a hanging backend scan and prints nothing on timeout."""
    import hermes_cli.gateway_migrate_backends as backends
    from hermes_cli.update_backend_notice import print_old_backend_notice

    started = threading.Event()
    release = threading.Event()

    def hanging_scan():
        started.set()
        release.wait(30)
        return []

    monkeypatch.setattr(backends, "collect_backends", hanging_scan)
    plan = UpdatePlan(runtimes=[RuntimeRecord("serve", "default", 7, "manual-serve")])
    deadline = 0.2
    t0 = time.monotonic()
    try:
        print_old_backend_notice(plan, None, timeout=deadline)
        elapsed = time.monotonic() - t0
        assert started.is_set()
        assert not release.is_set()
        assert deadline * 0.8 <= elapsed < 2.0
        assert capsys.readouterr().out == ""
    finally:
        release.set()
