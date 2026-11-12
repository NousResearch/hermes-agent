"""``hermes gateway migrate --backends``: the serve/dashboard service report (changes no service).

A fake host (process table, cgroups, systemctl, launchctl) drives the real detection code. Every
test also proves the report is read-only: no signal, no file write/rename/delete and no mutating
systemctl/launchctl verb is ever issued.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import plistlib
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_agent.hermes_cli import gateway_migrate as gm
from hermes_agent.hermes_cli import gateway_migrate_backends as gmb
from hermes_agent.hermes_cli import main_dashboard
from tests.waiting import wait_until

_READ_ONLY_SYSTEMCTL = {"show", "list-unit-files"}
_READ_ONLY_LAUNCHCTL = {"print"}


def _argv(*tail):
    return ["/opt/venv/bin/python", "-m", "hermes_cli.main", *tail]


class FakeHost:
    """Process table, cgroups and systemd state behind ``subprocess.run``; records every call."""

    def __init__(self):
        self.procs: dict[int, tuple[int, list[str]]] = {}  # pid -> (ppid, argv)
        self.cgroups: dict[int, str] = {}
        self.units: dict[tuple[str, str], dict[str, str]] = {}  # (scope, unit) -> show properties
        self.unit_files: dict[str, list[tuple[str, str]]] = {"user": [], "system": []}
        self.calls: list[list[str]] = []
        self.home = None

    def proc(self, pid, argv, *, ppid=1, cgroup=None):
        self.procs[pid] = (ppid, argv)
        if cgroup:
            self.cgroups[pid] = cgroup

    def unit(self, scope, name, *, main_pid=0, path, state="enabled", env=""):
        self.units[(scope, name)] = {"MainPID": str(main_pid), "FragmentPath": path, "UnitFileState": state,
                                     "ActiveState": "active" if main_pid else "inactive", "Environment": env}
        self.unit_files[scope].append((name, state))

    def run(self, cmd, *args, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        out, rc = "", 0
        if cmd[0] == "ps":
            if cmd[1:3] == ["-A", "-o"]:
                out = "".join(f"{pid} {shlex.join(argv)}\n" for pid, (_pp, argv) in self.procs.items())
            elif "ppid=" in cmd:
                pid = int(cmd[-1])
                out, rc = (f"{self.procs[pid][0]}\n", 0) if pid in self.procs else ("", 1)
            elif "command=" in cmd:
                pid = int(cmd[2])
                out, rc = (shlex.join(self.procs[pid][1]) + "\n", 0) if pid in self.procs else ("", 1)
        elif cmd[0] == "systemctl":
            scope = "user" if "--user" in cmd else "system"
            rest = [c for c in cmd[1:] if c != "--user"]
            if rest[0] == "list-unit-files":
                out = "".join(f"{n} {s} enabled\n" for n, s in self.unit_files[scope])
            elif rest[0] == "show":
                props = self.units.get((scope, rest[1]), {})
                wanted = [c.split("=", 1)[1] for c in rest if c.startswith("--property=")]
                if "--value" in rest:
                    out = "".join(f"{props.get(w, '')}\n" for w in wanted)
                else:
                    out = "".join(f"{w}={props.get(w, '')}\n" for w in wanted)
            else:
                rc = 1
        elif cmd[0] == "launchctl":
            rc = 1
        return subprocess.CompletedProcess(cmd, rc, out, "")

    def assert_read_only(self):
        for cmd in self.calls:
            assert cmd[0] in ("ps", "systemctl", "launchctl"), cmd
            verbs = [c for c in cmd[1:] if not c.startswith("-")]
            if cmd[0] == "systemctl":
                assert verbs and verbs[0] in _READ_ONLY_SYSTEMCTL, cmd
            if cmd[0] == "launchctl":
                assert verbs and verbs[0] in _READ_ONLY_LAUNCHCTL, cmd


@pytest.fixture
def host(monkeypatch, tmp_path):
    fake = FakeHost()
    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(main_dashboard, "_pid_unified_cgroup_entries",
                        lambda pid: iter([fake.cgroups[pid]] if pid in fake.cgroups else []))
    monkeypatch.setattr(main_dashboard, "_dashboard_cmdline_for_pid",
                        lambda pid: list(fake.procs[pid][1]) if pid in fake.procs else None)
    monkeypatch.setattr("hermes_agent.hermes_cli._startup_fast.is_container_startup_environment", lambda: False)
    monkeypatch.setattr("hermes_agent.hermes_cli.dashboard_procs._pid_environ", lambda pid: None)
    monkeypatch.setattr(main_dashboard, "_loaded_launchd_backend_jobs", list)
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake.home = home
    yield fake
    fake.assert_read_only()


@pytest.fixture
def no_writes(monkeypatch):
    """Fail the test on any signal, file write, rename or delete."""
    def boom(*a, **k):
        raise AssertionError(f"mutation attempted: {a!r}")

    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(m in mode for m in "wax+"):
            boom(file, mode)
        return real_open(file, mode, *a, **k)

    for name in ("kill", "killpg", "replace", "rename", "unlink", "remove", "rmdir", "mkdir", "makedirs"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, boom)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(subprocess, "Popen", boom)


def _by_supervisor(items):
    return {i.supervisor: i for i in items}


def test_systemd_units_by_mainpid_with_scope_path_and_nix_detection(host, no_writes):
    host.proc(4101, _argv("dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"),
              cgroup="/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-web.service")
    host.unit("user", "hermes-web.service", main_pid=4101, path="/home/u/.config/systemd/user/hermes-web.service")
    host.proc(4102, _argv("serve", "--host", "0.0.0.0", "--port", "9120"),
              cgroup="/system.slice/hermes-backend.service")
    host.unit("system", "hermes-backend.service", main_pid=4102,
              path="/nix/store/abc-unit-hermes-backend.service/hermes-backend.service")

    items = _by_supervisor(gmb.collect_backends())

    web = items["systemd"]
    assert (web.unit, web.scope, web.pid, web.running) == ("hermes-web.service", "user", 4101, True)
    assert web.unit_path == "/home/u/.config/systemd/user/hermes-web.service"
    assert any("written by hand" in n for n in web.notes)
    nix = items["nix-systemd"]
    assert (nix.unit, nix.scope) == ("hermes-backend.service", "system")
    assert any("generated by Nix and read-only" in n for n in nix.notes)
    # `hermes update`'s systemd pass does not match either name; both say so.
    assert any("hermes-backend.service" in n for n in nix.notes)
    assert any("hermes-web.service" in n for n in web.notes)


def test_cgroup_without_mainpid_is_not_claimed_by_the_unit(host, no_writes):
    """A dashboard started from a shell inside some unit's cgroup is not that unit's backend."""
    host.proc(4201, _argv("dashboard", "--port", "9119"), ppid=4200,
              cgroup="/user.slice/user-1000.slice/user@1000.service/app.slice/tmux.service")
    host.proc(4200, ["bash"], ppid=900)
    host.unit("user", "tmux.service", main_pid=4200, path="/usr/lib/systemd/user/tmux.service")

    (item,) = [i for i in gmb.collect_backends() if i.pid == 4201]
    assert item.supervisor == "manual" and item.unit is None


def test_installed_enabled_unit_that_is_not_running_is_reported_once(host, no_writes):
    host.unit("user", "hermes-dashboard.service", path="/home/u/.config/systemd/user/hermes-dashboard.service",
              env="HERMES_HOME=/srv/hermes PATH=/usr/bin")
    host.unit("system", "hermes-serve-old.service", path="/etc/systemd/system/hermes-serve-old.service",
              state="disabled")

    items = gmb.collect_backends()

    assert [(i.unit, i.running, i.state, i.hermes_home) for i in items] == [
        ("hermes-dashboard.service", False, "enabled", "/srv/hermes")]


@pytest.mark.platforms("linux")  # "detached" needs a non-launchd init as ppid 1
def test_s6_docker_desktop_and_detached_shapes(host, no_writes, monkeypatch):
    host.proc(4300, ["s6-supervise", "/run/service/dashboard"], ppid=1)
    host.proc(4301, _argv("dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"), ppid=4300)
    host.proc(4310, _argv("dashboard", "--host", "127.0.0.1", "--no-open"), ppid=4309,
              cgroup="/system.slice/docker-0123abcd.scope")
    host.proc(4309, ["tini", "--"], ppid=4000)
    host.proc(4320, _argv("serve", "--isolated", "--ssh-session-token-file", "/tmp/t", "--port", "0"))
    host.proc(4330, _argv("serve", "--host", "127.0.0.1", "--port", "0"), ppid=4329)
    host.proc(4329, ["/Applications/Hermes"], ppid=1)
    host.proc(4340, _argv("serve", "--host", "100.64.0.2", "--port", "9119"), ppid=1)
    host.proc(4350, ["vim", "notes about hermes serve"], ppid=1)  # not a backend

    items = {i.pid: i for i in gmb.collect_backends()}

    assert set(items) == {4301, 4310, 4320, 4330, 4340}
    assert (items[4301].supervisor, items[4301].unit, items[4301].unit_path) == (
        "s6", "dashboard", "/run/service/dashboard")
    assert items[4310].supervisor == "container"
    assert items[4320].supervisor == "desktop-ssh"
    assert items[4330].supervisor == "desktop-local"
    assert items[4340].supervisor == "detached"


@pytest.fixture
def launch_agents(tmp_path):
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    jobs = {"com.me.dash": _argv("dashboard", "--port", "9119"),
            "org.nix-community.home.hermes-backend": _argv("serve", "--port", "9120")}
    for label, argv in jobs.items():
        with open(agents / f"{label}.plist", "wb") as f:
            plistlib.dump({"Label": label, "ProgramArguments": argv}, f)
    return agents, jobs


def test_launchd_job_with_plist_path_and_nix_store_plist(host, launch_agents, no_writes, monkeypatch):
    agents, jobs = launch_agents
    nix_plist = str(agents / "org.nix-community.home.hermes-backend.plist")
    # Home Manager links the plist into the read-only store; stand in for that link target.
    monkeypatch.setattr(gmb, "_nix_managed", lambda p: p == nix_plist)
    monkeypatch.setattr(main_dashboard, "_launchd_plist_dirs", lambda: [("agent", agents)])
    monkeypatch.setattr(main_dashboard, "_loaded_launchd_backend_jobs", lambda: [
        ("gui/501", "com.me.dash", jobs["com.me.dash"], 4401),
        ("gui/501", "org.nix-community.home.hermes-backend", jobs["org.nix-community.home.hermes-backend"], None)])
    host.proc(4401, jobs["com.me.dash"], ppid=1)

    items = {i.unit: i for i in gmb.collect_backends()}

    dash = items["com.me.dash"]
    assert (dash.supervisor, dash.pid, dash.scope) == ("launchd", 4401, "gui/501")
    assert dash.unit_path == str(agents / "com.me.dash.plist")
    assert any("written by hand" in n for n in dash.notes)
    nix = items["org.nix-community.home.hermes-backend"]
    assert (nix.supervisor, nix.running) == ("nix-launchd", False)


@pytest.fixture
def nix_launcher_agent(tmp_path):
    """A Home Manager ``backend.waitFor`` agent: ProgramArguments is the launcher script alone."""
    label = "org.nix-community.home.hermes-backend"
    with open(tmp_path / f"{label}.plist", "wb") as f:
        plistlib.dump({"Label": label, "ProgramArguments": ["/nix/store/abc-hermes-backend-launch"],
                       "EnvironmentVariables": {"HERMES_HOME": "/Users/u/.hermes"}}, f)
    return tmp_path, label


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("live_pid", [4701, None])
def test_nix_wait_for_launcher_job_is_reported_with_its_plist_home(host, nix_launcher_agent, no_writes,
                                                                     monkeypatch, live_pid):
    agents, label = nix_launcher_agent
    monkeypatch.setattr(main_dashboard, "_launchd_plist_dirs", lambda: [("agent", agents)])
    monkeypatch.setattr(gmb, "_nix_managed", lambda p: p == str(agents / f"{label}.plist"))
    monkeypatch.setattr("hermes_agent.hermes_cli.gateway._launchd_print_service_pid",
                        lambda domain, lbl: (lbl == label, live_pid))
    if live_pid:  # the launcher `exec`s hermes, so launchd's live PID is the backend itself
        host.proc(live_pid, ["/nix/store/x-hermes/bin/hermes", "serve", "--host", "mac.local"], ppid=1)

    (item,) = gmb.collect_backends()

    assert (item.supervisor, item.unit, item.pid, item.hermes_home) == (
        "nix-launchd", label, live_pid, "/Users/u/.hermes")


@pytest.mark.platforms("not macos")
def test_launchd_plists_are_ignored_off_macos(host, nix_launcher_agent, no_writes, monkeypatch):
    """A leftover LaunchAgents plist on Linux or Windows is never scanned and never queried."""
    agents, _label = nix_launcher_agent
    monkeypatch.setattr(main_dashboard, "_launchd_plist_dirs", lambda: [("agent", agents)])
    monkeypatch.setattr("hermes_agent.hermes_cli.gateway._launchd_print_service_pid",
                        lambda domain, lbl: pytest.fail("launchctl queried off macOS"))

    assert gmb.collect_backends() == []
    assert not [c for c in host.calls if c[0] == "launchctl"]


@pytest.mark.platforms("macos")
def test_parent_launchd_is_not_reported_as_detached_on_macos(host, tmp_path, no_writes, monkeypatch):
    """launchd is every orphan's parent on macOS, and a plist that wraps hermes (``sh -c``) is not matched."""
    monkeypatch.setattr(main_dashboard, "_launchd_plist_dirs", lambda: [("agent", tmp_path)])
    host.proc(4801, _argv("dashboard", "--port", "9119"), ppid=1)

    (item,) = gmb.collect_backends()

    assert item.supervisor == "unknown"


def test_home_ownership_is_reported_against_this_hermes_home(host, no_writes, monkeypatch):
    host.proc(4501, _argv("dashboard", "--port", "9119"), ppid=1)
    host.proc(4502, _argv("serve", "--port", "9120"), ppid=1)
    monkeypatch.setattr("hermes_agent.hermes_cli.dashboard_procs._hermes_home_for_pid",
                        lambda pid: {4501: str(host.home), 4502: "/other/home"}[pid])

    items = {i.pid: i for i in gmb.collect_backends()}

    assert (items[4501].this_home, items[4502].this_home) == (True, False)


@pytest.mark.platforms("linux")  # "detached" needs a non-launchd init as ppid 1
def test_cli_json_mode_exits_zero_and_never_applies(host, no_writes, capsys):
    host.proc(4601, _argv("dashboard", "--port", "9119"), ppid=1)

    gm.cmd_migrate(SimpleNamespace(backends=True, json=True, dry_run=True, yes=False, multiplex=False))

    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is False
    assert [(b["pid"], b["supervisor"]) for b in report["backends"]] == [(4601, "detached")]
    (backend,) = report["backends"]
    assert not {"would", "advice"} & backend.keys()
    assert backend["argv"] == _argv("dashboard", "--port", "9119")
    assert "migration is not available yet" in backend["not_applied"]

    gm.cmd_migrate(SimpleNamespace(backends=True, json=False, dry_run=True, yes=False, multiplex=False))
    text = capsys.readouterr().out
    assert "detached: PID 4601" in text and "runs:" in text
    assert "No supervisor owns it" in text
    assert "would:" not in text and "advice:" not in text


def test_cli_text_mode_with_nothing_found(host, no_writes, capsys):
    gm.cmd_migrate(SimpleNamespace(backends=True, json=False, dry_run=False, yes=False, multiplex=False))
    out = capsys.readouterr().out
    assert "nothing was changed" in out and "No running or enabled" in out


@pytest.mark.parametrize("flags", [{"yes": True}, {"multiplex": True}])
def test_backends_refuses_apply_flags(host, flags):
    args = SimpleNamespace(backends=True, json=False, dry_run=False, **{"yes": False, "multiplex": False, **flags})
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(args)
    assert exc.value.code == 2
    assert host.calls == []


def test_json_without_backends_is_an_argument_error(host):
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(backends=False, json=True, dry_run=True, yes=False, multiplex=True))
    assert exc.value.code == 2


def test_parser_accepts_backends_dry_run_json():
    import argparse

    from hermes_agent.hermes_cli.subcommands.gateway import build_gateway_parser
    parser = argparse.ArgumentParser()
    noop = lambda args: None
    build_gateway_parser(parser.add_subparsers(dest="command"), cmd_gateway=noop, cmd_proxy=noop,
                         cmd_gateway_enroll=noop)
    args = parser.parse_args(["gateway", "migrate", "--backends", "--dry-run", "--json"])
    assert (args.backends, args.dry_run, args.json) == (True, True, True)


@pytest.mark.platforms("posix")
def test_scan_failure_still_exits_zero(host, no_writes, monkeypatch, capsys):
    def broken(cmd, *a, **k):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(subprocess, "run", broken)
    gm.cmd_migrate(SimpleNamespace(backends=True, json=True, dry_run=True, yes=False, multiplex=False))
    assert json.loads(capsys.readouterr().out)["backends"] == []


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Recording shims: every call is appended to $SHIM_LOG. ``ps`` reports one hand-started serve
# process so the report's classification path (ppid lookups, systemd/launchd queries) runs too.
_SHIM = """#!/bin/sh
printf '%s' "$(basename "$0")" >> "$SHIM_LOG"
for a in "$@"; do printf ' %s' "$a" >> "$SHIM_LOG"; done
printf '\\n' >> "$SHIM_LOG"
if [ "$(basename "$0")" = ps ]; then
  case "$*" in
    "-A -o pid=,command="*) echo "424242 /opt/venv/bin/python -m hermes_cli.main serve --port 9119" ;;
    *ppid=*) echo 1 ;;
  esac
fi
exit 0
"""
_SHIMMED = ("ps", "systemctl", "launchctl", "kill", "pkill", "killall", "docker")


def _wait_for_detached_prune(scratch, timeout=60.0):
    """Wait until the scratch-prune child startup detached (``hermes_constants_scratch_detach``) has
    exited. It outlives the command and may still create ``logs/`` under the home, so the trees are
    compared only once it is gone. It is spawned before the command exits, so it is either still in
    the process table (its argv names *scratch*) or already finished; a zombie counts as finished."""
    needles = {os.fsencode(str(scratch)), os.fsencode(os.path.realpath(scratch))}

    def _gone():
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as fh:
                    argv = fh.read().split(b"\0")
                with open(f"/proc/{entry}/stat", "rb") as fh:
                    state = fh.read().rsplit(b")", 1)[1].split()[0]
            except OSError:
                continue
            if needles.intersection(argv) and state != b"Z":
                return False
        return True

    wait_until(_gone, timeout=timeout, interval=0.05, what="detached scratch prune to exit")


def _fresh_hermes(tmp_path, name, *cli):
    """Run ``python -m hermes_agent.hermes_cli.main <cli>`` in a fresh process with its own homes and shimmed tools."""
    root = tmp_path / name
    shims = root / "bin"
    shims.mkdir(parents=True)
    for tool in _SHIMMED:
        (shims / tool).write_text(_SHIM)
        (shims / tool).chmod(0o755)
    log = root / "calls.log"
    log.touch()
    homes = {"hermes_home": root / "hermes-home", "home": root / "home"}
    homes["home"].mkdir()
    # No inherited temp vars (TMPDIR/TMP/TEMP, HERMES_SCRATCH_DIR): a user-set one makes startup
    # skip scratch setup, so both children run the default boot — scratch dir plus its detached prune.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_", "TERMUX_")) and k not in {"TMPDIR", "TMP", "TEMP"}}
    env.update({"HERMES_HOME": str(homes["hermes_home"]), "HOME": str(homes["home"]), "PYTHONPATH": _REPO,
                "PATH": f"{shims}{os.pathsep}{env.get('PATH', '/usr/bin:/bin')}", "SHIM_LOG": str(log),
                "PYTHONDONTWRITEBYTECODE": "1"})
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", *cli], cwd=_REPO, env=env,
                          capture_output=True, text=True, timeout=180, check=False)
    _wait_for_detached_prune(homes["hermes_home"] / "cache" / "scratch")
    trees = {key: sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else []
             for key, path in homes.items()}
    return proc, [line.split() for line in log.read_text().splitlines()], trees


@pytest.mark.platforms("linux")  # POSIX shell shims; the detached prune child is found via /proc
def test_real_entry_point_report_writes_only_what_plain_migrate_dry_run_writes(tmp_path):
    """Through the real entry point, ``--backends`` issues no mutating service/kill command and leaves
    the same files behind as the existing read-only ``gateway migrate --dry-run`` (normal startup)."""
    base, base_calls, base_trees = _fresh_hermes(tmp_path, "dry-run", "gateway", "migrate", "--dry-run")
    report, calls, trees = _fresh_hermes(tmp_path, "backends", "gateway", "migrate", "--backends",
                                         "--dry-run", "--json")

    assert base.returncode == 0, base.stderr
    assert report.returncode == 0, report.stderr
    payload = json.loads(report.stdout)
    assert payload["applied"] is False
    assert [b["pid"] for b in payload["backends"]] == [424242]

    assert ["ps", "-A"] in [c[:2] for c in calls]
    for call in calls:
        tool, rest = call[0], [a for a in call[1:] if a != "--user"]
        assert tool not in {"kill", "pkill", "killall", "docker"}, call
        if tool == "systemctl":
            assert rest and rest[0] in _READ_ONLY_SYSTEMCTL, call
        if tool == "launchctl":
            assert rest and rest[0] in _READ_ONLY_LAUNCHCTL, call
    assert not any(c[0] in {"kill", "pkill", "killall"} for c in base_calls)

    assert any(p.startswith("cache/scratch") for p in trees["hermes_home"]), trees  # default startup ran
    assert trees == base_trees
