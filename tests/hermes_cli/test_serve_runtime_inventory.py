"""Serve-kind runtime inventory (#63206, campaign #91277).

A network-bound `hermes serve --host <ip>` powering a remote Desktop used to
be invisible to the update pipeline. The spawn ledger's structured launch
identity (host/port/profile, registered at serve startup) now feeds the
update inventory and the dashboard process scan.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_cli.update_inventory as update_inventory
import hermes_cli.main_dashboard as main_dashboard

@pytest.fixture(autouse=True)
def _no_host_cgroup(monkeypatch):
    """The ledger PIDs below are fictional: never read this host's ``/proc`` or ask its systemd who
    owns them. The systemd tests install their own fake cgroup via ``_systemd_host``."""
    monkeypatch.setattr(main_dashboard, "_pid_unified_cgroup_entries", lambda pid: iter(()))

def _ledger_entry(**over):
    entry = {
        "pid": 4321,
        "create_time": 111.0,
        "purpose": "serve",
        "install": "inst",
        "spawner_pid": None,
        "spawner_create": None,
        "registered_at": 222.0,
        "argv": "hermes serve --host 100.94.65.93 --port 9119",
        "host": "100.94.65.93",
        "port": 9119,
        "profile": "",
    }
    entry.update(over)
    return entry

# ---------------------------------------------------------------------------
# process_identity: structured detail round-trip
# ---------------------------------------------------------------------------

def test_register_self_records_structured_detail(tmp_path, monkeypatch):
    from hermes_cli import process_identity as pi

    monkeypatch.setattr(pi, "_ledger_path", lambda: tmp_path / "ledger.json")
    monkeypatch.setattr(pi, "install_id", lambda *a, **k: "inst")
    assert pi.register_self(
        "serve", detail={"host": "100.94.65.93", "port": 9119, "profile": "work"}
    )
    entries = [
        e
        for e in pi._read_ledger(tmp_path / "ledger.json")
        if e["purpose"] == "serve"
    ]
    assert entries, "serve entry must be written"
    e = entries[-1]
    assert e["host"] == "100.94.65.93"
    assert e["port"] == 9119
    assert e["profile"] == "work"


def test_register_self_records_isolated_marker(tmp_path, monkeypatch):
    from hermes_cli import process_identity as pi

    monkeypatch.setattr(pi, "_ledger_path", lambda: tmp_path / "ledger.json")
    monkeypatch.setattr(pi, "install_id", lambda *a, **k: "inst")
    assert pi.register_self("serve", detail={"host": "127.0.0.1", "port": 9119, "isolated": True})
    assert pi._read_ledger(tmp_path / "ledger.json")[-1]["isolated"] is True


def test_register_self_without_detail_stays_backward_compatible(
    tmp_path, monkeypatch
):
    from hermes_cli import process_identity as pi

    monkeypatch.setattr(pi, "_ledger_path", lambda: tmp_path / "ledger.json")
    monkeypatch.setattr(pi, "install_id", lambda *a, **k: "inst")
    assert pi.register_self("gateway")
    e = pi._read_ledger(tmp_path / "ledger.json")[-1]
    assert e["host"] == "" and e["port"] is None and e["profile"] == ""

# ---------------------------------------------------------------------------
# update_inventory: serve collector
# ---------------------------------------------------------------------------

def test_inventory_includes_manual_serve_from_ledger(monkeypatch):
    entry = _ledger_entry()
    fake_pi = SimpleNamespace(
        ledger_entries=lambda **k: [entry],
        spawner_is_dead=lambda e: None,  # no spawner recorded → manual
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = update_inventory.collect_runtime_inventory()
    serves = [r for r in plan.runtimes if r.kind == "serve"]
    assert serves, "manual serve must appear in the inventory"
    row = serves[0]
    assert row.pid == 4321
    assert row.supervisor == "manual-serve"
    assert row.restart_via == "respawn-argv"
    assert row.detail["host"] == "100.94.65.93"
    assert row.detail["port"] == 9119

def test_inventory_classifies_desktop_owned_serve(monkeypatch):
    entry = _ledger_entry(spawner_pid=999, spawner_create=1.0)
    fake_pi = SimpleNamespace(
        ledger_entries=lambda **k: [entry],
        spawner_is_dead=lambda e: False,  # Electron parent alive
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = update_inventory.collect_runtime_inventory()
    serves = [r for r in plan.runtimes if r.kind == "serve"]
    assert serves and serves[0].supervisor == "desktop"
    assert serves[0].restart_via == "desktop"


_SSH_ARGV = ("hermes serve --isolated --host 127.0.0.1 --port 0 "
             "--ssh-session-token-file /h/.hermes/desktop-ssh/a/b.token --ssh-owner-nonce 0123456789abcdef")


def test_inventory_classifies_remote_desktop_ssh_serve_as_its_clients(monkeypatch):
    """A serve another machine's Desktop spawned over SSH has no local spawner, so the spawner
    probe alone reads ``manual-serve``: the update then files a manual-restart reminder nobody on
    this host can discharge, and the recovery pass may try an argv respawn of a process whose
    token file and owner nonce only its remote client holds. The remote client owns its restart."""
    from hermes_cli.update_serve_obligations import defer_manual_serve

    entry = _ledger_entry(argv=_SSH_ARGV, host="127.0.0.1", port=57474, isolated=True)
    fake_pi = SimpleNamespace(ledger_entries=lambda **k: [entry], spawner_is_dead=lambda e: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = update_inventory.collect_runtime_inventory()
    row = next(r for r in plan.runtimes if r.kind == "serve")
    assert row.supervisor not in ("manual-serve", "desktop")
    assert row.restart_via != "respawn-argv"
    monkeypatch.undo()  # real process_identity: no durable manual-restart reminder may be filed
    assert defer_manual_serve(asdict(row)) is False


def test_hand_started_isolated_serve_stays_manual(monkeypatch):
    """``--isolated`` alone is an opt-out of the host singleton, not remote ownership: a user's own
    ``hermes serve --isolated`` keeps its manual-serve relaunch."""
    entry = _ledger_entry(argv="hermes serve --isolated --host 127.0.0.1 --port 9119", isolated=True)
    fake_pi = SimpleNamespace(ledger_entries=lambda **k: [entry], spawner_is_dead=lambda e: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    row = next(r for r in update_inventory.collect_runtime_inventory().runtimes if r.kind == "serve")
    assert row.supervisor == "manual-serve"


def test_stale_remote_desktop_ssh_serve_is_deferred_to_its_client_not_unaccounted(monkeypatch):
    """Still on pre-update code after the update, the SSH serve is its remote client's to recycle:
    reported as deferred (not an unaccounted failure that fails the update), and never owed by the
    abort-recovery pass."""
    from hermes_cli.update_abort_recovery import _owed_stale_serve_rows

    entry = _ledger_entry(argv=_SSH_ARGV, host="127.0.0.1", port=57474, isolated=True)
    fake_pi = SimpleNamespace(ledger_entries=lambda **k: [entry], spawner_is_dead=lambda e: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = update_inventory.collect_runtime_inventory()
    outcomes = update_inventory.match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[], stale_serve_pids={entry["pid"]},
    )
    serve = next(o for o in outcomes if o["kind"] == "serve")
    assert serve["outcome"] == "deferred"
    assert update_inventory.report_unaccounted_runtimes(outcomes) is False
    row = next(r for r in plan.runtimes if r.kind == "serve")
    assert _owed_stale_serve_rows([{"supervisor": row.supervisor}]) == []


# ---------------------------------------------------------------------------
# dashboard_procs: ledger augmentation of the scan (#81564 half)
# ---------------------------------------------------------------------------

def test_scan_dashboard_processes_includes_ledger_only_serves(monkeypatch):
    """A profiled serve (`hermes --profile p serve ...`) matches no scan
    pattern; the ledger row must still surface it."""
    import hermes_cli.dashboard_procs as dp

    profiled = _ledger_entry(
        pid=8123,
        argv="hermes --profile work serve --host 100.94.65.93 --port 9119",
        profile="work",
    )
    fake_pi = SimpleNamespace(ledger_entries=lambda **k: [profiled])
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)

    # Force the ps/wmic scan itself to find nothing.
    fake_run = SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(
        dp.subprocess, "run", lambda *a, **k: fake_run
    )
    result = dp._scan_dashboard_processes()
    assert (8123, profiled["argv"]) in result

def test_scan_dashboard_processes_ledger_respects_exclusions(monkeypatch):
    import hermes_cli.dashboard_procs as dp

    entry = _ledger_entry(pid=8124)
    fake_pi = SimpleNamespace(ledger_entries=lambda **k: [entry])
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    fake_run = SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(dp.subprocess, "run", lambda *a, **k: fake_run)

    assert dp._scan_dashboard_processes(exclude_pids={8124}) == []

def test_inventory_records_the_serve_process_incarnation(monkeypatch):
    """The plan carries ``(pid, create_time)``, not just the PID (#92145 review).

    The post-abort survivor probe compares a planned serve against the live
    spawn ledger. With only the number to compare, a NEW serve that reused the
    old PID reads as the pre-update process that never restarted, and recovery
    stays incomplete forever.
    """
    entry = _ledger_entry(create_time=1712345678.5)
    fake_pi = SimpleNamespace(
        ledger_entries=lambda **k: [entry],
        spawner_is_dead=lambda e: None,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = update_inventory.collect_runtime_inventory()
    serves = [r for r in plan.runtimes if r.kind == "serve"]
    assert serves and serves[0].detail["create_time"] == 1712345678.5

# ---------------------------------------------------------------------------
# update_inventory: launchd-owned serve/dashboard classification (#116503)
# ---------------------------------------------------------------------------

def test_inventory_classifies_launchd_job_owned_serve(monkeypatch):
    """A KeepAlive LaunchAgent backend's recorded spawner (the bootstrap shell) is long dead,
    so the spawner probe alone reads manual-serve — and the update plan then restarts it as a
    detached argv respawn that fights the job's own KeepAlive respawn. A loaded job whose
    ProgramArguments match the ledger argv must classify the row launchd (kickstart restart)."""
    entry = _ledger_entry(spawner_pid=999, spawner_create=1.0)
    fake_pi = SimpleNamespace(
        ledger_entries=lambda **k: [entry],
        spawner_is_dead=lambda e: True,  # bootstrap shell provably gone
    )
    jobs = [("gui/501", "ai.hermes.dashboard",
             ["hermes", "serve", "--host", "100.94.65.93", "--port", "9119"], None)]
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    with patch.object(main_dashboard, "_loaded_launchd_backend_jobs", return_value=jobs), \
         patch("hermes_cli.dashboard_procs._process_ancestors", return_value=[]):
        plan = update_inventory.collect_runtime_inventory()
    serves = [r for r in plan.runtimes if r.kind == "serve"]
    assert serves, "launchd-owned serve must appear in the inventory"
    row = serves[0]
    assert row.supervisor == "launchd"
    assert row.restart_via == "launchd"
    assert row.detail["launchd_domain"] == "gui/501"
    assert row.detail["launchd_label"] == "ai.hermes.dashboard"

# ---------------------------------------------------------------------------
# update_inventory: systemd-unit-owned serve/dashboard classification
# ---------------------------------------------------------------------------

_USER_UNIT_CGROUP = "/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-dashboard-web.service"

def _systemd_host(monkeypatch, *, cgroup: str, main_pid: int):
    """Fake ``/proc/<pid>/cgroup`` and ``systemctl show -p MainPID`` for the ledger PID."""
    calls: list[list[str]] = []

    def fake_probe(cmd, *, timeout):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout=f"{main_pid}\n", stderr="")

    monkeypatch.setattr(main_dashboard, "_pid_unified_cgroup_entries", lambda pid: iter([cgroup]))
    monkeypatch.setattr(main_dashboard, "_run_probe", fake_probe)
    return calls

def _ledger_serve_rows(monkeypatch, entry):
    fake_pi = SimpleNamespace(
        ledger_entries=lambda **k: [entry],
        spawner_is_dead=lambda e: None,
        _pid_alive_matches=lambda pid, created: True,  # the ledger PID is the live incarnation
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = update_inventory.UpdatePlan()
    update_inventory._collect_ledger_runtimes(plan, set())
    return plan.runtimes

def test_inventory_classifies_systemd_unit_owned_dashboard(monkeypatch):
    """A dashboard run as its own systemd unit records no spawner, so the spawner probe alone
    reads it as manual-serve: the plan proposes a respawn-argv restart and the CLI files a
    "manual restart still pending" reminder that tells the operator to relaunch a process
    systemd owns. The unit whose MainPID IS the ledger PID is the supervisor."""
    calls = _systemd_host(monkeypatch, cgroup=_USER_UNIT_CGROUP, main_pid=4321)
    rows = _ledger_serve_rows(monkeypatch, _ledger_entry(purpose="dashboard"))
    assert len(rows) == 1
    row = rows[0]
    assert (row.kind, row.supervisor, row.restart_via) == ("dashboard", "systemd", "systemd")
    assert row.detail["systemd_unit"] == "hermes-dashboard-web.service"
    assert row.detail["systemd_scope"] == "user"
    assert calls == [["systemctl", "--user", "show", "hermes-dashboard-web.service", "--property=MainPID", "--value"]]

def test_inventory_asks_the_system_manager_for_a_system_unit(monkeypatch):
    calls = _systemd_host(monkeypatch, cgroup="/system.slice/hermes-serve.service", main_pid=4321)
    [row] = _ledger_serve_rows(monkeypatch, _ledger_entry())
    assert (row.supervisor, row.detail["systemd_scope"]) == ("systemd", "system")
    assert calls == [["systemctl", "show", "hermes-serve.service", "--property=MainPID", "--value"]]

def test_serve_inside_another_units_cgroup_stays_manual(monkeypatch):
    """A serve started from a shell that lives in some unit's cgroup (the gateway's own agent
    terminal, a desktop session service) is NOT supervised by that unit: restarting the unit
    would restart the wrong process. Only MainPID ownership classifies."""
    gateway_cgroup = "/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service"
    _systemd_host(monkeypatch, cgroup=gateway_cgroup, main_pid=777)
    [row] = _ledger_serve_rows(monkeypatch, _ledger_entry())
    assert (row.supervisor, row.restart_via) == ("manual-serve", "respawn-argv")
    assert "systemd_unit" not in row.detail

def test_systemd_owned_dashboard_files_no_manual_restart_reminder(tmp_path, monkeypatch):
    """The user-visible symptom: every CLI start printed "manual restart still pending" for a
    unit-run dashboard. A systemd row is its supervisor's to restart, so it is outside the
    gateway matrix's evidence without a durable manual reminder being written."""
    from dataclasses import asdict

    from hermes_cli.update_cmd_fleet_gatewayless import runtime_outside_gateway_evidence

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _systemd_host(monkeypatch, cgroup=_USER_UNIT_CGROUP, main_pid=4321)
    [row] = _ledger_serve_rows(monkeypatch, _ledger_entry(purpose="dashboard"))
    assert runtime_outside_gateway_evidence(asdict(row))
    assert not (home / "serve_restart_pending").exists()

def test_system_unit_outside_system_slice_is_still_classified(monkeypatch):
    """``Slice=`` is an ordinary unit option: a system unit under a custom slice, or with no slice
    at all, is still the system manager's. Only a ``user@<uid>.service`` segment means the user
    manager, whatever slice the user unit sits in."""
    for cgroup, scope, manager in (
        ("/hermes.slice/hermes-serve.service", "system", []),
        ("/hermes-serve.service", "system", []),
        ("/user.slice/user-1000.slice/user@1000.service/hermes.slice/hermes-serve.service", "user", ["--user"]),
    ):
        calls = _systemd_host(monkeypatch, cgroup=cgroup, main_pid=4321)
        [row] = _ledger_serve_rows(monkeypatch, _ledger_entry())
        assert (row.supervisor, row.detail["systemd_scope"]) == ("systemd", scope), cgroup
        assert calls == [["systemctl", *manager, "show", "hermes-serve.service", "--property=MainPID", "--value"]]

def test_unaccounted_systemd_row_names_its_own_unit(monkeypatch, capsys):
    """The operator hint used to hardcode ``hermes-serve.service``; a unit-run dashboard named
    anything else was told to restart a unit that does not exist."""
    monkeypatch.setattr(update_inventory.sys, "platform", "linux")
    outcomes = [{"kind": "dashboard", "profile": "default", "pid": 4321, "mechanism": "systemd",
                 "outcome": "unaccounted", "systemd_scope": "user", "systemd_unit": "hermes-dashboard-web.service"}]
    assert update_inventory.report_unaccounted_runtimes(outcomes) is True
    out = capsys.readouterr().out
    assert "systemctl --user restart hermes-dashboard-web.service" in out
    assert "hermes-serve.service" not in out

def test_stale_systemd_row_warning_names_its_own_unit(monkeypatch, capsys):
    from hermes_cli import update_abort_recovery

    monkeypatch.setattr(update_abort_recovery.sys, "platform", "linux")
    update_abort_recovery._warn_stale_serve_runtimes([
        {"pid": 4321, "kind": "serve", "profile": "default", "supervisor": "systemd",
         "systemd_scope": "system", "systemd_unit": "hermes-serve-lan.service"}])
    out = capsys.readouterr().out
    assert "`systemctl restart hermes-serve-lan.service`" in out
    assert "hermes-serve.service" not in out

def test_systemd_unit_reaches_outcomes_and_survivor_rows(monkeypatch):
    """The unit is recorded at inventory time; both hint paths read it off their own row shape."""
    from hermes_cli import update_abort_recovery

    _systemd_host(monkeypatch, cgroup=_USER_UNIT_CGROUP, main_pid=4321)
    entry = _ledger_entry(purpose="dashboard")
    plan = update_inventory.UpdatePlan()
    plan.runtimes = _ledger_serve_rows(monkeypatch, entry)
    [survivor] = update_abort_recovery._surviving_pre_update_serve_runtimes(plan)
    assert (survivor["systemd_scope"], survivor["systemd_unit"]) == ("user", "hermes-dashboard-web.service")
    [outcome] = update_inventory.match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[], stale_serve_pids={4321},
    )
    assert outcome["outcome"] == "unaccounted"
    assert outcome["systemd_unit"] == "hermes-dashboard-web.service"
