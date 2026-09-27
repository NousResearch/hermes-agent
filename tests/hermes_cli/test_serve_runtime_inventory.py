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

import hermes_cli.update_inventory as update_inventory
import hermes_cli.main_dashboard as main_dashboard

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
