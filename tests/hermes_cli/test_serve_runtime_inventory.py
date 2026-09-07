"""Serve-kind runtime inventory + stop/relaunch rung (#63206, campaign #91277).

A network-bound `hermes serve --host <ip>` powering a remote Desktop used to
be invisible to the update pipeline: not in the inventory, a dead-end at the
venv-holder guard, and never relaunched after `hermes update` killed it. The
fix threads the spawn ledger's structured launch identity (host/port/profile,
registered at serve startup) through inventory → guard rung → relaunch.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch  # noqa: F401 - kept for parity with siblings

import hermes_cli.update_cmd as update_cmd
import hermes_cli.update_inventory as update_inventory
from hermes_cli import main as cli_main


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


def test_describe_restart_mechanism_respawn_argv():
    text = update_inventory.describe_restart_mechanism("respawn-argv", "default")
    assert "relaunch" in text


# ---------------------------------------------------------------------------
# update_cmd: guard rung helpers
# ---------------------------------------------------------------------------


def test_ledger_manual_serve_holders_filters_correctly(monkeypatch):
    manual = _ledger_entry(pid=100)
    desktop_owned = _ledger_entry(pid=200, spawner_pid=999, spawner_create=1.0)
    gateway = _ledger_entry(pid=300, purpose="gateway")
    not_a_holder = _ledger_entry(pid=400)

    fake_pi = SimpleNamespace(
        ledger_entries=lambda **k: [manual, desktop_owned, gateway, not_a_holder],
        spawner_is_dead=lambda e: False if e["pid"] == 200 else None,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    holders = [(100, "python.exe", "..."), (200, "python.exe", "..."), (300, "python.exe", "...")]

    result = update_cmd._ledger_manual_serve_holders(holders)
    pids = [e["pid"] for e in result]
    assert pids == [100], (
        "only the manual serve holder qualifies: desktop-owned keeps the "
        "refusal, gateways belong to the pause machinery, non-holders skipped"
    )


def test_serve_relaunch_commands_built_from_structured_identity(monkeypatch):
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: None)
    entries = [
        _ledger_entry(),                                  # default profile
        _ledger_entry(pid=5000, profile="work", port=9200, host=""),
        _ledger_entry(pid=6000, port=None),               # no port → skipped
        _ledger_entry(pid=7000, purpose="dashboard", host="0.0.0.0", port=9300),
    ]
    cmds = update_cmd._serve_relaunch_commands(entries)
    assert ["hermes", "serve", "--host", "100.94.65.93", "--port", "9119"] in cmds
    assert ["hermes", "--profile", "work", "serve", "--port", "9200"] in cmds
    assert ["hermes", "dashboard", "--host", "0.0.0.0", "--port", "9300"] in cmds
    assert len(cmds) == 3  # the port-less entry is skipped


def test_relaunch_stopped_serves_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_main, "_respawn_dashboard_processes", lambda cmds: calls.append(cmds) or []
    )
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: None)
    token = {"pending": True, "entries": [_ledger_entry()]}

    update_cmd._relaunch_stopped_serves(token)
    update_cmd._relaunch_stopped_serves(token)  # atexit double-fire

    assert len(calls) == 1, "relaunch must fire exactly once"
    assert token["pending"] is False


def test_relaunch_stopped_serves_untriggered_token_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_main, "_respawn_dashboard_processes", lambda cmds: calls.append(cmds) or []
    )
    update_cmd._relaunch_stopped_serves({"pending": False, "entries": [_ledger_entry()]})
    assert calls == []


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
