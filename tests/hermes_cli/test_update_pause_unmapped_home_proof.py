"""A paused unmapped gateway keeps the home discovery proved for it (#132338 R2 / review thread 7).

Discovery owns a gateway whose environment is unreadable (elevated, another account) when this
install's spawn ledger holds an exact-birth witness of its home. The pause record must keep THAT
home: written as ``home: null`` it read as a pre-home record, whose argv-only readiness match let
another home's gateway with the same selectorless argv retire the paused runtime's debt.

Real processes, each under its own ``HERMES_HOME``, a real ledger entry, the real pause producer and
durable record, and the real readiness predicate. Seams: the OS denying A's environment
(``_pid_environ``), the Windows-only gateway enumeration and the stop itself (never signalled here).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil
import pytest

from gateway.status import get_process_start_time
from hermes_cli import dashboard_procs, gateway, process_identity, update_cmd_windows
from hermes_cli import update_pause_record as pause_record
from hermes_cli.update_cmd_windows import _pause_windows_gateways_for_update, _unmapped_ready_filter


@pytest.fixture
def homes_and_spawn(tmp_path):
    entry = tmp_path / "bin" / "hermes"  # a ``hermes`` entry token: the cmdline reads as a gateway
    entry.parent.mkdir()
    entry.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    argv = [sys.executable, str(entry), "gateway", "run"]
    root = os.environ["HERMES_HOME"]  # the sandboxed home this update owns, with two profiles
    home_a, home_b = (os.path.join(root, "profiles", name) for name in ("a", "b"))
    os.makedirs(home_a)
    os.makedirs(home_b)
    procs: list[subprocess.Popen] = []

    def spawn(home) -> subprocess.Popen:
        proc = subprocess.Popen(argv, env={**os.environ, "HERMES_HOME": str(home)})
        procs.append(proc)
        return proc

    yield home_a, home_b, spawn
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # stub children that sleep; the fixture reaps them
def test_a_ledger_proven_unmapped_gateway_is_recorded_with_its_home(homes_and_spawn, monkeypatch):
    home_a, home_b, spawn = homes_and_spawn
    a = spawn(home_a)
    time.sleep(0.05)
    ct = process_identity._process_create_time(a.pid)
    assert process_identity._append_entry(process_identity.LedgerEntry(
        a.pid, ct, "gateway", process_identity.install_id(), None, None, time.time(), "", hermes_home=home_a))
    real_environ = dashboard_procs._pid_environ
    monkeypatch.setattr(dashboard_procs, "_pid_environ", lambda pid: None if pid == a.pid else real_environ(pid))
    from hermes_cli import main
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [a.pid])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda strict=False: [])
    monkeypatch.setattr(gateway, "find_windows_gateway_services", lambda profile_processes=(): [])
    stopped = []
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", lambda *a, **k: stopped.append(a[0]) or {})
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *a: None)

    _pause_windows_gateways_for_update()
    assert stopped == [[a.pid]]
    entry = pause_record.read(pause_record.record_path())["token"]["unmapped"][0]
    assert entry["home"] == home_a, "the pause dropped the home discovery proved"

    a.kill()
    a.wait(timeout=10)
    time.sleep(0.05)  # a later clock tick: a replacement is born after the process it replaces
    sibling = spawn(home_b)
    assert _unmapped_ready_filter(entry, set())([sibling.pid]) == [], \
        "home B's gateway with A's argv retired A's restart debt"
    replacement = spawn(home_a)
    assert _unmapped_ready_filter(entry, set())([sibling.pid, replacement.pid]) == [replacement.pid]


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_a_recorded_but_empty_home_is_never_matched_only_a_pre_home_entry_is(homes_and_spawn):
    home_a, home_b, spawn = homes_and_spawn
    old = spawn(home_a)
    argv = list(psutil.Process(old.pid).cmdline())
    born = get_process_start_time(old.pid)
    old.kill()
    old.wait(timeout=10)
    time.sleep(0.05)
    sibling = spawn(home_b)
    assert _unmapped_ready_filter({"pid": old.pid, "argv": argv, "home": None, "ct": born}, set())([sibling.pid]) == [], \
        "a newly incomplete record took the legacy argv-only match"
    # A token from an updater that predates the home field has nothing else to match on.
    assert _unmapped_ready_filter({"pid": old.pid, "argv": argv}, set())([sibling.pid]) == [sibling.pid]
