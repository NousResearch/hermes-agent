"""Post-update relaunch verification polls every paused target at once (#132338 review, kshitij).

Each relaunched profile / unmapped gateway is retired only on its own stable readiness, and before
this every target ran its own confirmation window in turn: N targets cost N windows (5 profiles,
10 s) where the fleet-wide wait had cost one. One loop keeps the per-target crediting.

Real ``_verify_relaunched_gateways_alive`` and its poll; the seam is the PID-file read that says
which gateway each profile home runs (``get_running_pid``), so no process is spawned or signalled.
"""

from __future__ import annotations

import time

import pytest

from gateway import status
from hermes_cli import gateway_windows, update_cmd_windows
from hermes_cli.update_cmd_windows import _READY_CONFIRM_S, _verify_relaunched_gateways_alive


@pytest.fixture
def running(monkeypatch):
    """``{profile: callable -> pid or None}``: what each profile home's PID file names right now."""
    table: dict = {}
    monkeypatch.setattr(status, "get_running_pid",
                        lambda path, cleanup_stale=False: (table.get(path.parent.name) or (lambda: None))())
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)  # every old gateway is gone
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **k: None)
    return table


def test_many_ready_targets_cost_one_confirmation_window(running):
    profiles = {f"p{i}": 40000 + i for i in range(6)}
    for i, name in enumerate(profiles):
        running[name] = lambda pid=50000 + i: pid
    token = {"profiles": dict(profiles), "unmapped": []}
    started = time.monotonic()
    _verify_relaunched_gateways_alive(token, profiles, [])
    elapsed = time.monotonic() - started
    assert token["relaunched_profiles"] == sorted(profiles) and token["profiles"] == {}
    assert elapsed < 2 * _READY_CONFIRM_S, f"6 ready targets took {elapsed:.1f}s: verified one after another"


def test_each_target_is_credited_only_on_its_own_stable_gateway(running, monkeypatch):
    monkeypatch.setattr(update_cmd_windows, "_RELAUNCH_VERIFY_TIMEOUT_S", 3.0)
    born = time.monotonic()
    running["up"] = lambda: 50001
    running["old"] = lambda: 40002  # only the process the relaunch replaces
    running["flaps"] = lambda: 50003 if time.monotonic() - born < 1.0 else None  # died during confirmation
    profiles = {"up": 40001, "old": 40002, "flaps": 40003, "never": 40004}
    token = {"profiles": dict(profiles), "unmapped": []}
    with pytest.raises(RuntimeError, match="not verified alive: flaps, never, old"):
        _verify_relaunched_gateways_alive(token, profiles, [])
    assert token["relaunched_profiles"] == ["up"]
    assert sorted(token["profiles"]) == ["flaps", "never", "old"]


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # a stub child that sleeps; reaped below
def test_one_gateway_never_retires_two_unmapped_debts(running, monkeypatch, tmp_path):
    import os
    import subprocess
    import sys

    from hermes_cli import gateway
    monkeypatch.setattr(update_cmd_windows, "_RELAUNCH_VERIFY_TIMEOUT_S", 3.0)
    entry = tmp_path / "bin" / "hermes"
    entry.parent.mkdir()
    entry.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    argv = [sys.executable, str(entry), "gateway", "run"]
    # Pre-home entries (an older updater's token): argv is all that identifies them.
    debts = [{"pid": 40011, "argv": argv}, {"pid": 40012, "argv": argv}]
    child = subprocess.Popen(argv, env=dict(os.environ))
    try:
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [child.pid])
        token = {"profiles": {}, "unmapped": list(debts)}
        with pytest.raises(RuntimeError, match="pid 40012"):
            _verify_relaunched_gateways_alive(token, {}, debts)
        assert token["unmapped"] == [debts[1]], "one relaunched gateway retired both debts"
    finally:
        child.kill()
        child.wait(timeout=10)
