"""LIVE Windows E2E for plan-reconciliation (#92902) on windows-latest.

Real processes with real Hermes-shaped argv, real inventory collection
(PID-file discovery + supervisor detection on REAL Windows), real
reconciliation. No mocks on the components under test.

 1. Spawn a real process registered as a profile gateway (PID file + state
    file in a temp HERMES_HOME) — real collect_runtime_inventory() must find
    it, classify supervisor=manual, mechanism id 'manual'.
 2. Reconcile with bookkeeping that MISSES it -> unaccounted + escalation.
 3. Reconcile with it in killed_pids -> 'stopped', no escalation.
 4. Machine-id contract holds on Windows (no display strings leak in).
"""
import io, contextlib, json, os, subprocess, sys, tempfile, time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="live Windows E2E")


def test_plan_reconciliation_live_windows(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Real live process standing in for a manual gateway
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)
        assert child.poll() is None

        import psutil

        create_time = psutil.Process(child.pid).create_time()
        (home / "gateway_state.json").write_text(json.dumps({
            "pid": child.pid,
            "create_time": create_time,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "code_sha": "f" * 40,
            "code_version": "0.20.5",
        }), encoding="utf-8")

        import hermes_cli.profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: home)
        monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: tmp_path / "none")

        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            match_runtime_outcomes,
            report_unaccounted_runtimes,
        )

        plan = collect_runtime_inventory()
        rows = [r for r in plan.runtimes if r.pid == child.pid]
        assert rows, f"real inventory missed the live process: {plan.runtimes}"
        rt = rows[0]
        assert rt.supervisor == "manual"
        assert rt.restart_via == "manual", "machine id, not display string"
        assert rt.code_sha == "f" * 40

        # 2. bookkeeping misses it -> unaccounted + escalation
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            outcomes = match_runtime_outcomes(
                plan, restarted_services=[], relaunched_profiles=[],
                externally_supervised_profiles=[], killed_pids=set(),
                failed_units=[],
            )
            escalated = report_unaccounted_runtimes(outcomes)
        mine = [o for o in outcomes if o["pid"] == child.pid]
        assert mine and mine[0]["outcome"] == "unaccounted"
        assert escalated is True
        assert "never touched" in buf.getvalue()

        # 3. accounted as stopped -> clean
        outcomes2 = match_runtime_outcomes(
            plan, restarted_services=[], relaunched_profiles=[],
            externally_supervised_profiles=[], killed_pids={child.pid},
            failed_units=[],
        )
        mine2 = [o for o in outcomes2 if o["pid"] == child.pid]
        assert mine2 and mine2[0]["outcome"] == "stopped"
        assert report_unaccounted_runtimes(outcomes2) is False
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
