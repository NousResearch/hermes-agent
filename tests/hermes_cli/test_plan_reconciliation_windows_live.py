"""LIVE Windows E2E for plan-reconciliation (#92902) on windows-latest.

Real processes with real Hermes-shaped argv, real inventory collection
(PID-file discovery + supervisor detection on REAL Windows), real
reconciliation. No mocks on the components under test.

 1. Spawn a real process with a `gateway run` command line registered as the
    profile gateway (state file in a temp HERMES_HOME) — real
    collect_runtime_inventory() must verify and find it, classify
    supervisor=manual, mechanism id 'manual' (machine id, not display string).
 2. Reconcile with bookkeeping that MISSES it -> unaccounted + escalation.
 3. Reconcile with it in killed_pids -> 'stopped', no escalation.
 4. A running record naming a live NON-gateway process is not a runtime
    (verified identity, not bare PID existence — #109680).
"""
import io, contextlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))

import pytest

from tests.live_process_fixtures import SLEEPER_MARKER, sleeper_script_path

# The stand-in wears a `gateway run` argv; the test spawns and reaps it itself.
pytestmark = [pytest.mark.platforms("windows"), pytest.mark.spawns_gateway_lookalike]


def _wait_until(predicate, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _argv_visible(pid: int, marker: str) -> bool:
    """True once the process table shows *pid* with *marker* in its argv."""
    import psutil

    try:
        return marker in " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def test_plan_reconciliation_live_windows(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Real live process standing in for a manual gateway. The inventory's
    # state-file fallback proves identity through live_gateway_pid_for_home,
    # which requires a live `gateway run` command line (#109680), so the
    # stand-in wears one; a bare sleeper is recorded too and must NOT count.
    child = subprocess.Popen(
        [sys.executable, sleeper_script_path(), "hermes", "gateway", "run"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    foreign = subprocess.Popen(
        [sys.executable, sleeper_script_path()],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until(lambda: _argv_visible(child.pid, "gateway")), "stand-in argv never visible"
        assert _wait_until(lambda: _argv_visible(foreign.pid, SLEEPER_MARKER)), "sleeper never visible"

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

        # 4. The same running record naming a live NON-gateway process (a
        # recycled PID) is not a runtime: bare liveness is not identity.
        (home / "gateway_state.json").write_text(json.dumps({
            "pid": foreign.pid,
            "create_time": psutil.Process(foreign.pid).create_time(),
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "code_sha": "f" * 40,
        }), encoding="utf-8")
        plan2 = collect_runtime_inventory()
        assert [r for r in plan2.runtimes if r.pid == foreign.pid] == [], plan2.runtimes
    finally:
        for proc in (child, foreign):
            if proc.poll() is None:
                # /T: uv's venv python.exe is a trampoline; kill the real interpreter too.
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            proc.wait()
