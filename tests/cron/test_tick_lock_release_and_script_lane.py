"""A long cron run must not stall every other scheduled job.

Live incident (2026-10-22, macOS host, system-crontab ticker + gateway ticker): the crontab's
``hermes cron tick`` is ``sync=True`` and used to hold ``<home>/cron/.tick.lock`` until its last
job finished. An agent job that ran for an hour therefore turned every gateway tick into "Tick
skipped — another instance holds the lock", and a 10-minute no-agent script silently missed its
slots. With ``cron.max_parallel_jobs: 1`` the same script also queued behind the agent run inside
the shared pool.

Drives the REAL ``tick()`` against a throwaway HERMES_HOME. ``run_job`` is the only stub: it blocks
the "long" job on an event so overlap is observable, and records every start.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.waiting import wait_until

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MACHINE_ID", raising=False)
    monkeypatch.delenv("HERMES_CRON_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("HERMES_CRON_MAX_PARALLEL_SCRIPTS", raising=False)
    monkeypatch.delenv("HERMES_CRON_HEADLESS_DEFER_AGENT_JOBS", raising=False)

    import cron.executions as E
    import cron.jobs as J
    import cron.scheduler as S

    monkeypatch.setattr(J, "HERMES_DIR", home)
    monkeypatch.setattr(J, "CRON_DIR", home / "cron")
    monkeypatch.setattr(J, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(J, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(E, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(S, "_hermes_home", home)
    S._shutdown_parallel_pool()
    S._running_job_ids.clear()
    S._running_since.clear()
    S._running_futures.clear()

    release_long = threading.Event()
    long_started = threading.Event()
    starts: list[str] = []
    active: dict[str, int] = {}
    max_active: dict[str, int] = {}
    hooks: dict = {}
    guard = threading.Lock()

    def fake_run_job(job, **_kw):
        name = job.get("name")
        with guard:
            starts.append(name)
            active[name] = active.get(name, 0) + 1
            max_active[name] = max(max_active.get(name, 0), active[name])
        try:
            if name in hooks:
                hooks[name](job)
            if name == "long":
                long_started.set()
                assert release_long.wait(30), "test never released the long job"
            return True, "out", "[SILENT]", None
        finally:
            with guard:
                active[name] -= 1

    monkeypatch.setattr(S, "run_job", fake_run_job)
    monkeypatch.setattr(S, "_deliver_result", lambda *_a, **_kw: None)

    def add_job(name: str, *, no_agent: bool = False, schedule: str = "every 10m",
                ago: int = 30) -> str:
        if no_agent:
            job = J.create_job(prompt=None, schedule=schedule, name=name, script="noop.sh",
                               no_agent=True, deliver="local")
        else:
            job = J.create_job(prompt="p", schedule=schedule, name=name, deliver="local")
        make_due(job["id"], ago)
        return job["id"]

    def make_due(job_id: str, ago: int = 30) -> None:
        """Set a past slot; distinct ``ago`` values give distinct occurrence identities."""
        slot = (J._hermes_now() - timedelta(seconds=ago)).replace(microsecond=0).isoformat()
        stored = J.load_jobs()
        next(r for r in stored if r["id"] == job_id)["next_run_at"] = slot
        J.save_jobs(stored)

    def tick_in_thread() -> threading.Thread:
        t = threading.Thread(target=lambda: S.tick(verbose=False, sync=True), daemon=True)
        t.start()
        assert long_started.wait(20), "long job never started"
        return t

    ns = dict(home=home, S=S, J=J, E=E, release_long=release_long, starts=starts,
              max_active=max_active, hooks=hooks, add_job=add_job, make_due=make_due,
              tick_in_thread=tick_in_thread)
    yield ns
    release_long.set()
    S._shutdown_parallel_pool()


def _tick_lock_is_free(S) -> bool:
    _lock_dir, lock_file = S._get_lock_paths()
    fd = S._acquire_tick_lock(lock_file)
    if fd is None:
        return False
    S._release_tick_lock(fd)
    return True


def test_sync_tick_releases_the_tick_lock_while_its_long_job_runs(env):
    S = env["S"]
    env["add_job"]("long")
    t = env["tick_in_thread"]()

    # The sync tick is still waiting on its run, but another ticker may scan now.
    assert t.is_alive()
    assert _tick_lock_is_free(S)

    env["add_job"]("short", no_agent=True)
    assert S.tick(verbose=False, sync=True) == 1
    assert env["starts"] == ["long", "short"]

    env["release_long"].set()
    t.join(10)
    assert not t.is_alive()


def test_script_lane_runs_scripts_beside_a_long_agent_job_at_max_parallel_one(env, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
    monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL_SCRIPTS", "1")
    S = env["S"]
    env["add_job"]("long")
    t = env["tick_in_thread"]()

    env["add_job"]("sync", no_agent=True)
    done = threading.Event()
    threading.Thread(target=lambda: (S.tick(verbose=False, sync=True), done.set()),
                     daemon=True).start()
    assert done.wait(10), "script queued behind the agent run despite its own lane"
    assert "sync" in env["starts"]

    env["release_long"].set()
    t.join(10)


def test_without_the_setting_scripts_still_share_the_single_slot(env, monkeypatch):
    """Default unchanged: max_parallel_jobs stays the one limit for both kinds."""
    monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
    S = env["S"]
    env["add_job"]("long")
    t = env["tick_in_thread"]()

    env["add_job"]("sync", no_agent=True)
    done = threading.Event()
    threading.Thread(target=lambda: (S.tick(verbose=False, sync=True), done.set()),
                     daemon=True).start()
    assert not done.wait(1.5)
    assert "sync" not in env["starts"]

    env["release_long"].set()
    assert done.wait(10)
    assert env["starts"] == ["long", "sync"]
    t.join(10)


_OTHER_PROCESS_TICK = """
import sys
sys.path.insert(0, sys.argv[1])
import cron.constants
if sys.argv[2]:
    cron.constants.FIRE_CLAIM_TTL_SECONDS = float(sys.argv[2])
import cron.scheduler as S
starts = []
S.run_job = lambda job, **kw: starts.append(job["name"]) or (True, "out", "[SILENT]", None)
S._deliver_result = lambda *a, **k: None
S.tick(verbose=False, sync=True)
print("STARTS=" + ",".join(starts))
"""


def _tick_in_other_process(home: Path, lease_seconds: float | None = None) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _OTHER_PROCESS_TICK, str(REPO),
         "" if lease_seconds is None else str(lease_seconds)],
        env=dict(os.environ, HERMES_HOME=str(home)), cwd=str(REPO),
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout.rsplit("STARTS=", 1)[-1].strip()


def test_overlapping_tick_in_another_process_never_runs_the_same_job_twice(env):
    """The lock no longer serializes runs, so the per-job fire claim must: while this process
    runs ``long``, another process's tick that finds it due again loses the claim."""
    jid = env["add_job"]("long", schedule="every 1m")
    t = env["tick_in_thread"]()

    env["make_due"](jid)
    assert _tick_in_other_process(env["home"]) == ""
    assert env["max_active"]["long"] == 1

    env["release_long"].set()
    t.join(10)
    assert env["starts"] == ["long"]
    assert env["J"].get_job(jid)["fire_claim"] is None  # mark_job_run released it


def test_job_paused_mid_run_finishes_stays_paused_and_does_not_replay(env):
    S, J = env["S"], env["J"]
    jid = env["add_job"]("long")
    env["hooks"]["long"] = lambda _job: J.pause_job(jid)
    t = env["tick_in_thread"]()
    env["release_long"].set()
    t.join(10)

    paused = J.get_job(jid)
    assert paused["state"] == "paused" and paused["enabled"] is False
    assert paused["fire_claim"] is None

    env["make_due"](jid, ago=20)  # a slot passes while paused
    assert S.tick(verbose=False, sync=True) == 0
    assert env["starts"] == ["long"]

    # Resume: the slot that elapsed while paused goes through the catch-up policy -- one fire,
    # never one per missed slot -- and the schedule then moves to the future.
    env["hooks"].clear()
    env["release_long"].set()
    J.resume_job(jid)
    assert J.get_job(jid)["state"] == "scheduled"
    assert S.tick(verbose=False, sync=True) == 1
    assert S.tick(verbose=False, sync=True) == 0
    assert env["starts"] == ["long", "long"]
    assert datetime.fromisoformat(J.get_job(jid)["next_run_at"]) > J._hermes_now()


_CRASH_MID_RUN = """
import os, sys
sys.path.insert(0, sys.argv[1])
import cron.scheduler as S
S.run_job = lambda job, **kw: os._exit(137)  # SIGKILL while the claimed run is executing
S.tick(verbose=False, sync=True)
"""


def test_run_killed_mid_flight_never_blocks_or_doubles_the_next_slot(env):
    """Process 1 claims an occurrence and dies inside the run (gateway restart, reboot). The
    survivor reaps the dead owner at once (no claim-TTL wait) and records that execution as
    unknown: an occurrence whose side effects may already have happened is not replayed
    (at-most-once). The next slot then runs exactly once."""
    S, J, E = env["S"], env["J"], env["E"]
    jid = env["add_job"]("crashy", schedule="every 1m", ago=50)
    proc = subprocess.run([sys.executable, "-c", _CRASH_MID_RUN, str(REPO)],
                          env=dict(os.environ, HERMES_HOME=str(env["home"])), cwd=str(REPO),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 137, proc.stderr[-3000:]
    assert J.get_job(jid)["fire_claim"] is not None  # the dead run's claim was left behind

    assert S.tick(verbose=False, sync=True) == 0  # crashed occurrence: not replayed
    assert [r["status"] for r in E.list_executions(job_id=jid)] == ["unknown"]

    env["make_due"](jid, ago=5)  # next slot
    assert S.tick(verbose=False, sync=True) == 1
    assert S.tick(verbose=False, sync=True) == 0
    assert env["starts"] == ["crashy"]
    assert env["max_active"]["crashy"] == 1


def test_sync_tick_keeps_queued_runs_leases_fresh_while_it_waits(env, monkeypatch):
    """At max_parallel_jobs=1 the second due job waits in the pool with no fire claim. Its
    pending_slot stamp and execution row must stay fresh, or another process's scan (now
    possible: the lock is free) would restore the slot and the occurrence would run twice."""
    import cron.scheduler_tick as tick_mod

    monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
    monkeypatch.setattr(tick_mod, "_QUEUED_LEASE_REFRESH_SECONDS", 0.2)
    J, E = env["J"], env["E"]
    env["add_job"]("long")
    queued = env["add_job"]("queued", ago=40)
    t = env["tick_in_thread"]()

    first = J.get_job(queued)["pending_slot"]
    later = wait_until(
        lambda: (slot := J.get_job(queued)["pending_slot"])["at"] != first["at"] and slot,
        timeout=10, what="queued run's pending_slot refreshed")
    assert later["by"] == first["by"] and later["scheduled_at"] == first["scheduled_at"]
    assert datetime.fromisoformat(later["at"]) > datetime.fromisoformat(first["at"])
    row = wait_until(lambda: (r := E.list_executions(job_id=queued)[0]).get("progress_at") and r,
                     timeout=10, what="queued run's execution row stamped")
    assert row["status"] == "claimed"
    assert "queued" not in env["starts"]

    # Another process scans with a 1s slot lease (>> the 0.2s refresh): the stamp still reads
    # owned, so it neither restores nor runs the queued occurrence.
    assert _tick_in_other_process(env["home"], lease_seconds=1) == ""

    env["release_long"].set()
    t.join(10)
    assert env["starts"] == ["long", "queued"]
    assert J.get_job(queued).get("pending_slot") is None  # its claim cleared it


def test_headless_tick_defers_agent_jobs_to_a_live_gateway_ticker(env, monkeypatch):
    import cron.scheduler_gateway_gate as gate
    import cron.scheduler_ownership as ownership

    J = env["J"]
    agent = env["add_job"]("agent")
    script = env["add_job"]("script", no_agent=True)
    gateway = {"pid": os.getpid()}
    monkeypatch.setattr(ownership, "live_gateway_ticking", lambda _home=None: gateway)
    monkeypatch.setattr(gate, "_gateway_unavailable", lambda _home: None)
    J.record_ticker_heartbeat(success=True)  # the gateway's ticker loop is iterating and firing
    due = [J.get_job(agent), J.get_job(script)]

    # Off (default): a ready gateway means the headless tick fires both itself.
    assert [j["id"] for j in gate.refuse_agent_jobs_without_gateway(due)] == [agent, script]

    # On: the REAL headless tick runs only the script; the agent job stays due, unstamped.
    monkeypatch.setenv("HERMES_CRON_HEADLESS_DEFER_AGENT_JOBS", "1")
    S = env["S"]
    assert S.tick(verbose=False, sync=True, headless=True) == 1
    assert env["starts"] == ["script"]
    left = J.get_job(agent)
    assert not left.get("last_fire_error")  # deferred, not a failure
    assert not left.get("pending_slot")
    assert left["next_run_at"] == due[0]["next_run_at"]  # still due for the gateway's tick
    assert [j["id"] for j in J.get_due_jobs()] == [agent]
    env["make_due"](script)  # due again for the probes below
    J.record_ticker_heartbeat(success=True)
    assert [j["id"] for j in gate.refuse_agent_jobs_without_gateway(due)] == [script]

    # A heartbeat not written by that gateway (bare epoch from a wrapper, another pid) is no
    # proof its ticker runs: fire them here as before.
    (env["home"] / "cron" / "ticker_heartbeat").write_text(f"{time.time()}\n", encoding="utf-8")
    assert [j["id"] for j in gate.refuse_agent_jobs_without_gateway(due)] == [agent, script]
    J.record_ticker_heartbeat(success=True)
    gateway["pid"] = os.getpid() + 1
    assert [j["id"] for j in gate.refuse_agent_jobs_without_gateway(due)] == [agent, script]

    # No live gateway ticker: nothing to defer to, fall back to the ready/refuse probe.
    monkeypatch.setattr(ownership, "live_gateway_ticking", lambda _home=None: None)
    assert [j["id"] for j in gate.refuse_agent_jobs_without_gateway(due)] == [agent, script]
