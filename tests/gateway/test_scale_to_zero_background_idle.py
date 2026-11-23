"""Scale-to-zero: a background process idle for ``background_idle_minutes`` stops blocking a suspend.

Background: a long-lived job that never exits (a dev server started with
``terminal(background=true)``) kept ``process_registry.has_any_active()`` True forever, so an
opted-in host never self-suspended. On a remote sandbox backend the env poller's exec every 2 s
also kept the sandbox awake. "Idle" = no new output AND no ``process_manage`` action for the
window; the process is not killed, it just no longer counts.

These drive the REAL ProcessRegistry (sessions in ``_running``, the real output paths, the real
``resolve_actionable`` agent-action gate) and the real ``GatewayRunner`` call sites, with a fake
monotonic clock.
"""
from __future__ import annotations

import logging
import sys
import time
from types import SimpleNamespace

import pytest

from gateway import scale_to_zero as s2z
from gateway.run import GatewayRunner
from tests.waiting import wait_until
import tools.process_registry as pr_mod
from tools.process_registry import ProcessRegistry, ProcessSession

MIN = 60.0
WINDOW = 30 * MIN


@pytest.fixture
def registry(tmp_path, monkeypatch):
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    reg = ProcessRegistry()
    monkeypatch.setattr(pr_mod, "process_registry", reg)
    s2z._idle_logged.clear()
    return reg


def _session(registry, *, last_activity_at=1000.0, sid="proc_idletest01", **kw) -> ProcessSession:
    s = ProcessSession(id=sid, command="python3 panel.py 8080", task_id="t1", **kw)
    s.last_activity_at = last_activity_at
    with registry._lock:
        registry._running[s.id] = s
    return s


# --- config parsing ----------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (None, WINDOW), ("junk", WINDOW), (-5, WINDOW), (float("nan"), WINDOW), (float("inf"), WINDOW),
    ([], WINDOW), (45, 45 * MIN), ("10", 10 * MIN), (0.5, 30.0),
    (0, None), ("0", None), (0.0, None),
])
def test_parse_background_idle_seconds(value, expected):
    assert s2z.parse_background_idle_seconds(value) == expected


# --- the predicate -----------------------------------------------------------


def test_fresh_process_blocks(registry):
    s = ProcessSession(id="proc_fresh0001", command="sleep 1000", task_id="t1")
    with registry._lock:
        registry._running[s.id] = s
    assert s2z.has_live_background_processes(WINDOW) is True


@pytest.mark.parametrize("idle_minutes, blocks", [(0, True), (29, True), (29.99, True), (30, False), (31, False)])
def test_idle_window_boundary(registry, idle_minutes, blocks):
    s = _session(registry)
    assert s2z.has_live_background_processes(WINDOW, now=s.last_activity_at + idle_minutes * MIN) is blocks


def test_rule_off_keeps_legacy_blocking(registry):
    s = _session(registry)
    assert s2z.has_live_background_processes(None) is True
    assert s2z.has_live_background_processes(None, now=s.last_activity_at + 10_000 * MIN) is True


def test_one_active_process_among_idle_ones_still_blocks(registry):
    _session(registry, sid="proc_idle00001", last_activity_at=0.0)
    _session(registry, sid="proc_busy00001", last_activity_at=31 * MIN)
    assert s2z.has_live_background_processes(WINDOW, now=31 * MIN + 1) is True
    assert s2z.has_live_background_processes(WINDOW, now=62 * MIN) is False


def test_exited_process_never_blocks(registry):
    s = _session(registry, last_activity_at=0.0)
    registry._finish_exited(s, 0)
    assert s2z.has_live_background_processes(WINDOW, now=1.0) is False


@pytest.fixture
def clock(monkeypatch):
    """Fake monotonic clock for the registry module (the rest of ``time`` stays real)."""
    now = [1000.0]
    fake = SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
    fake.monotonic = lambda: now[0]
    fake.sleep = lambda _s: None
    monkeypatch.setattr(pr_mod, "time", fake)
    import tools.process_registry_access as access
    monkeypatch.setattr(access, "time", fake)
    return now


def test_new_session_starts_its_clock_at_spawn(registry, clock):
    s = ProcessSession(id="proc_spawn0001", command="x")
    assert s.last_activity_at == 1000.0


def test_output_resets_the_clock(registry, clock):
    s = _session(registry, last_activity_at=clock[0])
    clock[0] += 31 * MIN
    assert s2z.has_live_background_processes(WINDOW, now=clock[0]) is False
    registry._ingest_output(s, "GET / 200\n")  # the local pipe/PTY reader path
    assert s.last_activity_at == clock[0]
    assert s2z.has_live_background_processes(WINDOW, now=clock[0] + 29 * MIN) is True


def test_agent_action_resets_the_clock(registry, clock):
    from tools.process_registry_access import resolve_actionable
    s = _session(registry, last_activity_at=clock[0])
    clock[0] += 31 * MIN
    assert resolve_actionable(s.id, "t1") == (True, s.id)  # every process_manage action goes through here
    assert s.last_activity_at == clock[0]
    assert s2z.has_live_background_processes(WINDOW, now=clock[0] + 1) is True


def test_refused_agent_action_does_not_reset_the_clock(registry, clock, monkeypatch):
    from tools import process_registry_results as results
    from tools.process_registry_access import resolve_actionable
    monkeypatch.setattr(results, "_owns_result", lambda owner, parent: False)
    s = _session(registry, last_activity_at=clock[0])
    clock[0] += 31 * MIN
    assert resolve_actionable(s.id, "someone-else")[0] is False
    assert s.last_activity_at == 1000.0


def test_process_manage_poll_resets_the_clock(registry, clock):
    s = _session(registry, last_activity_at=clock[0])
    clock[0] += 31 * MIN
    pr_mod._handle_process({"action": "poll", "session_id": s.id}, task_id="t1")
    assert s.last_activity_at == clock[0]


class _SandboxEnv:
    """A remote backend whose log grows once, then reports the job gone (warm-stopped sandbox)."""

    def __init__(self, clock, replies):
        self.clock, self.replies = clock, list(replies)

    def execute(self, command, cwd=None, timeout=None):
        self.clock[0] += 2 * MIN
        return {"output": self.replies.pop(0)}


def test_env_poller_delta_resets_the_clock(registry, clock):
    s = _session(registry, last_activity_at=clock[0], pid_scope="sandbox")
    env = _SandboxEnv(clock, ["6 0 run\nhello\n", "6 6 exit:0\n"])
    registry._env_poller_loop(s, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")
    assert s.last_activity_at == 1000.0 + 2 * MIN  # the tick that carried output
    assert s.output_buffer == "hello\n" and s.total_output_chars == 6
    assert (s.exited, s.exit_code) == (True, 0)


def test_env_job_gone_after_wake_settles_and_stops_blocking(registry, clock):
    """After the host suspends, the sandbox warm-stops and the job dies; on wake the poller sees the
    pid gone (no exit file) and settles the session, which then no longer counts at all."""
    s = _session(registry, last_activity_at=clock[0], pid_scope="sandbox")
    assert s2z.has_live_background_processes(WINDOW, now=clock[0] + 31 * MIN) is False
    registry._env_poller_loop(s, _SandboxEnv(clock, ["0 0 exit:\n"]), "/tmp/l", "/tmp/p", "/tmp/e")
    assert (s.exited, s.exit_code) == (True, -1)
    assert s2z.has_live_background_processes(None) is False


# --- pending notify watchers ---------------------------------------------------


def _watch(registry, sid):
    registry.pending_watchers.append({"session_id": sid, "check_interval": 30, "notify_on_complete": True})


def test_pending_watcher_of_an_idle_process_does_not_block(registry):
    s = _session(registry, notify_on_complete=True)
    _watch(registry, s.id)
    assert s2z.has_live_background_processes(WINDOW, now=s.last_activity_at + 31 * MIN) is False
    assert s2z.has_live_background_processes(WINDOW, now=s.last_activity_at + 29 * MIN) is True
    assert s2z.has_live_background_processes(None, now=s.last_activity_at + 31 * MIN) is True


def test_pending_watcher_of_an_exited_or_unknown_process_still_blocks(registry):
    s = _session(registry, notify_on_complete=True)
    registry._finish_exited(s, 0)
    _watch(registry, s.id)  # its completion still has to be delivered
    assert s2z.has_live_background_processes(WINDOW, now=s.last_activity_at + 31 * MIN) is True
    registry.pending_watchers[:] = []
    _watch(registry, "proc_unknown01")
    assert s2z.has_live_background_processes(WINDOW, now=10_000 * MIN) is True


# --- diagnosability -----------------------------------------------------------


def test_idle_exclusion_logs_once_per_idle_stretch(registry, caplog):
    s = _session(registry)
    caplog.set_level(logging.INFO, logger="gateway.scale_to_zero")

    def lines():
        return [r.getMessage() for r in caplog.records if "no longer blocks suspend" in r.getMessage()]
    for _ in range(3):
        s2z.has_live_background_processes(WINDOW, now=s.last_activity_at + 31 * MIN)
    assert lines() == [f"scale-to-zero: background process {s.id} idle 31 min; no longer blocks suspend"]
    s.last_activity_at += 40 * MIN  # it woke up, then went quiet again
    s2z.has_live_background_processes(WINDOW, now=s.last_activity_at + 45 * MIN)
    assert len(lines()) == 2 and "idle 45 min" in lines()[1]


# --- gateway call sites -------------------------------------------------------


def _runner(monkeypatch, cfg):
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._last_inbound_at = time.time() - 3600
    r._running_agents = {}
    r._background_tasks = set()
    r.adapters = {}
    monkeypatch.setattr(r, "_gateway_cfg_section", lambda name: cfg if name == "scale_to_zero" else None,
                        raising=False)
    monkeypatch.setattr("cron.scheduler.get_running_job_ids", list)
    monkeypatch.setattr(s2z, "dashboard_client_last_seen", lambda: None)
    return r


@pytest.mark.parametrize("cfg, idle", [
    ({}, True),                                   # default 30 min
    ({"background_idle_minutes": "junk"}, True),  # junk -> default
    ({"background_idle_minutes": 60}, False),     # 45 min quiet < 60 min window
    ({"background_idle_minutes": 0}, False),      # rule off: any running process blocks
])
def test_scale_to_zero_is_idle_applies_the_configured_window(registry, monkeypatch, cfg, idle):
    _session(registry, last_activity_at=time.monotonic() - 45 * MIN)
    r = _runner(monkeypatch, cfg)
    assert r._scale_to_zero_is_idle() is idle


def test_unmanaged_idle_exit_still_counts_idle_processes(registry, monkeypatch):
    """Exiting the gateway (unlike a suspend) takes its owned local processes with it, so the
    unmanaged idle-exit check keeps the old rule."""
    _session(registry, last_activity_at=time.monotonic() - 45 * MIN)
    r = _runner(monkeypatch, {})
    assert r._scale_to_zero_has_live_background_work() is True
    assert r._scale_to_zero_has_live_background_work(WINDOW) is False


# --- a real local background process ---------------------------------------------


def test_real_local_process_goes_idle_without_being_killed(registry):
    cmd = f"{sys.executable} -c \"import time; print('ready', flush=True); time.sleep(60)\""
    s = registry.spawn_local(cmd, cwd="/tmp", task_id="t-real")
    try:
        wait_until(lambda: "ready" in s.output_buffer, timeout=20)
        seen = s.last_activity_at
        assert time.monotonic() - seen < 20
        assert s2z.has_live_background_processes(WINDOW) is True
        assert s2z.has_live_background_processes(WINDOW, now=seen + 29 * MIN) is True
        assert s2z.has_live_background_processes(WINDOW, now=seen + 31 * MIN) is False
        assert registry.poll(s.id)["status"] == "running"  # excluded, not killed
        assert s2z.has_live_background_processes(None) is True
    finally:
        registry.kill_process(s.id)
