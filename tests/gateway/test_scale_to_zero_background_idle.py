"""Scale-to-zero: a background process the agent has not touched for ``background_idle_minutes``
stops blocking a suspend.

Background: a long-lived job that never exits (a dev server started with
``terminal(background=true)``) kept ``process_registry.has_any_active()`` True forever, so an
opted-in host never self-suspended. On a remote sandbox backend the env poller's exec every 2 s
also kept the sandbox awake. A process now counts only while the agent touched it (launch,
``process_manage`` action, handoff) within the window. Its own output never extends it: a server
logging a health check every minute would otherwise pin the host forever. The process is not
killed, it just no longer counts.

These drive the REAL ProcessRegistry (sessions in ``_running``, the real output paths, the real
``process_manage`` dispatcher and handoff) and the real ``GatewayRunner`` call sites, with a fake
wall clock.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import weakref
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


def _session(registry, *, last_touched_at=1000.0, sid="proc_idletest01", **kw) -> ProcessSession:
    s = ProcessSession(id=sid, command="python3 panel.py 8080", task_id="t1", **kw)
    s.last_touched_at = last_touched_at
    with registry._lock:
        registry._running[s.id] = s
    return s


# --- config parsing ----------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (None, WINDOW), ("junk", WINDOW), (-5, WINDOW), (float("nan"), WINDOW), (float("inf"), WINDOW),
    ([], WINDOW), pytest.param(10 ** 1000, WINDOW, id="10**1000"), (1e308, WINDOW), ("1e400", WINDOW),
    (45, 45 * MIN), ("10", 10 * MIN), (0.5, 30.0),
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


@pytest.mark.parametrize("minutes, blocks", [(0, True), (29, True), (29.99, True), (30, False), (31, False)])
def test_window_boundary(registry, minutes, blocks):
    s = _session(registry)
    assert s2z.has_live_background_processes(WINDOW, now=s.last_touched_at + minutes * MIN) is blocks


def test_rule_off_keeps_legacy_blocking(registry):
    s = _session(registry)
    assert s2z.has_live_background_processes(None) is True
    assert s2z.has_live_background_processes(None, now=s.last_touched_at + 10_000 * MIN) is True


def test_one_touched_process_among_untouched_ones_still_blocks(registry):
    _session(registry, sid="proc_idle00001", last_touched_at=0.0)
    _session(registry, sid="proc_busy00001", last_touched_at=31 * MIN)
    assert s2z.has_live_background_processes(WINDOW, now=31 * MIN + 1) is True
    assert s2z.has_live_background_processes(WINDOW, now=62 * MIN) is False


def test_exited_process_never_blocks(registry):
    s = _session(registry, last_touched_at=0.0)
    registry._finish_exited(s, 0)
    assert s2z.has_live_background_processes(WINDOW, now=1.0) is False


@pytest.fixture
def clock(monkeypatch):
    """Fake wall clock for the registry modules that stamp touches (the rest of ``time`` stays real)."""
    now = [1000.0]
    fake = SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
    fake.time = lambda: now[0]
    fake.sleep = lambda _s: None
    import tools.process_registry_access as access
    import tools.process_registry_handoff as handoff
    for mod in (pr_mod, access, handoff):
        monkeypatch.setattr(mod, "time", fake)
    return now


def test_launch_starts_the_clock(registry, clock):
    assert ProcessSession(id="proc_spawn0001", command="x").last_touched_at == 1000.0


def test_health_log_every_minute_stops_blocking_at_31_min(registry, clock):
    """The always-on hole: routine output must not keep the window open."""
    s = _session(registry, last_touched_at=clock[0])
    for minute in range(1, 32):
        clock[0] = 1000.0 + minute * MIN
        registry._ingest_output(s, "GET /health 200\n")  # the local pipe/PTY reader path
        assert s2z.has_live_background_processes(WINDOW, now=clock[0]) is (minute < 30), minute
    assert s.last_touched_at == 1000.0
    assert s.total_output_chars == 31 * len("GET /health 200\n")


def test_agent_action_resets_the_clock(registry, clock):
    from tools.process_registry_access import resolve_actionable
    s = _session(registry, last_touched_at=clock[0])
    clock[0] += 31 * MIN
    assert resolve_actionable(s.id, "t1") == (True, s.id)  # every process_manage session action goes through here
    assert s.last_touched_at == clock[0]
    assert s2z.has_live_background_processes(WINDOW, now=clock[0] + 1) is True


def test_refused_agent_action_does_not_reset_the_clock(registry, clock, monkeypatch):
    from tools import process_registry_results as results
    from tools.process_registry_access import resolve_actionable
    monkeypatch.setattr(results, "_owns_result", lambda owner, parent: False)
    s = _session(registry, last_touched_at=clock[0])
    clock[0] += 31 * MIN
    assert resolve_actionable(s.id, "someone-else")[0] is False
    assert s.last_touched_at == 1000.0


def test_process_manage_poll_resets_the_clock(registry, clock):
    s = _session(registry, last_touched_at=clock[0])
    clock[0] += 31 * MIN
    pr_mod._handle_process({"action": "poll", "session_id": s.id}, task_id="t1")
    assert s.last_touched_at == clock[0]


class _Parent:
    session_id = "sess-parent"
    _current_task_id = "parent-turn-1"


class _Child:
    def __init__(self, parent):
        self._delegate_parent_ref = weakref.ref(parent)


@pytest.fixture
def child_agent():
    from tools.delegate_tool import _register_subagent, _unregister_subagent
    parent = _Parent()
    child = _Child(parent)
    _register_subagent({"subagent_id": "sa-0-idle01", "parent_id": None, "depth": 0, "goal": "g", "model": "m",
                        "started_at": time.time(), "status": "running", "tool_count": 0, "agent": child})
    yield "sa-0-idle01", parent
    _unregister_subagent("sa-0-idle01")


def test_successful_handoff_resets_the_clock(registry, clock, child_agent):
    sid, _parent = child_agent
    s = _session(registry, last_touched_at=clock[0], owner_task_id=sid)
    clock[0] += 31 * MIN
    out = json.loads(pr_mod._handle_process(
        {"action": "handoff", "session_id": s.id, "data": "dev server for the parent"}, task_id=sid))
    assert out["status"] == "handed_off" and s.owner_task_id == "parent-turn-1"
    assert s.last_touched_at == clock[0]
    assert s2z.has_live_background_processes(WINDOW, now=clock[0] + 29 * MIN) is True


def test_refused_handoff_does_not_reset_the_clock(registry, clock, child_agent):
    sid, _parent = child_agent
    s = _session(registry, last_touched_at=clock[0], owner_task_id="someone-else")
    clock[0] += 31 * MIN
    out = json.loads(pr_mod._handle_process({"action": "handoff", "session_id": s.id, "data": "x"}, task_id=sid))
    assert "error" in out
    assert s.last_touched_at == 1000.0


class _SandboxEnv:
    """A remote backend: one scripted ``log_delta_command`` reply per poll tick."""

    def __init__(self, clock, replies):
        self.clock, self.replies = clock, list(replies)

    def execute(self, command, cwd=None, timeout=None):
        self.clock[0] += 2 * MIN
        return {"output": self.replies.pop(0)}


def test_env_poller_output_is_kept_but_does_not_reset_the_clock(registry, clock):
    s = _session(registry, last_touched_at=clock[0], pid_scope="sandbox")
    env = _SandboxEnv(clock, ["6 0 run\nhello\n", "6 6 exit:0\n"])
    registry._env_poller_loop(s, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")
    assert s.last_touched_at == 1000.0
    assert s.output_buffer == "hello\n" and s.total_output_chars == 6
    assert (s.exited, s.exit_code, s.completion_reason) == (True, 0, "exited")


def test_env_job_gone_after_wake_settles_as_lost_sandbox_stopped(registry, clock):
    """Host suspends, the sandbox stops and the job dies without writing its exit file. On wake the
    poller sees the pid gone: the session settles as lost, with a message naming the stopped sandbox."""
    from tools.process_registry_notifications import format_process_notification
    s = _session(registry, last_touched_at=clock[0], pid_scope="sandbox", notify_on_complete=True)
    assert s2z.has_live_background_processes(WINDOW, now=clock[0] + 31 * MIN) is False
    registry._env_poller_loop(s, _SandboxEnv(clock, ["0 0 exit:\n"]), "/tmp/l", "/tmp/p", "/tmp/e")
    assert (s.exited, s.exit_code, s.completion_reason, s.termination_source) == (True, -1, "lost", "sandbox_stopped")
    assert s2z.has_live_background_processes(None) is False
    evt = registry.completion_queue.get_nowait()
    assert "sandbox stopped while the host was asleep" in (format_process_notification(evt) or "")


# --- pending notify watchers ---------------------------------------------------


def _watch(registry, sid):
    registry.pending_watchers.append({"session_id": sid, "check_interval": 30, "notify_on_complete": True})


def test_pending_watcher_gets_no_grace_beyond_the_window(registry):
    """An awaited job is protected for 30 min from its launch (the last touch), no longer."""
    s = _session(registry, notify_on_complete=True)
    _watch(registry, s.id)
    assert s2z.has_live_background_processes(WINDOW, now=s.last_touched_at + 29 * MIN) is True
    assert s2z.has_live_background_processes(WINDOW, now=s.last_touched_at + 31 * MIN) is False
    assert s2z.has_live_background_processes(None, now=s.last_touched_at + 31 * MIN) is True


def test_pending_watcher_of_an_exited_or_unknown_process_still_blocks(registry):
    s = _session(registry, notify_on_complete=True)
    registry._finish_exited(s, 0)
    _watch(registry, s.id)  # its completion still has to be delivered
    assert s2z.has_live_background_processes(WINDOW, now=s.last_touched_at + 31 * MIN) is True
    registry.pending_watchers[:] = []
    _watch(registry, "proc_unknown01")
    assert s2z.has_live_background_processes(WINDOW, now=10_000 * MIN) is True


# --- diagnosability -----------------------------------------------------------


def test_exclusion_logs_once_per_untouched_stretch(registry, caplog):
    s = _session(registry)
    caplog.set_level(logging.INFO, logger="gateway.scale_to_zero")

    def lines():
        return [r.getMessage() for r in caplog.records if "no longer blocks suspend" in r.getMessage()]
    for _ in range(3):
        s2z.has_live_background_processes(WINDOW, now=s.last_touched_at + 31 * MIN)
    assert lines() == [f"scale-to-zero: background process {s.id} untouched for 31 min; no longer blocks suspend"]
    s.last_touched_at += 40 * MIN  # the agent polled it, then left it again
    s2z.has_live_background_processes(WINDOW, now=s.last_touched_at + 45 * MIN)
    assert len(lines()) == 2 and "untouched for 45 min" in lines()[1]


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
    ({"background_idle_minutes": 60}, False),     # 45 min untouched < 60 min window
    ({"background_idle_minutes": 0}, False),      # rule off: any running process blocks
])
def test_scale_to_zero_is_idle_applies_the_configured_window(registry, monkeypatch, cfg, idle):
    _session(registry, last_touched_at=time.time() - 45 * MIN)
    r = _runner(monkeypatch, cfg)
    assert r._scale_to_zero_is_idle() is idle


def test_unmanaged_idle_exit_still_counts_untouched_processes(registry, monkeypatch):
    """Exiting the gateway (unlike a suspend) takes its owned local processes with it, so the
    unmanaged idle-exit check keeps the old rule."""
    _session(registry, last_touched_at=time.time() - 45 * MIN)
    r = _runner(monkeypatch, {})
    assert r._scale_to_zero_has_live_background_work() is True
    assert r._scale_to_zero_has_live_background_work(WINDOW) is False


# --- a real local background process ---------------------------------------------


def test_real_chatty_local_process_stops_blocking_without_being_killed(registry):
    code = "import time\nwhile True:\n    print('GET /health 200', flush=True)\n    time.sleep(0.2)\n"
    s = registry.spawn_local(f"{sys.executable} -c \"{code}\"", cwd="/tmp", task_id="t-real")
    try:
        launched = s.last_touched_at
        wait_until(lambda: s.output_buffer.count("GET /health") >= 3, timeout=20)
        assert s.last_touched_at == launched  # its output did not touch it
        assert s2z.has_live_background_processes(WINDOW) is True
        assert s2z.has_live_background_processes(WINDOW, now=launched + 29 * MIN) is True
        assert s2z.has_live_background_processes(WINDOW, now=launched + 31 * MIN) is False
        assert registry.poll(s.id)["status"] == "running"  # excluded, not killed
        assert s2z.has_live_background_processes(None) is True
    finally:
        registry.kill_process(s.id)
