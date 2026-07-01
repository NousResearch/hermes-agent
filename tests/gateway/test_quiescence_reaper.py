"""Busy-gateway-quiescence Phase 2 — the task-liveness reaper (run.py).

Spec: ~/.hermes/plans/2026-06-30_safe-restart-busy-gateway-quiescence-SPEC.md (v0.4) §5 D-3/D-4

The reaper evicts a `_running_agents` entry ONLY when its turn task is genuinely
done()/cancelled (a leaked slot) — NEVER a live long-running turn, regardless of age.
This is the amputation-safe replacement for the unsafe idle/wall-age `:8861` predicate
(which can't run proactively because `_touch_activity` is dead code → the idle signal is
fake). A genuinely-leaked slot only happens when a turn coroutine is killed without
unwinding (a hard kickstart SIGKILL mid-turn, an unhandled handler crash).

ACs covered: 4 (dead reaped), 5 (live survives — the B1 killer), 5b (registration window —
the B1-NEW killer), 6 (crash-safe), 10 (idempotent double-release).
"""
import time
from unittest.mock import MagicMock

from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL


def _runner():
    r = object.__new__(GatewayRunner)
    r._running_agents = {}
    r._running_agents_ts = {}
    r._running_agent_tasks = {}
    r._active_session_leases = {}
    r._session_run_generation = {}
    r._update_runtime_status = MagicMock()
    r._persist_active_agents = MagicMock()
    return r


def _done_task():
    t = MagicMock(name="done_task")
    t.done.return_value = True
    return t


def _live_task():
    t = MagicMock(name="live_task")
    t.done.return_value = False
    return t


def test_reaps_entry_whose_task_is_done():
    """AC-4: a slot whose turn task is done() (leaked — the finally never ran) IS reaped."""
    r = _runner()
    sk = "agent:main:discord:group:1:2"
    r._running_agents[sk] = object()             # real agent, but its task is dead
    r._running_agents_ts[sk] = time.time() - 5
    r._running_agent_tasks[sk] = _done_task()

    n = r._reap_dead_running_agents(now=time.time())

    assert n == 1
    assert sk not in r._running_agents           # leaked slot reclaimed
    assert sk not in r._running_agent_tasks


def test_never_reaps_live_long_running_task():
    """AC-5 (the B1 killer): a live task running FAR past any idle/wall threshold (a 6h
    delegated fan-out) is NEVER reaped — task.done() is False, that's the only thing that matters."""
    r = _runner()
    sk = "agent:main:discord:group:1:2"
    r._running_agents[sk] = object()
    r._running_agents_ts[sk] = time.time() - 6 * 3600   # 6 HOURS old — would trip wall_ttl
    r._running_agent_tasks[sk] = _live_task()            # but task is still running

    n = r._reap_dead_running_agents(now=time.time())

    assert n == 0
    assert sk in r._running_agents                       # SURVIVES — no amputation


def test_registration_window_entry_survives_grace(tmp_path):
    """AC-5b (the B1-NEW killer): a non-sentinel entry whose task handle is NOT yet recorded
    (task is None) but whose entry-age is within the grace window SURVIVES (it's a turn that
    just swapped sentinel→real and is a line away from recording its task)."""
    r = _runner()
    sk = "agent:main:discord:group:1:2"
    r._running_agents[sk] = object()             # real agent, mid-registration
    r._running_agents_ts[sk] = time.time() - 0.1  # 100ms old — within grace
    # no entry in _running_agent_tasks (not yet captured)

    n = r._reap_dead_running_agents(now=time.time())

    assert n == 0
    assert sk in r._running_agents                # survives the grace window


def test_no_task_past_grace_is_reaped():
    """The complement of 5b: a non-sentinel entry with NO task handle AND age past the grace
    window is a genuine leak (the capture truly never ran / was lost) → reaped."""
    r = _runner()
    sk = "agent:main:discord:group:1:2"
    r._running_agents[sk] = object()
    r._running_agents_ts[sk] = time.time() - 9999   # way past grace
    # no task handle

    n = r._reap_dead_running_agents(now=time.time())

    assert n == 1
    assert sk not in r._running_agents


def test_pending_sentinel_within_window_survives():
    """A pending sentinel (async setup not finished) is NEVER reaped within its setup window."""
    r = _runner()
    sk = "agent:main:discord:group:1:2"
    r._running_agents[sk] = _AGENT_PENDING_SENTINEL
    r._running_agents_ts[sk] = time.time() - 1     # fresh
    n = r._reap_dead_running_agents(now=time.time())
    assert n == 0
    assert sk in r._running_agents


def test_mixed_sweep_only_dead_reaped():
    """A realistic sweep: dead-task + live-task + fresh-sentinel → only the dead one goes."""
    r = _runner()
    now = time.time()
    r._running_agents = {
        "dead": object(),
        "live": object(),
        "pending": _AGENT_PENDING_SENTINEL,
    }
    r._running_agents_ts = {"dead": now - 10, "live": now - 10, "pending": now - 1}
    r._running_agent_tasks = {"dead": _done_task(), "live": _live_task()}

    n = r._reap_dead_running_agents(now=now)

    assert n == 1
    assert "dead" not in r._running_agents
    assert "live" in r._running_agents
    assert "pending" in r._running_agents


def test_sweep_is_crash_safe_per_entry():
    """AC-6: an exception classifying ONE entry does not abort the sweep — the others are
    still processed (per-entry try/except)."""
    r = _runner()
    now = time.time()
    boom = MagicMock(name="boom_task")
    boom.done.side_effect = RuntimeError("boom")     # this entry raises on inspection
    r._running_agents = {"boom": object(), "dead": object()}
    r._running_agents_ts = {"boom": now - 10, "dead": now - 10}
    r._running_agent_tasks = {"boom": boom, "dead": _done_task()}

    n = r._reap_dead_running_agents(now=now)   # must NOT raise

    assert "dead" not in r._running_agents     # the good entry was still reaped
    assert n == 1


def test_sweep_persists_after_eviction():
    """A sweep that evicts ≥1 entry refreshes the persisted status (so active_agent_keys updates)."""
    r = _runner()
    sk = "x"
    r._running_agents[sk] = object()
    r._running_agents_ts[sk] = time.time() - 10
    r._running_agent_tasks[sk] = _done_task()
    r._reap_dead_running_agents(now=time.time())
    r._persist_active_agents.assert_called()


def test_empty_sweep_no_persist_no_error():
    """An idle gateway (no running agents) sweeps to a no-op, no error, no spurious persist."""
    r = _runner()
    n = r._reap_dead_running_agents(now=time.time())
    assert n == 0


def test_concurrent_register_release_during_sweep_no_keyerror():
    """AC-6 (concurrency): a slot released mid-sweep (the snapshot keys it but the dict no
    longer has it) does not KeyError — the sweep snapshots items() and _release is idempotent."""
    r = _runner()
    now = time.time()
    r._running_agents = {"a": object(), "b": object()}
    r._running_agents_ts = {"a": now - 10, "b": now - 10}
    r._running_agent_tasks = {"a": _done_task(), "b": _done_task()}
    # simulate a's finally racing: remove it from under the sweep by patching _release
    orig_release = r._release_running_agent_state
    def _racing_release(key, *a, **k):
        # the turn's own finally already popped 'b' before the sweep reached it
        r._running_agents.pop("b", None)
        r._running_agent_tasks.pop("b", None)
        return orig_release(key, *a, **k)
    r._release_running_agent_state = _racing_release
    n = r._reap_dead_running_agents(now=now)   # must not raise
    assert "a" not in r._running_agents
    assert "b" not in r._running_agents


def test_reaper_loop_runs_a_sweep(monkeypatch):
    """Integration: the async loop actually invokes the sweep on its interval and stamps
    the heartbeat, then stops when _running flips False."""
    import asyncio

    r = _runner()
    r._running = True
    r._background_tasks = set()
    monkeypatch.setattr(GatewayRunner, "_reap_interval_secs", staticmethod(lambda: 0.01))
    sweeps = {"n": 0}
    def _count_sweep(now=None):
        sweeps["n"] += 1
        if sweeps["n"] >= 2:
            r._running = False   # let the loop exit
        return 0
    r._reap_dead_running_agents = _count_sweep

    async def _drive():
        task = asyncio.create_task(r._reap_dead_running_agents_loop())
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(_drive())
    assert sweeps["n"] >= 2
    assert hasattr(r, "_reaper_last_sweep_ts")   # heartbeat stamped


def test_reaper_loop_survives_a_sweep_exception(monkeypatch):
    """AC-6/INV-5: a sweep that RAISES does not kill the loop — it logs and continues; a
    dead reaper would silently re-introduce the active_agents-inflation bug."""
    import asyncio

    r = _runner()
    r._running = True
    monkeypatch.setattr(GatewayRunner, "_reap_interval_secs", staticmethod(lambda: 0.01))
    calls = {"n": 0}
    def _boom_then_stop(now=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("sweep boom")   # first sweep explodes
        r._running = False                      # second sweep proves the loop survived
        return 0
    r._reap_dead_running_agents = _boom_then_stop
    r._stamp_reaper_heartbeat = lambda: None

    async def _drive():
        await asyncio.wait_for(r._reap_dead_running_agents_loop(), timeout=2.0)

    asyncio.run(_drive())
    assert calls["n"] >= 2   # the loop kept going past the exception
