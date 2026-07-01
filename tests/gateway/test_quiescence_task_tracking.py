"""Busy-gateway-quiescence Phase 1 — per-session turn-task tracking (run.py).

Spec: ~/.hermes/plans/2026-06-30_safe-restart-busy-gateway-quiescence-SPEC.md (v0.4)
Pinned fork HEAD: 0e21a7b93479592a3b666c4076662f732070c0ea

Phase 1 tracks the asyncio Task running each session's current turn in
`_running_agent_tasks`, so the Phase-2 reaper can evict ONLY genuinely-dead slots
(task.done()) and NEVER a live long-running turn. This file covers the release
chokepoint + the persist-keys wiring; the capture-at-slot-set no-await invariant is
guarded by a code comment (RC-1) and exercised end-to-end in the live smoke.
"""
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
    return r


def test_release_clears_the_task_handle_with_the_slot():
    """Task 1.3: _release_running_agent_state pops the task handle alongside the slot,
    so the handle never outlives the slot (no _running_agent_tasks leak)."""
    r = _runner()
    sk = "agent:main:discord:group:111:222"
    r._running_agents[sk] = object()         # a real (non-sentinel) agent
    r._running_agents_ts[sk] = 1.0
    r._running_agent_tasks[sk] = MagicMock(name="turn_task")

    r._release_running_agent_state(sk)

    assert sk not in r._running_agents
    assert sk not in r._running_agents_ts
    assert sk not in r._running_agent_tasks      # handle cleared with the slot


def test_release_is_idempotent_double_release_safe():
    """AC-10: a reaper eviction racing the turn's own finally double-releases harmlessly
    (pop-on-absent), no KeyError, no double-decrement of anything."""
    r = _runner()
    sk = "agent:main:discord:group:111:222"
    r._running_agents[sk] = object()
    r._running_agent_tasks[sk] = MagicMock()
    r._release_running_agent_state(sk)
    # second release (the racing caller) must not raise
    r._release_running_agent_state(sk)
    assert sk not in r._running_agents
    assert sk not in r._running_agent_tasks


def test_persist_active_agents_writes_keys(monkeypatch):
    """Task 1.4: _persist_active_agents passes the live (non-sentinel) session keys to
    write_runtime_status as active_agent_keys, alongside the count."""
    import gateway.run as run_mod

    r = _runner()
    r._running_agents = {
        "agent:main:discord:group:1:2": object(),       # real agent → counted + keyed
        "agent:main:discord:group:3:4": _AGENT_PENDING_SENTINEL,  # sentinel → excluded from keys
    }
    captured = {}

    def _fake_write(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_mod, "write_runtime_status", _fake_write, raising=False)
    # _persist_active_agents imports write_runtime_status locally; patch the source module
    import gateway.status as status_mod
    monkeypatch.setattr(status_mod, "write_runtime_status", _fake_write, raising=False)

    r._persist_active_agents()

    assert "active_agent_keys" in captured
    # the pending sentinel is excluded; only the real running session is a key
    assert captured["active_agent_keys"] == ["agent:main:discord:group:1:2"]


def test_snapshot_excludes_pending_sentinel():
    """_snapshot_running_agents (the source of active_agent_keys) excludes the sentinel,
    so a session in its async setup window is not yet advertised as 'active'."""
    r = _runner()
    r._running_agents = {
        "real": object(),
        "pending": _AGENT_PENDING_SENTINEL,
    }
    snap = r._snapshot_running_agents()
    assert "real" in snap
    assert "pending" not in snap
