"""Orphan-task sweep must honor A2A_REPLY_TIMEOUT (issue #106972).

Raising A2A_REPLY_TIMEOUT used to only extend the HTTP reply wait; the
watchdog still swept with a hardcoded 300s, so long tasks were marked
TASK_STATE_FAILED and the real reply was silently discarded.
"""

from __future__ import annotations

import inspect
import time

from plugins.platforms.a2a import adapter, protocol


def _age_nonterminal_task(store: protocol.TaskStore, task_id: str, age_seconds: float) -> None:
    rec = store.get(task_id)
    assert rec is not None
    store._tasks[task_id]["created_at"] = time.time() - age_seconds


def test_orphan_timeout_follows_a2a_reply_timeout(monkeypatch):
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "600")
    assert adapter._orphan_timeout() == adapter._reply_timeout()
    assert adapter._orphan_timeout() == 600.0


def test_orphan_timeout_default_is_300(monkeypatch):
    monkeypatch.delenv("A2A_REPLY_TIMEOUT", raising=False)
    assert adapter._orphan_timeout() == adapter._reply_timeout()
    assert adapter._orphan_timeout() == 300.0


def test_orphan_timeout_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "not-a-number")
    assert adapter._orphan_timeout() == adapter._reply_timeout()
    assert adapter._orphan_timeout() == 300.0


def test_orphan_timeout_clamps_below_one(monkeypatch):
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "0")
    assert adapter._orphan_timeout() == adapter._reply_timeout()
    assert adapter._orphan_timeout() == 1.0


def test_orphan_sweep_spares_task_younger_than_raised_timeout(monkeypatch):
    """age=400s must survive when A2A_REPLY_TIMEOUT=600 (hardcoded 300 would fail it)."""
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "600")
    store = protocol.TaskStore()
    store.create("t-long", "ctx", "peer")
    _age_nonterminal_task(store, "t-long", 400)
    failed = store.fail_orphans(int(adapter._orphan_timeout()))
    assert failed == []
    assert store.get("t-long")["state"] not in protocol.TERMINAL_STATES


def test_orphan_sweep_still_fails_when_timeout_is_default(monkeypatch):
    """CONTROL: sweep still works — age=400s fails under the default 300s timeout."""
    monkeypatch.delenv("A2A_REPLY_TIMEOUT", raising=False)
    store = protocol.TaskStore()
    store.create("t-stale", "ctx", "peer")
    _age_nonterminal_task(store, "t-stale", 400)
    failed = store.fail_orphans(int(adapter._orphan_timeout()))
    assert failed == ["t-stale"]
    assert store.get("t-stale")["state"] == protocol.STATE_FAILED


def test_watchdog_loop_uses_orphan_timeout_helper():
    """_watchdog_loop must pass the env-derived timeout, not the old 300s constant."""
    src = inspect.getsource(adapter.A2AAdapter._watchdog_loop)
    assert "_orphan_timeout()" in src
    assert "fail_orphans(_ORPHAN_TIMEOUT)" not in src
