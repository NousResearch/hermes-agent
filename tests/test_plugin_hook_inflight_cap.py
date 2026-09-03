"""Concurrent invocations of one bounded hook must not be mistaken for a hang.

Regression: the in-flight latch was a single slot per (hook, callback), so
parallel tool calls (executor runs up to 8 workers) or concurrent api_server
runs overlapping on a healthy ~50 ms ``pre_tool_call`` shell hook produced
fail-closed "pre_tool_call plugin callback timed out or is still running"
blocks. Observed fleet-wide: 285 false blocks across nine profiles in five
days; one voice backend run spent minutes retrying blocked kanban_list calls.
"""
from __future__ import annotations

import threading
import time

from hermes_cli import plugins as P


def _manager_with(cb, hook="pre_tool_call"):
    pm = P.PluginManager()
    pm._hooks.setdefault(hook, []).append(cb)
    return pm


def test_concurrent_pre_tool_call_invocations_all_run():
    def gate(**kw):
        time.sleep(0.2)
        return None

    pm = _manager_with(gate)
    results = {}

    def call(i):
        results[i] = pm.invoke_hook("pre_tool_call", tool_name="kanban_list", args={})

    threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results[i] == [] for i in range(4)), results
    assert pm._hook_running_callbacks == {}


def test_timed_out_hook_is_suppressed_without_spawning_more_threads(monkeypatch):
    release = threading.Event()

    def hung(**kw):
        release.wait(5)
        return None

    pm = _manager_with(hung)
    monkeypatch.setattr(P, "_resolve_hook_callback_timeout", lambda: 0.05)
    first = pm.invoke_hook("pre_tool_call", tool_name="x", args={})
    assert first == [P._pre_tool_call_timeout_block()]
    key = next(iter(pm._hook_running_callbacks))
    assert len(pm._hook_running_callbacks[key]) == 1
    # Suppression window is active; the next fire is skipped without a new thread.
    second = pm.invoke_hook("pre_tool_call", tool_name="x", args={})
    assert second == [P._pre_tool_call_timeout_block()]
    assert len(pm._hook_running_callbacks[key]) == 1
    release.set()
    time.sleep(0.2)
    assert pm._hook_running_callbacks == {}


def test_saturated_callback_is_skipped_fail_closed(monkeypatch):
    release = threading.Event()

    def slow(**kw):
        release.wait(5)
        return None

    pm = _manager_with(slow)
    monkeypatch.setattr(P, "_HOOK_MAX_INFLIGHT_PER_CALLBACK", 1)
    monkeypatch.setattr(P, "_resolve_hook_callback_timeout", lambda: 5.0)
    results = {}

    def call(i):
        results[i] = pm.invoke_hook("pre_tool_call", tool_name="x", args={})

    t1 = threading.Thread(target=call, args=(1,))
    t1.start()
    time.sleep(0.1)
    results[2] = pm.invoke_hook("pre_tool_call", tool_name="x", args={})
    assert results[2] == [P._pre_tool_call_timeout_block()]
    release.set()
    t1.join()
    assert results[1] == []
