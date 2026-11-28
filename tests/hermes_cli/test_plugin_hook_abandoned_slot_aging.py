"""Aging out wedged abandoned-worker slots (#138644).

A pre_tool_call shell hook whose timeout kill did not unwind the worker's ``communicate()``
left its abandoned slot held forever; at the ``_HOOK_MAX_ABANDONED_WORKERS`` cap every later
tool call then failed closed until a restart. These tests pin the containment: young slots
keep the gate closed, an aged slot is evicted so one replacement may start.
"""

import logging
import threading
import time

import hermes_cli.plugins_dispatch as dispatch
from hermes_cli.plugins import PluginManager, _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE


def _wedge_every_worker():
    hold = threading.Event()
    starts = []

    def guard(**_kwargs):
        starts.append(1)
        hold.wait(timeout=10.0)  # every worker wedges for good — the bug's precondition
        return "late"

    return hold, starts, guard


def test_young_slots_at_the_cap_keep_failing_closed(monkeypatch, caplog):
    """Below the max age the pre-cap behavior is unchanged: no new worker, fail-closed skip."""
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
    )
    hold, starts, guard = _wedge_every_worker()

    mgr = PluginManager()
    mgr._hook_timeout_suppression_seconds = 0.0  # isolate the gate from suppression
    mgr._hooks["pre_tool_call"] = [guard]

    blocked = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]
    for i in range(dispatch._HOOK_MAX_ABANDONED_WORKERS):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id=f"call-{i}") == blocked
    assert len(starts) == dispatch._HOOK_MAX_ABANDONED_WORKERS

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-young") == blocked
    assert len(starts) == dispatch._HOOK_MAX_ABANDONED_WORKERS
    assert "abandoned worker(s) still running" in caplog.text
    hold.set()


def test_aged_slot_is_evicted_and_admits_one_replacement(monkeypatch, caplog):
    """A slot older than the max age belongs to a wedged thread: evict it, start one
    replacement (fail-closed skips resume if the replacement times out too)."""
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
    )
    hold, starts, guard = _wedge_every_worker()

    mgr = PluginManager()
    mgr._hook_timeout_suppression_seconds = 0.0  # isolate the gate from suppression
    mgr._hooks["pre_tool_call"] = [guard]

    blocked = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]
    for i in range(dispatch._HOOK_MAX_ABANDONED_WORKERS):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id=f"call-{i}") == blocked

    slots = mgr._hook_abandoned[("pre_tool_call", id(guard))]
    oldest = min(slots, key=slots.__getitem__)
    slots[oldest] = time.monotonic() - dispatch._HOOK_ABANDONED_SLOT_MAX_AGE_SECONDS - 1.0

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-aged") == blocked
    assert len(starts) == dispatch._HOOK_MAX_ABANDONED_WORKERS + 1
    assert "evicting an abandoned worker slot" in caplog.text
    # The replacement wedged too (its own timeout re-fills the evicted slot) — bounded, and
    # the next aged slot would admit the next replacement instead of wedging until restart.
    assert len(mgr._hook_abandoned[("pre_tool_call", id(guard))]) == dispatch._HOOK_MAX_ABANDONED_WORKERS
    hold.set()


def test_suppression_window_gates_replacement_after_an_aged_out_slot(monkeypatch, caplog):
    """Suppression still rate-limits replacement starts: even when the oldest slot has aged
    out and the cap would admit a replacement, a live suppression window skips first — no
    new worker until the window has passed."""
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
    )
    hold, starts, guard = _wedge_every_worker()

    mgr = PluginManager()
    mgr._hook_timeout_suppression_seconds = 0.0  # fill the cap without inter-call waits
    mgr._hooks["pre_tool_call"] = [guard]

    blocked = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]
    for i in range(dispatch._HOOK_MAX_ABANDONED_WORKERS):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id=f"call-{i}") == blocked

    slots = mgr._hook_abandoned[("pre_tool_call", id(guard))]
    oldest = min(slots, key=slots.__getitem__)
    slots[oldest] = time.monotonic() - dispatch._HOOK_ABANDONED_SLOT_MAX_AGE_SECONDS - 1.0
    # A live suppression window (as the last timeout would have set) skips before the
    # abandoned-slot logic runs: no eviction, no replacement worker.
    with mgr._hook_timeout_lock:
        mgr._hook_timeout_suppressed_until[("pre_tool_call", id(guard))] = time.monotonic() + 999.0

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-suppressed") == blocked
    assert len(starts) == dispatch._HOOK_MAX_ABANDONED_WORKERS
    assert "skipped after previous timeout" in caplog.text
    assert "evicting an abandoned worker slot" not in caplog.text
    assert len(mgr._hook_abandoned[("pre_tool_call", id(guard))]) == dispatch._HOOK_MAX_ABANDONED_WORKERS

    # Once the window has passed, the same aged slot is evicted and one replacement starts.
    with mgr._hook_timeout_lock:
        mgr._hook_timeout_suppressed_until[("pre_tool_call", id(guard))] = time.monotonic() - 1.0
    assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-admitted") == blocked
    assert len(starts) == dispatch._HOOK_MAX_ABANDONED_WORKERS + 1
    hold.set()


def test_worker_finishing_after_its_slot_was_aged_out_drains_cleanly(monkeypatch):
    """The release path for a slot that aged out before its worker finished:
    ``abandoned.pop(gate_key, None)`` passes silently and must not block the other slots —
    the bookkeeping empties once every wedged worker finally unwinds."""
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
    )
    hold, starts, guard = _wedge_every_worker()

    mgr = PluginManager()
    mgr._hook_timeout_suppression_seconds = 0.0  # isolate the gate from suppression
    mgr._hooks["pre_tool_call"] = [guard]

    blocked = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]
    for i in range(dispatch._HOOK_MAX_ABANDONED_WORKERS):
        assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id=f"call-{i}") == blocked

    slots = mgr._hook_abandoned[("pre_tool_call", id(guard))]
    oldest = min(slots, key=slots.__getitem__)
    slots[oldest] = time.monotonic() - dispatch._HOOK_ABANDONED_SLOT_MAX_AGE_SECONDS - 1.0
    assert mgr.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-aged") == blocked
    assert len(starts) == dispatch._HOOK_MAX_ABANDONED_WORKERS + 1
    assert oldest not in mgr._hook_abandoned[("pre_tool_call", id(guard))]

    hold.set()  # every wedged worker finally unwinds, including the aged-out one
    deadline = time.monotonic() + 5.0
    drained = False
    while time.monotonic() < deadline:
        with mgr._hook_timeout_lock:
            drained = (("pre_tool_call", id(guard)) not in mgr._hook_abandoned
                       and not mgr._hook_running_callbacks)
        if drained:
            break
        time.sleep(0.02)
    assert drained, (
        f"bookkeeping did not drain: abandoned={mgr._hook_abandoned} "
        f"running={mgr._hook_running_callbacks}")
