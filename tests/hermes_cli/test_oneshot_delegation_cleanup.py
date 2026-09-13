"""Offline native workers and SQLite exercise the actual one-shot cleanup hook."""
import threading
import time

from hermes_cli import main as cli_main
from tools import async_delegation as ad
from tools.process_registry import process_registry


def test_cleanup_waits_for_interrupt_and_terminalizes_survivor(monkeypatch):
    ad._reset_for_tests()
    interrupted = threading.Event()
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    def prompt():
        started[0].set()
        interrupted.wait(10)
        return {"status": "interrupted", "summary": None, "error": "cancelled"}
    def survivor():
        started[1].set()
        release.wait(10)
        return {"status": "completed", "summary": "late"}
    handles = []
    for runner, signal in [(prompt, interrupted.set), (survivor, lambda: None)]:
        h = ad.dispatch_async_delegation(goal="offline", context=None, toolsets=None,
            role="leaf", model="fixture", session_key="", runner=runner, interrupt_fn=signal)
        assert h["status"] == "dispatched"
        handles.append(h["delegation_id"])
    assert all(e.wait(5) for e in started)
    step = next(s for s in cli_main._ONESHOT_CLEANUPS if s[0] == "tools.async_delegation")
    if step[1] == "finalize_for_oneshot_shutdown":
        step = (step[0], step[1], {"grace_seconds": 0.2}, step[3])
    monkeypatch.setattr(cli_main, "_ONESHOT_CLEANUPS", (step,))
    monkeypatch.setattr(cli_main, "_oneshot_cleanup_done", False)
    try:
        before = time.monotonic()
        cli_main._cleanup_oneshot_runtime()
        assert time.monotonic() - before < 3
        rows = [ad.get_durable_delegation(i) for i in handles]
        assert [r["state"] for r in rows] == ["interrupted", "unknown"]
        assert "outcome unknown" in rows[1]["result"]["error"]
        events = [process_registry.completion_queue.get(timeout=3) for _ in handles]
        assert {e["delegation_id"]: e["status"] for e in events} == dict(zip(handles, ["interrupted", "unknown"]))
        release.set()
        ad._executor.shutdown(wait=True)
        assert ad.get_durable_delegation(handles[1])["state"] == "unknown"
        assert process_registry.completion_queue.empty()
    finally:
        interrupted.set()
        release.set()
        ad._executor.shutdown(wait=True)
        ad._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
