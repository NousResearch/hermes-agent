"""Tests for the `log` tool_progress mode (salvage of #3459 / #3458).

`display.tool_progress: log` keeps the chat silent and appends tool-call
lines to ~/.hermes/logs/tool_calls.log via write_tool_log's rotating handler.
These tests exercise the mode's building blocks without spinning up a full
gateway run: the callback log-branch semantics and the writer coroutine.
"""

import asyncio
import queue
from datetime import datetime

import pytest


def _log_branch(log_queue, progress_queue, event_type, tool_name, preview=None):
    """Replica of the log-mode branch in gateway/run.py progress_callback."""
    if log_queue is not None:
        if event_type == "tool.started" and tool_name and tool_name != "_thinking":
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            preview_str = f' "{preview}"' if preview else ""
            log_queue.put(f"{ts}  {tool_name}:{preview_str}".rstrip())
        if not progress_queue:
            return "returned"
    return "fell-through"


class TestLogBranchSemantics:
    def test_tool_started_enqueued(self):
        q = queue.Queue()
        assert _log_branch(q, None, "tool.started", "terminal", "ls -la") == "returned"
        line = q.get_nowait()
        assert "terminal" in line and "ls -la" in line


    def test_thinking_not_enqueued(self):
        q = queue.Queue()
        _log_branch(q, None, "tool.started", "_thinking", "pondering")
        assert q.empty()


@pytest.mark.asyncio
async def test_write_tool_log_shares_one_logger_across_turns(tmp_path, monkeypatch):
    """Two turns with distinct queues register no new Logger (loggerDict is process-lifetime) and
    every line lands in tool_calls.log exactly once through the shared handler."""
    import logging

    import gateway.run as gateway_run
    from gateway.run_turn import GatewayTurnMixin

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    shared = logging.getLogger("hermes.tool_calls")
    for h in list(shared.handlers):
        shared.removeHandler(h)
        h.close()
    before = set(logging.Logger.manager.loggerDict)
    queues = []
    try:
        for i in range(2):
            q: queue.Queue = queue.Queue()
            queues.append(q)  # kept alive: distinct id()s, the shape the per-turn name leaked on
            q.put(f"2026-09-18 10:00:0{i}  terminal: \"echo {i}\"")
            task = asyncio.ensure_future(GatewayTurnMixin._run_agent_write_tool_log(None, q))
            await asyncio.sleep(0.05)
            task.cancel()
            await task  # the writer swallows the cancel after draining
        assert set(logging.Logger.manager.loggerDict) - before <= {"hermes.tool_calls"}
        lines = (tmp_path / "logs" / "tool_calls.log").read_text(encoding="utf-8").splitlines()
        assert [l.split()[-1] for l in lines] == ['0"', '1"']
    finally:
        for h in list(shared.handlers):
            shared.removeHandler(h)
            h.close()
