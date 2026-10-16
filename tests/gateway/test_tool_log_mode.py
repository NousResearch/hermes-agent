"""Tests for the `log` tool_progress mode (salvage of #3459 / #3458).

`display.tool_progress: log` keeps the chat silent and appends tool-call
lines to ~/.hermes/logs/tool_calls.log via write_tool_log's rotating handler.
These tests exercise the mode's building blocks without spinning up a full
gateway run: the writer coroutine.
"""

import asyncio
import queue

import pytest


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
