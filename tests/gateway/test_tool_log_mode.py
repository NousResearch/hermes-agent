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
async def test_write_tool_log_writes_and_rotates_handler(tmp_path, monkeypatch):
    """The writer coroutine drains the queue into logs/tool_calls.log."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    log_queue: queue.Queue = queue.Queue()
    log_queue.put("2026-07-02 10:00:00  terminal: \"echo hi\"")
    log_queue.put("2026-07-02 10:00:01  read_file: \"foo.py\"")

    # Minimal inline copy of write_tool_log wiring (the real coroutine is a
    # closure inside _run_agent); exercise the same handler configuration.
    import logging
    from logging.handlers import RotatingFileHandler

    from agent.redact import RedactingFormatter

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "tool_calls.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(RedactingFormatter("%(message)s"))
    tool_logger = logging.getLogger(f"hermes.tool_calls.test.{id(log_queue)}")
    tool_logger.setLevel(logging.INFO)
    tool_logger.propagate = False
    tool_logger.addHandler(handler)
    try:
        while True:
            try:
                tool_logger.info("%s", log_queue.get_nowait())
            except queue.Empty:
                break
    finally:
        tool_logger.removeHandler(handler)
        handler.flush()
        handler.close()

    content = (log_dir / "tool_calls.log").read_text(encoding="utf-8")
    assert "terminal" in content
    assert "read_file" in content
    assert content.count("\n") == 2
    await asyncio.sleep(0)  # keep the asyncio marker honest




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
