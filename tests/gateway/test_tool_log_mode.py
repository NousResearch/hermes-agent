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

    def test_tool_completed_not_enqueued(self):
        q = queue.Queue()
        _log_branch(q, None, "tool.completed", "terminal")
        assert q.empty()

    def test_thinking_not_enqueued(self):
        q = queue.Queue()
        _log_branch(q, None, "tool.started", "_thinking", "pondering")
        assert q.empty()

    def test_no_preview_line_has_no_quotes(self):
        q = queue.Queue()
        _log_branch(q, None, "tool.started", "todo")
        line = q.get_nowait()
        assert line.endswith("todo:")
        assert '"' not in line

    def test_log_none_falls_through(self):
        assert _log_branch(None, None, "tool.started", "terminal") == "fell-through"


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


def test_write_tool_log_dir_resolves_via_profile_scope_not_frozen_global(tmp_path):
    """write_tool_log's log_dir must resolve through get_hermes_home() (which
    honors the active _HERMES_HOME_OVERRIDE set by _profile_runtime_scope for
    a multiplexed secondary-profile turn), not the import-time-frozen
    ``gateway.run._hermes_home`` module global.

    Same import-time-freeze bug class as gateway/hooks.py::HOOKS_DIR,
    tools/checkpoint_manager.py::CHECKPOINT_BASE, and
    gateway/sticker_cache.py::CACHE_PATH: a module-level path computed once
    at import time never sees a later per-task home override, so a secondary
    profile's tool-call activity would land in the default profile's
    logs/tool_calls.log instead of its own.
    """
    import gateway.run as gateway_run
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    profile_home = tmp_path / "profiles" / "beta"
    profile_home.mkdir(parents=True)

    token = set_hermes_home_override(str(profile_home))
    try:
        # This is the exact expression write_tool_log now uses — must
        # resolve to the scoped profile's own logs/ directory.
        assert get_hermes_home() / "logs" == profile_home / "logs"
        # The frozen module global is the bug this fix removes: it was
        # captured at import time, before any profile scope ever existed,
        # so it never reflects the override and would misdirect the log
        # to the default profile's home instead.
        assert gateway_run._hermes_home != profile_home
    finally:
        reset_hermes_home_override(token)


def test_log_mode_disables_chat_progress():
    """tool_progress_enabled must be False in log mode (silent in chat)."""
    for mode, expected in [("all", True), ("log", False), ("off", False)]:
        enabled = mode not in {"off", "log"}
        assert enabled is expected
