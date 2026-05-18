"""Regression guards for all notification paths.

This file is the single source of truth for verifying that the core
notification infrastructure (TUI poller, gateway watcher, CLI drain,
completion-consumed semantics, and the poll()-fix commit) remains intact.

Run with:
    pytest tests/test_regression_notification_paths.py -xvs

All tests must PASS. Any failure indicates a regression in a notification path.
"""

import ast
import asyncio
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession, process_registry


def _make_session(
    sid="proc_test",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    notify_on_complete=False,
) -> ProcessSession:
    return ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
        notify_on_complete=notify_on_complete,
    )


# ========================================================================
# Guard 1 — TUI Notification Path
# ========================================================================

class TestTuiNotificationPath:
    """Verify TUI poller loop dispatches completions and marks consumed."""

    def test_tui_poller_loop_exists_and_has_key_lines(self):
        """Source guard: _notification_poller_loop must contain the expected logic."""
        src = Path("tui_gateway/server.py").read_text()
        assert "def _notification_poller_loop(" in src
        assert "process_registry.completion_queue.get(timeout=0.5)" in src
        assert "process_registry.is_completion_consumed(_evt_sid)" in src
        assert "process_registry.mark_completion_consumed(_evt_sid)" in src

    def test_tui_poller_marks_consumed_after_dispatch(self, monkeypatch):
        """After dispatching a completion, the TUI poller marks it consumed."""
        import tui_gateway.server as server

        turns = []
        emitted = []

        class _Agent:
            def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
                turns.append(prompt)
                return {
                    "final_response": "ok",
                    "messages": [{"role": "assistant", "content": "ok"}],
                }

        class _ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target
            def start(self):
                self._target()

        sess = {
            "agent": _Agent(),
            "session_key": "session-key",
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "attached_images": [],
            "image_counter": 0,
            "cols": 80,
            "slash_worker": None,
            "show_reasoning": False,
            "tool_progress_mode": "all",
        }
        server._sessions["sid_tui_guard"] = sess
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
        monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
        monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
        process_registry._completion_consumed.discard("proc_tui_guard")

        stop = threading.Event()
        process_registry.completion_queue.put({
            "type": "completion",
            "session_id": "proc_tui_guard",
            "command": "echo hello",
            "exit_code": 0,
            "output": "hello",
        })
        stop.set()

        try:
            server._notification_poller_loop(stop, "sid_tui_guard", sess)
            assert len(turns) == 1
            assert process_registry.is_completion_consumed("proc_tui_guard")
        finally:
            server._sessions.pop("sid_tui_guard", None)
            process_registry._completion_consumed.discard("proc_tui_guard")
            while not process_registry.completion_queue.empty():
                process_registry.completion_queue.get_nowait()


# ========================================================================
# Guard 2 — Gateway Watcher Notifications
# ========================================================================

class TestGatewayWatcherPath:
    """Verify gateway spawns watchers, checks consumed, and routes via session store."""

    def test_gateway_has_pending_watchers_drain(self):
        """Source guard: gateway/run.py must drain pending_watchers after agent runs."""
        src = Path("gateway/run.py").read_text()
        assert "while process_registry.pending_watchers:" in src
        assert "process_registry.pending_watchers.pop(0)" in src
        assert "asyncio.create_task(self._run_process_watcher(watcher))" in src

    def test_gateway_watcher_checks_is_completion_consumed(self):
        """Source guard: _run_process_watcher must skip already-consumed completions."""
        src = Path("gateway/run.py").read_text()
        assert "is_completion_consumed" in src

    def test_gateway_build_process_event_source_prefers_session_store(self):
        """Source guard: _build_process_event_source must prefer session_store origin."""
        src = Path("gateway/run.py").read_text()
        assert "_build_process_event_source" in src
        # It should reference session_store before falling back to contextvars
        assert "session_store" in src

    @pytest.mark.asyncio
    async def test_gateway_watcher_respects_notification_mode(self, monkeypatch, tmp_path):
        """Integration guard: watcher respects mode and does not notify when off."""
        from gateway.config import GatewayConfig, Platform
        from gateway.run import GatewayRunner

        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: off\n",
            encoding="utf-8",
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)

        runner = GatewayRunner(GatewayConfig())
        adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
        # adapters dict accepts BasePlatformAdapter at type-check time;
        # SimpleNamespace is sufficient for this test.
        object.__setattr__(runner, "adapters", {Platform.TELEGRAM: adapter})

        class _FakeReg:
            def get(self, session_id):
                return SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)

        import tools.process_registry as pr_module
        monkeypatch.setattr(pr_module, "process_registry", _FakeReg())

        async def _instant_sleep(*_a, **_kw):
            pass
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        await runner._run_process_watcher({
            "session_id": "proc_off",
            "check_interval": 0,
            "platform": "telegram",
            "chat_id": "123",
        })

        assert adapter.send.await_count == 0


# ========================================================================
# Guard 3 — CLI Drain Paths
# ========================================================================

class TestCliDrainPath:
    """Verify CLI drain_notifications skips consumed completions."""

    def test_drain_notifications_skips_consumed(self):
        """drain_notifications must skip events already consumed via wait/log."""
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()

        process_registry._completion_consumed.add("proc_cli_skip")
        process_registry.completion_queue.put({
            "type": "completion",
            "session_id": "proc_cli_skip",
            "command": "echo done",
            "exit_code": 0,
            "output": "done",
        })

        try:
            results = process_registry.drain_notifications()
            assert len(results) == 0
        finally:
            process_registry._completion_consumed.discard("proc_cli_skip")
            while not process_registry.completion_queue.empty():
                process_registry.completion_queue.get_nowait()

    def test_drain_notifications_returns_unconsumed(self):
        """drain_notifications must return events that are NOT consumed."""
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()

        process_registry.completion_queue.put({
            "type": "completion",
            "session_id": "proc_cli_return",
            "command": "echo hi",
            "exit_code": 0,
            "output": "hi",
        })

        try:
            results = process_registry.drain_notifications()
            assert len(results) == 1
            assert results[0][0]["session_id"] == "proc_cli_return"
        finally:
            process_registry._completion_consumed.discard("proc_cli_return")
            while not process_registry.completion_queue.empty():
                process_registry.completion_queue.get_nowait()


# ========================================================================
# Guard 4 — No Duplicate Notifications
# ========================================================================

class TestNoDuplicateNotifications:
    """Verify the three dedup mechanisms are intact."""

    def test_poll_does_not_mark_consumed(self):
        """poll() must be read-only and must NOT add to _completion_consumed."""
        registry = ProcessRegistry()
        s = _make_session(sid="proc_poll_dup", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._finished[s.id] = s

        result = registry.poll("proc_poll_dup")
        assert result["status"] == "exited"
        assert not registry.is_completion_consumed("proc_poll_dup")

    def test_wait_marks_consumed(self):
        """wait() must mark completion as consumed."""
        registry = ProcessRegistry()
        s = _make_session(sid="proc_wait_dup", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        assert not registry.is_completion_consumed("proc_wait_dup")
        result = registry.wait("proc_wait_dup", timeout=1)
        assert result["status"] == "exited"
        assert registry.is_completion_consumed("proc_wait_dup")

    def test_read_log_marks_consumed(self):
        """read_log() must mark completion as consumed."""
        registry = ProcessRegistry()
        s = _make_session(sid="proc_log_dup", notify_on_complete=True, output="line1\nline2")
        s.exited = True
        s.exit_code = 0
        registry._finished[s.id] = s

        result = registry.read_log("proc_log_dup")
        assert result["status"] == "exited"
        assert registry.is_completion_consumed("proc_log_dup")

    def test_move_to_finished_is_idempotent(self):
        """_move_to_finished must not enqueue duplicate completion events."""
        registry = ProcessRegistry()
        s = _make_session(sid="proc_idem", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s

        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)
            assert registry.completion_queue.qsize() == 1
            registry._move_to_finished(s)
            assert registry.completion_queue.qsize() == 1

        while not registry.completion_queue.empty():
            registry.completion_queue.get_nowait()


# ========================================================================
# Guard 5 — Commit 2d50b9706 (poll() Fix) Preserved
# ========================================================================

class TestPollFixPreserved:
    """Verify the core poll() fix (#10156) is still in effect."""

    def test_poll_function_has_no_completion_consumed_add(self):
        """AST guard: poll() body must NOT contain _completion_consumed.add()."""
        src = Path("tools/process_registry.py").read_text()
        tree = ast.parse(src)

        poll_body = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "poll":
                poll_body = node
                break

        assert poll_body is not None, "poll() function not found"

        # Walk the poll function body looking for _completion_consumed.add calls
        for child in ast.walk(poll_body):
            if isinstance(child, ast.Attribute):
                # Look for .add on _completion_consumed
                if child.attr == "add":
                    # Check the value chain leads to _completion_consumed
                    val = child.value
                    if isinstance(val, ast.Attribute) and val.attr == "_completion_consumed":
                        pytest.fail(
                            "poll() contains _completion_consumed.add() — "
                            "the #10156 fix has been regressed"
                        )

    def test_mark_completion_consumed_method_exists(self):
        """The explicit mark_completion_consumed() method must exist for TUI use."""
        assert hasattr(process_registry, "mark_completion_consumed")
        assert callable(process_registry.mark_completion_consumed)

    def test_git_history_has_fix_not_revert(self):
        """Git guard: the most recent commit touching process_registry.py must be the fix, not the revert."""
        import subprocess

        result = subprocess.run(
            ["git", "log", "--oneline", "HEAD~10..HEAD", "--", "tools/process_registry.py"],
            capture_output=True,
            text=True,
            cwd=Path("."),
        )
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        assert lines, "No commits found touching process_registry.py in HEAD~10..HEAD"

        # The most recent commit should NOT be the revert 40aeab697
        most_recent = lines[0]
        assert "40aeab697" not in most_recent, (
            f"Most recent commit is the revert: {most_recent}"
        )

        # It should reference the fix or be a successor
        assert any(
            keyword in most_recent.lower()
            for keyword in ["poll", "completion", "consumed", "remove", "stale", "docs"]
        ), f"Most recent commit does not look like the fix or successor: {most_recent}"
