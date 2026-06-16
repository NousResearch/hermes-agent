"""Tests for ClaudeCodeACPClient — model routing, safety, session lifecycle, handlers."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

from agent.claude_code_acp_client import (
    ClaudeCodeACPClient,
    _ensure_path_within_cwd,
    _jsonrpc_error,
    _walk_text_blocks,
    _extract_text_from_content_blocks,
    _format_messages_as_prompt,
    _render_message_content,
    _coerce_str,
    _short_preview,
    resolve_effective_timeout,
    ToolCallRecord,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_client(**kwargs) -> ClaudeCodeACPClient:
    """Create a ClaudeCodeACPClient with sensible test defaults."""
    defaults = dict(
        acp_command="/usr/bin/false",  # never actually spawn
        acp_args=["--stdio"],
        acp_cwd="/tmp",
    )
    defaults.update(kwargs)
    return ClaudeCodeACPClient(**defaults)


class _FakeProcess:
    """Minimal subprocess stand-in with writable stdin and dummy pid."""
    def __init__(self, pid: int = 99999):
        self.stdin = io.StringIO()
        self.pid = pid
        self._poll_result = None

    def poll(self):
        return self._poll_result

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


# ===========================================================================
# 1. Model name normalization & alias resolution
# ===========================================================================
class TestNormalizeModelName(unittest.TestCase):
    def test_short_aliases(self):
        for alias in ("haiku", "sonnet", "opus", "claude", "default"):
            self.assertEqual(ClaudeCodeACPClient._normalize_model_name(alias), alias)

    def test_short_aliases_case_insensitive(self):
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("Opus"), "opus")
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("SONNET"), "sonnet")

    def test_full_names_pass_through_verbatim(self):
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("claude-opus-4-7"), "claude-opus-4-7")
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("claude-sonnet-4-6"), "claude-sonnet-4-6")
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("claude-haiku-4-5"), "claude-haiku-4-5")

    def test_context_window_suffixes_stripped(self):
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("opus[1m]"), "opus")
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("sonnet[200k]"), "sonnet")
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("claude-opus-4-7[1m]"), "claude-opus-4-7")

    def test_empty_passthrough(self):
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name(""), "")

    def test_non_claude_passthrough(self):
        self.assertEqual(ClaudeCodeACPClient._normalize_model_name("gpt-4o"), "gpt-4o")


class TestIsValidClaudeAlias(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_valid_aliases(self):
        for name in ("haiku", "sonnet", "opus", "claude", "default",
                      "Opus", "SONNET", "claude-opus-4-7", "opus[1m]"):
            self.assertTrue(self.client._is_valid_claude_alias(name), name)

    def test_invalid_aliases(self):
        for name in ("gpt-4o", "mistral", "llama", ""):
            self.assertFalse(self.client._is_valid_claude_alias(name), name)


# ===========================================================================
# 2. File-safety guards
# ===========================================================================
class TestEnsurePathWithinCwd(unittest.TestCase):
    def test_absolute_within_cwd(self):
        result = _ensure_path_within_cwd("/tmp/foo.txt", "/tmp")
        self.assertEqual(result, Path("/tmp/foo.txt").resolve())

    def test_relative_raises(self):
        with self.assertRaises(PermissionError):
            _ensure_path_within_cwd("relative.txt", "/tmp")

    def test_traversal_raises(self):
        with self.assertRaises(PermissionError):
            _ensure_path_within_cwd("/etc/passwd", "/tmp")

    def test_dot_dot_within_cwd_ok(self):
        # /tmp/sub/../file resolves inside /tmp
        result = _ensure_path_within_cwd("/tmp/sub/../file.txt", "/tmp")
        self.assertEqual(result, Path("/tmp/file.txt").resolve())

    def test_dot_dot_escape_raises(self):
        with self.assertRaises(PermissionError):
            _ensure_path_within_cwd("/tmp/../etc/shadow", "/tmp")


# ===========================================================================
# 3. JSON-RPC error helper
# ===========================================================================
class TestJsonrpcError(unittest.TestCase):
    def test_shape(self):
        err = _jsonrpc_error(42, -32601, "not found")
        self.assertEqual(err["jsonrpc"], "2.0")
        self.assertEqual(err["id"], 42)
        self.assertEqual(err["error"]["code"], -32601)
        self.assertEqual(err["error"]["message"], "not found")


# ===========================================================================
# 4. Text extraction helpers
# ===========================================================================
class TestWalkTextBlocks(unittest.TestCase):
    def test_string(self):
        self.assertEqual(list(_walk_text_blocks("hello")), ["hello"])

    def test_dict_text(self):
        self.assertEqual(list(_walk_text_blocks({"text": "hi"})), ["hi"])

    def test_nested_list(self):
        obj = [{"text": "a"}, {"content": [{"text": "b"}]}, "c"]
        self.assertEqual(list(_walk_text_blocks(obj)), ["a", "b", "c"])

    def test_none(self):
        self.assertEqual(list(_walk_text_blocks(None)), [])

    def test_skips_non_text(self):
        self.assertEqual(list(_walk_text_blocks({"type": "image", "url": "..."})), [])


class TestExtractTextFromContentBlocks(unittest.TestCase):
    def test_string_content(self):
        self.assertEqual(_extract_text_from_content_blocks("hello"), "hello")

    def test_list_of_dicts(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        self.assertEqual(_extract_text_from_content_blocks(content), "hello\nworld")

    def test_none_returns_empty(self):
        self.assertEqual(_extract_text_from_content_blocks(None), "")


class TestShortPreview(unittest.TestCase):
    def test_short_text(self):
        self.assertEqual(_short_preview("hello", limit=10), "hello")

    def test_truncated(self):
        result = _short_preview("a" * 200, limit=10)
        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(len(result), 11)  # 10 + "…"


class TestCoerceStr(unittest.TestCase):
    def test_string(self):
        self.assertEqual(_coerce_str("hello"), "hello")

    def test_int(self):
        self.assertEqual(_coerce_str(42), "42")

    def test_none(self):
        self.assertEqual(_coerce_str(None), "")


# ===========================================================================
# 5. resolve_effective_timeout
# ===========================================================================
class TestResolveEffectiveTimeout(unittest.TestCase):
    def test_numeric(self):
        self.assertEqual(resolve_effective_timeout(30, default=60), 30.0)

    def test_string_uses_default(self):
        # strings are not int/float, so they go through getattr path
        self.assertEqual(resolve_effective_timeout("45", default=60), 60.0)

    def test_none_uses_default(self):
        self.assertEqual(resolve_effective_timeout(None, default=60), 60.0)

    def test_garbage_uses_default(self):
        self.assertEqual(resolve_effective_timeout("abc", default=60), 60.0)

    def test_negative_passthrough(self):
        # -1 is a valid number, returns -1.0 (caller's responsibility)
        self.assertEqual(resolve_effective_timeout(-1, default=60), -1.0)

    def test_zero_passthrough(self):
        # 0 is a valid number, returns 0.0
        self.assertEqual(resolve_effective_timeout(0, default=60), 0.0)


# ===========================================================================
# 6. Server message dispatch (_handle_server_message)
# ===========================================================================
class TestHandleServerMessage(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()
        self.process = _FakeProcess()
        self.dispatch_ctx = {"text_parts": [], "reasoning_parts": []}

    def test_session_update_returns_true(self):
        msg = {
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": [{"text": "hi"}]}},
        }
        self.assertTrue(self.client._handle_server_message(
            msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
        ))
        self.assertIn("hi", self.dispatch_ctx["text_parts"])

    def test_permission_request_returns_allow(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}}
        self.assertTrue(self.client._handle_server_message(
            msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
        ))
        written = self.process.stdin.getvalue()
        resp = json.loads(written.strip())
        self.assertEqual(resp["result"]["outcome"]["outcome"], "allow_once")

    def test_unknown_method_returns_error(self):
        msg = {"jsonrpc": "2.0", "id": 5, "method": "unknown/method", "params": {}}
        self.assertTrue(self.client._handle_server_message(
            msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
        ))
        written = self.process.stdin.getvalue()
        resp = json.loads(written.strip())
        self.assertEqual(resp["error"]["code"], -32601)

    def test_non_method_message_returns_false(self):
        msg = {"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "s1"}}
        self.assertFalse(self.client._handle_server_message(
            msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
        ))

    def test_fs_read_within_cwd(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir="/tmp", delete=False) as f:
            f.write("test content")
            tmp_path = f.name
        try:
            msg = {"jsonrpc": "2.0", "id": 10, "method": "fs/read_text_file", "params": {"path": tmp_path}}
            self.assertTrue(self.client._handle_server_message(
                msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
            ))
            resp = json.loads(self.process.stdin.getvalue().strip())
            self.assertEqual(resp["result"]["content"], "test content")
        finally:
            os.unlink(tmp_path)

    def test_fs_read_outside_cwd_returns_error(self):
        msg = {"jsonrpc": "2.0", "id": 10, "method": "fs/read_text_file", "params": {"path": "/etc/passwd"}}
        self.client._acp_cwd = "/tmp"
        self.assertTrue(self.client._handle_server_message(
            msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
        ))
        resp = json.loads(self.process.stdin.getvalue().strip())
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_fs_write_outside_cwd_returns_error(self):
        msg = {"jsonrpc": "2.0", "id": 11, "method": "fs/write_text_file",
               "params": {"path": "/etc/evil", "content": "bad"}}
        self.client._acp_cwd = "/tmp"
        self.assertTrue(self.client._handle_server_message(
            msg, process=self.process, dispatch_ctx=self.dispatch_ctx,
        ))
        resp = json.loads(self.process.stdin.getvalue().strip())
        self.assertIn("error", resp)


# ===========================================================================
# 7. session/update routing (tool_call_start / tool_call_update)
# ===========================================================================
class TestSessionUpdateRouting(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()
        self.dispatch_ctx = {"text_parts": [], "reasoning_parts": []}

    def test_agent_message_chunk(self):
        update = {"sessionUpdate": "agent_message_chunk", "content": [{"text": "hello"}]}
        self.client._handle_session_update(update, dispatch_ctx=self.dispatch_ctx)
        self.assertEqual(self.dispatch_ctx["text_parts"], ["hello"])

    def test_agent_thought_chunk(self):
        update = {"sessionUpdate": "agent_thought_chunk", "content": [{"text": "thinking..."}]}
        self.client._handle_session_update(update, dispatch_ctx=self.dispatch_ctx)
        self.assertEqual(self.dispatch_ctx["reasoning_parts"], ["thinking..."])

    def test_tool_call_start_creates_record(self):
        update = {
            "sessionUpdate": "tool_call_start",
            "toolCallId": "tc-1",
            "title": "Read",
            "kind": "read",
            "rawInput": {"path": "/tmp/f.txt"},
        }
        self.client._handle_session_update(update, dispatch_ctx=self.dispatch_ctx)
        self.assertEqual(len(self.client._tool_trace), 1)
        rec = self.client._tool_trace[0]
        self.assertEqual(rec.tool_call_id, "tc-1")
        self.assertEqual(rec.name, "Read")
        self.assertEqual(rec.status, "in_progress")

    def test_tool_call_update_completes_record(self):
        # First create
        self.client._handle_session_update({
            "sessionUpdate": "tool_call_start",
            "toolCallId": "tc-2",
            "title": "Bash",
        }, dispatch_ctx=self.dispatch_ctx)
        # Then complete
        self.client._handle_session_update({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-2",
            "status": "completed",
            "content": [{"text": "done"}],
        }, dispatch_ctx=self.dispatch_ctx)
        rec = self.client._tool_records_by_id["tc-2"]
        self.assertEqual(rec.status, "completed")
        self.assertEqual(rec.raw_output, "done")

    def test_plan_update_does_not_crash(self):
        self.client._handle_session_update({
            "sessionUpdate": "plan",
            "entries": [{"step": 1, "description": "do stuff"}],
        }, dispatch_ctx=self.dispatch_ctx)

    def test_available_commands_update_ignored(self):
        self.client._handle_session_update({
            "sessionUpdate": "available_commands_update",
            "commands": [],
        }, dispatch_ctx=self.dispatch_ctx)


# ===========================================================================
# 8. ToolCallRecord
# ===========================================================================
class TestToolCallRecord(unittest.TestCase):
    def test_duration_none_when_no_end(self):
        rec = ToolCallRecord(tool_call_id="t1", name="Bash")
        self.assertIsNone(rec.duration)

    def test_to_dict_roundtrip(self):
        rec = ToolCallRecord(tool_call_id="t1", name="Bash", raw_input={"cmd": "ls"})
        d = rec.to_dict()
        self.assertEqual(d["id"], "t1")
        self.assertEqual(d["name"], "Bash")
        self.assertEqual(d["raw_input"], {"cmd": "ls"})


# ===========================================================================
# 9. Client initialization defaults
# ===========================================================================
class TestClientInit(unittest.TestCase):
    def test_hermes_session_id_from_env(self):
        with patch.dict(os.environ, {"HERMES_SESSION_ID": "env-sess-123"}):
            client = _make_client()
            self.assertEqual(client._hermes_session_id, "env-sess-123")

    def test_hermes_session_id_generated_when_empty(self):
        with patch.dict(os.environ, {"HERMES_SESSION_ID": ""}, clear=False):
            client = _make_client(hermes_session_id=None)
            self.assertTrue(client._hermes_session_id.startswith("hermes-"))

    def test_hermes_session_id_explicit_wins(self):
        client = _make_client(hermes_session_id="explicit-id")
        self.assertEqual(client._hermes_session_id, "explicit-id")

    def test_default_empty_state(self):
        client = _make_client()
        self.assertIsNone(client._session_proc)
        self.assertIsNone(client._session_id)
        self.assertFalse(client.is_closed)
        self.assertEqual(client._tool_trace, [])

    def test_tool_trace_and_reset(self):
        client = _make_client()
        client._tool_trace.append(ToolCallRecord(tool_call_id="x", name="test"))
        self.assertEqual(len(client.tool_trace), 1)
        client.reset_tool_trace()
        self.assertEqual(len(client.tool_trace), 0)

    def test_close_sets_flag(self):
        client = _make_client()
        self.assertFalse(client.is_closed)
        client.close()
        self.assertTrue(client.is_closed)


# ===========================================================================
# 10. _format_messages_as_prompt (smoke)
# ===========================================================================
class TestFormatMessagesAsPrompt(unittest.TestCase):
    def test_basic_conversation(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's 2+2?"},
        ]
        result = _format_messages_as_prompt(messages)
        self.assertIn("Hello", result)
        self.assertIn("2+2", result)

    def test_tool_messages_included(self):
        messages = [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt\nfile2.txt"},
        ]
        result = _format_messages_as_prompt(messages)
        self.assertIn("file1.txt", result)

    def test_empty_messages(self):
        result = _format_messages_as_prompt([])
        self.assertIsInstance(result, str)


# ===========================================================================
# 11. _render_message_content
# ===========================================================================
class TestRenderMessageContent(unittest.TestCase):
    def test_string(self):
        self.assertEqual(_render_message_content("hello"), "hello")

    def test_list_of_dicts(self):
        content = [{"type": "text", "text": "hi"}]
        self.assertEqual(_render_message_content(content), "hi")

    def test_none(self):
        self.assertEqual(_render_message_content(None), "")


# ===========================================================================
# 12. close() cleanup
# ===========================================================================
class TestCloseCleanup(unittest.TestCase):
    def test_close_terminates_active_process(self):
        client = _make_client()
        fake = _FakeProcess()
        client._active_process = fake
        client._session_id = "sess-1"
        client.close()
        self.assertTrue(client.is_closed)
        self.assertIsNone(client._session_id)
        self.assertIsNone(client._active_process)

    # ------------------------------------------------------------------
    # C1 (2e62a19ba) — reader thread join on close()
    # ------------------------------------------------------------------
    def test_close_joins_reader_threads(self):
        """C1 (2e62a19ba): close() must join the stdout/stderr reader threads
        before returning, so daemon threads can't race with closed pipes and
        trigger I/O-on-closed-file errors after close() returns."""
        import time
        client = _make_client()
        completed = []

        def reader(label):
            # Brief work, then exit — simulating a reader that drains
            # naturally after the subprocess pipes are closed.
            time.sleep(0.01)
            completed.append(label)

        t1 = threading.Thread(target=reader, args=("stdout",), daemon=True)
        t2 = threading.Thread(target=reader, args=("stderr",), daemon=True)
        t1.start()
        t2.start()
        # Need an active process so close() reaches the reader-join block
        # (otherwise it short-circuits with is_closed=True and returns).
        client._active_process = _FakeProcess()
        client._reader_threads = [t1, t2]
        client.close()
        self.assertFalse(t1.is_alive(), "stdout reader thread should be joined")
        self.assertFalse(t2.is_alive(), "stderr reader thread should be joined")
        self.assertEqual(client._reader_threads, [], "thread list should be reset")
        self.assertIn("stdout", completed)
        self.assertIn("stderr", completed)

    def test_close_idempotent_on_empty_threads(self):
        """C1: close() with no threads and no process must not raise and
        must still set is_closed=True."""
        client = _make_client()
        self.assertEqual(client._reader_threads, [])
        self.assertIsNone(client._active_process)
        client.close()  # should not raise
        self.assertTrue(client.is_closed)

    def test_close_does_not_hang_on_wedged_thread(self):
        """C1: a reader thread stuck on readline must not block close() past
        the 1.0s join timeout specified in the fix. The wedged thread is
        left alive (daemon=True) rather than forcibly killed."""
        import time
        client = _make_client()
        # Thread that blocks forever; close() must time out joining it.
        t = threading.Thread(
            target=lambda: threading.Event().wait(60), daemon=True
        )
        t.start()
        # Need an active process so close() reaches the reader-join block.
        client._active_process = _FakeProcess()
        client._reader_threads = [t]
        t0 = time.monotonic()
        client.close()
        elapsed = time.monotonic() - t0
        self.assertLess(
            elapsed, 1.5,
            f"close() took {elapsed:.2f}s; expected <1.5s (1.0s join + slack)",
        )
        self.assertTrue(
            t.is_alive(),
            "wedged thread should be left alive after timeout, not joined",
        )

    # ------------------------------------------------------------------
    # C2 (3082fad77) — close() race: lock + is_closed ordering
    # ------------------------------------------------------------------
    def test_close_resets_active_process_under_lock(self):
        """C2 (3082fad77): _active_process=None must happen INSIDE the
        _active_process_lock to avoid a race with concurrent close() callers
        who might otherwise see a stale non-None reference."""
        client = _make_client()
        held_during_reset: list[str] = []
        original_lock = client._active_process_lock

        class TrackingLock:
            """Proxy that records entry/exit while delegating to the
            real threading.Lock()."""

            def __enter__(self):
                held_during_reset.append("enter")
                return original_lock.__enter__()

            def __exit__(self, *a):
                held_during_reset.append("exit")
                return original_lock.__exit__(*a)

        client._active_process_lock = TrackingLock()  # type: ignore[assignment]
        fake = _FakeProcess()
        client._active_process = fake
        client._session_id = "sess-1"
        client.close()
        # Lock must have been held around the reset.
        self.assertIn("enter", held_during_reset)
        self.assertIn("exit", held_during_reset)
        # enter must come before exit (sanity check on the proxy).
        self.assertLess(
            held_during_reset.index("enter"),
            held_during_reset.index("exit"),
        )
        self.assertIsNone(client._active_process)
        self.assertIsNone(client._session_id)

    def test_close_is_closed_set_after_terminate(self):
        """C2: is_closed=True must come AFTER proc.terminate(), so concurrent
        callers observing is_closed=False don't skip the terminate path."""
        client = _make_client()
        fake = _FakeProcess()
        client._active_process = fake
        states_during_terminate: list[bool] = []
        original_terminate = fake.terminate

        def tracking_terminate():
            states_during_terminate.append(client.is_closed)
            original_terminate()

        fake.terminate = tracking_terminate  # type: ignore[assignment]
        client.close()
        self.assertIn(
            False, states_during_terminate,
            "is_closed must be False when proc.terminate() is called",
        )
        self.assertTrue(
            client.is_closed,
            "is_closed must be True after close() returns",
        )

    def test_concurrent_close_safe(self):
        """C2: two threads calling close() concurrently must not raise.
        Before the fix, _active_process=None was outside the lock and two
        concurrent callers could race past the proc-guard and double-terminate."""
        client = _make_client()
        client._active_process = _FakeProcess()
        client._session_id = "sess-1"
        errors: list[Exception] = []

        def do_close():
            try:
                client.close()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=do_close)
        t2 = threading.Thread(target=do_close)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        self.assertEqual(
            errors, [],
            f"concurrent close() raised: {errors!r}",
        )
        self.assertTrue(client.is_closed)
        # Final state must be clean for both session/proc state.
        self.assertIsNone(client._active_process)
        self.assertIsNone(client._session_id)


# ===========================================================================
# 13. _is_valid_effort_level
# ===========================================================================
class TestIsValidEffortLevel(unittest.TestCase):
    def test_valid_levels(self):
        client = _make_client()
        for level in ("default", "low", "medium", "high", "xhigh", "max"):
            self.assertTrue(
                client._is_valid_effort_level(level),
                f"{level} should be valid",
            )

    def test_case_insensitive(self):
        client = _make_client()
        self.assertTrue(client._is_valid_effort_level("High"))
        self.assertTrue(client._is_valid_effort_level("DEFAULT"))

    def test_invalid_level(self):
        client = _make_client()
        self.assertFalse(client._is_valid_effort_level("turbo"))
        self.assertFalse(client._is_valid_effort_level("extreme"))


# ===========================================================================
# 14. _apply_acp_arg_directives
# ===========================================================================
class TestApplyAcpArgDirectives(unittest.TestCase):
    def test_model_flag_seeds_pending(self):
        client = _make_client(acp_args=["--model", "opus", "--stdio"])
        client._apply_acp_arg_directives()
        self.assertEqual(client._pending_session_configs.get("model"), "opus")

    def test_model_flag_invalid_ignored(self):
        client = _make_client(acp_args=["--model", "gpt-4o", "--stdio"])
        client._apply_acp_arg_directives()
        self.assertNotIn("model", client._pending_session_configs)

    def test_permission_mode_flag(self):
        client = _make_client(acp_args=["--permission-mode", "plan", "--stdio"])
        client._apply_acp_arg_directives()
        self.assertEqual(client._pending_session_configs.get("mode"), "plan")

    def test_effort_flag(self):
        client = _make_client(acp_args=["--effort", "high", "--stdio"])
        client._apply_acp_arg_directives()
        self.assertEqual(client._pending_session_configs.get("effort"), "high")

    def test_effort_flag_invalid_ignored(self):
        client = _make_client(acp_args=["--effort", "turbo", "--stdio"])
        client._apply_acp_arg_directives()
        self.assertNotIn("effort", client._pending_session_configs)

    def test_multiple_directives(self):
        client = _make_client(
            acp_args=["--model", "sonnet", "--permission-mode", "auto", "--effort", "low", "--stdio"]
        )
        client._apply_acp_arg_directives()
        self.assertEqual(client._pending_session_configs["model"], "sonnet")
        self.assertEqual(client._pending_session_configs["mode"], "auto")
        self.assertEqual(client._pending_session_configs["effort"], "low")

    def test_unrecognized_flag_passthrough(self):
        args = ["--debug", "--model", "haiku", "--unknown", "val", "--stdio"]
        client = _make_client(acp_args=args)
        client._apply_acp_arg_directives()
        # acp_args should be unchanged
        self.assertEqual(client._acp_args, args)
        self.assertEqual(client._pending_session_configs.get("model"), "haiku")

    def test_duplicate_flag_first_wins(self):
        client = _make_client(acp_args=["--model", "opus", "--model", "haiku", "--stdio"])
        client._apply_acp_arg_directives()
        self.assertEqual(client._pending_session_configs.get("model"), "opus")

    def test_model_flag_no_value_skipped(self):
        client = _make_client(acp_args=["--model"])
        client._apply_acp_arg_directives()
        self.assertNotIn("model", client._pending_session_configs)

    def test_empty_acp_args(self):
        client = _make_client(acp_args=[])
        client._apply_acp_arg_directives()
        self.assertNotIn("model", client._pending_session_configs)
        self.assertEqual(client._pending_session_configs, {})

    # ------------------------------------------------------------------
    # C3 (e40bffccc) — model/config changes on alive session must re-flush
    # ------------------------------------------------------------------
    def test_model_change_flushes_to_alive_session(self):
        """C3 (e40bffccc): when _ensure_session() reuses a live session and
        _pending_session_configs has a new value (e.g. --model re-scanned),
        it MUST call _flush_pending_configs to re-flush via
        session/set_config_option — not just leave them stashed in
        _pending_session_configs.

        Regression: before C3 fix, the flush logic was inline in
        _ensure_session and only ran after session/new. Model/config changes
        on an already-alive session were silently dropped, so e.g. switching
        from opus → sonnet mid-session never actually switched the model.

        This test sets up a fully-alive session, stashes a NEW model into
        _pending_session_configs, and verifies that _ensure_session() (on
        its early-return / "session already alive" branch) triggers the
        flush.
        """
        from collections import deque
        from queue import Queue
        client = _make_client(acp_args=["--model", "opus", "--stdio"])
        client._apply_acp_arg_directives()
        # Sanity: initial scan populated the pending dict.
        self.assertEqual(
            client._pending_session_configs.get("model"), "opus",
        )
        # Simulate an alive session — all four invariants of
        # _ensure_session's early-return branch must hold.
        fake_proc = _FakeProcess()  # _poll_result defaults to None → alive
        client._session_proc = fake_proc
        client._session_inbox = Queue()
        client._session_stderr = deque()
        client._session_id = "sess-alive"
        # Simulate a NEW model directive scanned in (e.g. user re-applied
        # --model with a different value mid-session).
        client._pending_session_configs["model"] = "sonnet"
        # Track flush calls. Real signature is
        # (proc, inbox, stderr_tail, session_id) — NOT (force=False).
        flush_calls: list[dict] = []

        def tracking_flush(proc, inbox, stderr_tail, session_id):
            flush_calls.append(
                {
                    "proc": proc,
                    "session_id": session_id,
                    "pending_model": client._pending_session_configs.get(
                        "model"
                    ),
                }
            )

        client._flush_pending_configs = tracking_flush  # type: ignore[assignment]
        # Trigger _ensure_session — the alive-session branch must flush.
        client._ensure_session()
        self.assertEqual(
            len(flush_calls), 1,
            "C3 regression: alive session path did not call "
            "_flush_pending_configs to re-flush pending --model changes",
        )
        self.assertEqual(flush_calls[0]["session_id"], "sess-alive")
        self.assertIs(flush_calls[0]["proc"], fake_proc)
        self.assertEqual(flush_calls[0]["pending_model"], "sonnet")


# ===========================================================================
# 15. B1 (4a798e19f) — dead function regression guard
# ===========================================================================
class TestDeadCodeRemoved(unittest.TestCase):
    def test_trace_to_messages_snapshot_removed(self):
        """B1 (4a798e19f) removed the 49-line trace_to_messages_snapshot
        function. Re-addition would indicate a regression of the dead-code
        review. This test guards against that.
        """
        client = _make_client()
        self.assertFalse(
            hasattr(client, "trace_to_messages_snapshot"),
            "trace_to_messages_snapshot was removed in B1; "
            "re-add only if a real caller emerges",
        )


if __name__ == "__main__":
    unittest.main()
