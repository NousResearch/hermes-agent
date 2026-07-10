"""Unit tests for run_agent.py (AIAgent) — tool execution (sequential + concurrent), path-scope, codex arg normalization.

Split out of the former monolithic ``tests/run_agent/test_run_agent.py`` (which
outgrew the per-file CI wall-clock cap). Shared fixtures live in ``conftest.py``;
mock-builders in ``_run_agent_helpers.py``.
"""

import ast
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from agent.codex_responses_adapter import _normalize_codex_response

from tests.run_agent._run_agent_helpers import (
    _mock_assistant_msg,
    _mock_response,
    _mock_tool_call,
)


class TestExecuteToolCalls:
    def test_single_tool_executed(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch(
            "run_agent.handle_function_call", return_value="search result"
        ) as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            # enabled_tools passes the agent's own valid_tool_names
            args, kwargs = mock_hfc.call_args
            assert args[:3] == ("web_search", {"q": "test"}, "task-1")
            assert set(kwargs.get("enabled_tools", [])) == agent.valid_tool_names
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "search result" in messages[0]["content"]

    def test_keyboard_interrupt_emits_cancelled_post_tool_hook(self, agent, monkeypatch):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        hook_calls = []
        agent.session_id = "session-1"
        agent._current_turn_id = "turn-1"
        agent._current_api_request_id = "api-1"

        def _capture_hook(hook_name, **kwargs):
            hook_calls.append((hook_name, kwargs))
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture_hook)
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

        with (
            patch("run_agent.handle_function_call", side_effect=KeyboardInterrupt),
            patch("run_agent._set_interrupt"),
            pytest.raises(KeyboardInterrupt),
        ):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["tool_name"] == "web_search"
        assert post_calls[0]["tool_call_id"] == "c1"
        assert post_calls[0]["session_id"] == "session-1"
        assert post_calls[0]["turn_id"] == "turn-1"
        assert post_calls[0]["api_request_id"] == "api-1"
        assert post_calls[0]["status"] == "cancelled"
        assert post_calls[0]["error_type"] == "keyboard_interrupt"
        assert json.loads(post_calls[0]["result"])["status"] == "cancelled"

    def test_interrupt_skips_remaining(self, agent):
        tc1 = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments="{}", call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("run_agent._set_interrupt"):
            agent.interrupt()

        agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Both calls should be skipped with cancellation messages
        assert len(messages) == 2
        assert (
            "cancelled" in messages[0]["content"].lower()
            or "interrupted" in messages[0]["content"].lower()
        )

    def test_invalid_json_args_defaults_empty(self, agent):
        # BEHAVIOR CHANGE (2026-07-10 upstream parity sync): the fork used to
        # leniently coerce malformed tool arguments to ``{}`` and run the tool
        # anyway. Upstream refactored arg-parsing into ``_parse_tool_arguments``
        # which REJECTS malformed JSON — it appends a malformed-args error tool
        # result and does NOT dispatch the tool (safer: never execute an action
        # with fabricated/empty args the model didn't actually send). The merge
        # adopts upstream's strict behavior consistently across the concurrent
        # and sequential paths.
        tc = _mock_tool_call(
            name="web_search", arguments="not valid json", call_id="c1"
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch("run_agent.handle_function_call", return_value="ok") as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            # Malformed args are rejected before dispatch — the tool never runs.
            mock_hfc.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "c1"
        # The tool result carries a structured malformed-arguments error.
        assert "error" in messages[0]["content"].lower() or "invalid" in messages[0]["content"].lower()

    def test_result_truncation_over_100k(self, agent, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        big_result = "x" * 150_000
        with patch("run_agent.handle_function_call", return_value=big_result):
            agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Content should be replaced with persisted-output or truncation
        assert len(messages[0]["content"]) < 150_000
        assert ("Truncated" in messages[0]["content"] or "<persisted-output>" in messages[0]["content"])

    def test_quiet_tool_output_suppressed_when_progress_callback_present(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        agent.tool_progress_callback = lambda *args, **kwargs: None

        with patch("run_agent.handle_function_call", return_value="search result"), \
             patch.object(agent, "_safe_print") as mock_print:
            agent._execute_tool_calls(mock_msg, messages, "task-1")

        mock_print.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"

    def test_quiet_tool_output_prints_without_progress_callback(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        agent.platform = "cli"
        agent.tool_progress_callback = None

        with patch("run_agent.handle_function_call", return_value="search result"), \
             patch.object(agent, "_safe_print") as mock_print:
            agent._execute_tool_calls(mock_msg, messages, "task-1")

        mock_print.assert_called_once()
        assert "search" in str(mock_print.call_args.args[0]).lower()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"

    def test_quiet_tool_output_suppressed_without_progress_callback_for_non_cli_agent(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        agent.platform = None
        agent.tool_progress_callback = None

        with patch("run_agent.handle_function_call", return_value="search result"), \
             patch.object(agent, "_safe_print") as mock_print:
            agent._execute_tool_calls(mock_msg, messages, "task-1")

        mock_print.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"

    def test_vprint_suppressed_in_parseable_quiet_mode(self, agent):
        agent.suppress_status_output = True

        with patch.object(agent, "_safe_print") as mock_print:
            agent._vprint("status line", force=True)
            agent._vprint("normal line")

        mock_print.assert_not_called()

    def test_run_conversation_suppresses_retry_noise_in_parseable_quiet_mode(self, agent):
        class _RateLimitError(Exception):
            status_code = 429

            def __str__(self):
                return "Error code: 429 - Rate limit exceeded."

        responses = [_RateLimitError(), _mock_response(content="Recovered")]

        def _fake_api_call(api_kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        agent.suppress_status_output = True
        agent._interruptible_api_call = _fake_api_call
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None

        captured = io.StringIO()
        agent._print_fn = lambda *args, **kw: print(*args, file=captured, **kw)

        with patch("run_agent.time.sleep", return_value=None):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered"
        output = captured.getvalue()
        assert "API call failed" not in output
        assert "Rate limit reached" not in output


class TestConcurrentToolExecution:
    """Tests for _execute_tool_calls_concurrent and dispatch logic."""

    def test_single_tool_uses_sequential_path(self, agent):
        """Single tool call should use sequential path, not concurrent."""
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_clarify_forces_sequential(self, agent):
        """Batch containing clarify should use sequential path."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="clarify", arguments='{"question":"ok?"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_multiple_tools_uses_concurrent_path(self, agent):
        """Multiple read-only tools should use concurrent path."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="read_file", arguments='{"path":"x.py"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_con.assert_called_once()
                mock_seq.assert_not_called()

    def test_terminal_batch_forces_sequential(self, agent):
        """Stateful tools should not share the concurrent execution path."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="terminal", arguments='{"command":"pwd"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_write_batch_forces_sequential(self, agent):
        """File mutations should stay ordered within a turn."""
        tc1 = _mock_tool_call(name="read_file", arguments='{"path":"x.py"}', call_id="c1")
        tc2 = _mock_tool_call(name="write_file", arguments='{"path":"x.py","content":"print(1)"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_disjoint_write_batch_uses_concurrent_path(self, agent):
        """Independent file writes should still run concurrently."""
        tc1 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/a.py","content":"print(1)"}',
            call_id="c1",
        )
        tc2 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/b.py","content":"print(2)"}',
            call_id="c2",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_con.assert_called_once()
                mock_seq.assert_not_called()

    def test_overlapping_write_batch_forces_sequential(self, agent):
        """Writes to the same file must stay ordered."""
        tc1 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/a.py","content":"print(1)"}',
            call_id="c1",
        )
        tc2 = _mock_tool_call(
            name="patch",
            arguments='{"path":"src/a.py","old_string":"1","new_string":"2"}',
            call_id="c2",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_malformed_json_args_forces_sequential(self, agent):
        """Unparseable tool arguments should fall back to sequential."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments="NOT JSON {{{", call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_non_dict_args_forces_sequential(self, agent):
        """Tool arguments that parse to a non-dict type should fall back to sequential."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='"just a string"', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_concurrent_executes_all_tools(self, agent):
        """Concurrent path should execute all tools and append results in order."""
        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"alpha"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"beta"}', call_id="c2")
        tc3 = _mock_tool_call(name="web_search", arguments='{"q":"gamma"}', call_id="c3")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2, tc3])
        messages = []

        call_log = []

        def fake_handle(name, args, task_id, **kwargs):
            call_log.append(name)
            return json.dumps({"result": args.get("q", "")})

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 3
        # Results must be in original order
        assert messages[0]["tool_call_id"] == "c1"
        assert messages[1]["tool_call_id"] == "c2"
        assert messages[2]["tool_call_id"] == "c3"
        # All should be tool messages
        assert all(m["role"] == "tool" for m in messages)
        # Content should contain the query results
        assert "alpha" in messages[0]["content"]
        assert "beta" in messages[1]["content"]
        assert "gamma" in messages[2]["content"]

    def test_concurrent_preserves_order_despite_timing(self, agent):
        """Even if tools finish in different order, messages should be in original order."""
        import time as _time

        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"slow"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"fast"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        def fake_handle(name, args, task_id, **kwargs):
            q = args.get("q", "")
            if q == "slow":
                _time.sleep(0.1)  # Slow tool
            return f"result_{q}"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert messages[0]["tool_call_id"] == "c1"
        assert "result_slow" in messages[0]["content"]
        assert messages[1]["tool_call_id"] == "c2"
        assert "result_fast" in messages[1]["content"]

    def test_concurrent_handles_tool_error(self, agent):
        """If one tool raises, others should still complete."""
        # Distinguish the two calls by their arguments so the error is tied to
        # a SPECIFIC tool call rather than invocation order. Concurrent
        # execution gives no guarantee that c1's handler runs before c2's, so
        # keying the raise on a call-order counter is racy: under thread-pool
        # scheduling c2 could be invoked first, take the "first call raises"
        # branch, and the error would land in messages[1] instead of
        # messages[0]. Keying on args makes the assertion deterministic.
        tc1 = _mock_tool_call(name="web_search", arguments='{"q": "boom"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q": "ok"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        def fake_handle(name, args, task_id, **kwargs):
            if args.get("q") == "boom":
                raise RuntimeError("boom")
            return "success"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 2
        # Results are ordered by tool_call_id; c1 raised, c2 succeeded.
        assert messages[0]["tool_call_id"] == "c1"
        assert "Error" in messages[0]["content"] or "boom" in messages[0]["content"]
        # Second tool should succeed
        assert messages[1]["tool_call_id"] == "c2"
        assert "success" in messages[1]["content"]

    def test_concurrent_interrupt_before_start(self, agent):
        """If interrupt is requested before concurrent execution, all tools are skipped."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="read_file", arguments='{}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("run_agent._set_interrupt"):
            agent.interrupt()

        agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")
        assert len(messages) == 2
        assert "cancelled" in messages[0]["content"].lower() or "skipped" in messages[0]["content"].lower()
        assert "cancelled" in messages[1]["content"].lower() or "skipped" in messages[1]["content"].lower()

    def test_concurrent_truncates_large_results(self, agent, tmp_path, monkeypatch):
        """Concurrent path should save oversized results to file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        big_result = "x" * 150_000

        with patch("run_agent.handle_function_call", return_value=big_result):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 2
        for m in messages:
            assert len(m["content"]) < 150_000
            assert ("Truncated" in m["content"] or "<persisted-output>" in m["content"])

    def test_invoke_tool_dispatches_to_handle_function_call(self, agent):
        """_invoke_tool should route regular tools through handle_function_call."""
        with patch("run_agent.handle_function_call", return_value="result") as mock_hfc:
            result = agent._invoke_tool("web_search", {"q": "test"}, "task-1")
            mock_hfc.assert_called_once_with(
                "web_search", {"q": "test"}, "task-1",
                tool_call_id=None,
                session_id=agent.session_id,
                turn_id="",
                api_request_id="",
                enabled_tools=list(agent.valid_tool_names),
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                enabled_toolsets=agent.enabled_toolsets,
                disabled_toolsets=agent.disabled_toolsets,
                tool_request_middleware_trace=[],
            )
            assert result == "result"

    def test_sequential_tool_callbacks_fire_in_order(self, agent):
        tool_call = _mock_tool_call(name="web_search", arguments='{"query":"hello"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        starts = []
        completes = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))

        with patch("run_agent.handle_function_call", return_value='{"success": true}'):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert starts == [("c1", "web_search", {"query": "hello"})]
        assert completes == [("c1", "web_search", {"query": "hello"}, '{"success": true}')]

    def test_concurrent_tool_callbacks_fire_for_each_tool(self, agent):
        tc1 = _mock_tool_call(name="web_search", arguments='{"query":"one"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"query":"two"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        starts = []
        completes = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))

        with patch("run_agent.handle_function_call", side_effect=['{"id":1}', '{"id":2}']):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert starts == [
            ("c1", "web_search", {"query": "one"}),
            ("c2", "web_search", {"query": "two"}),
        ]
        assert len(completes) == 2
        assert {entry[0] for entry in completes} == {"c1", "c2"}
        assert {entry[3] for entry in completes} == {'{"id":1}', '{"id":2}'}

    def test_invoke_tool_handles_agent_level_tools(self, agent):
        """_invoke_tool should handle todo tool directly."""
        with patch("tools.todo_tool.todo_tool", return_value='{"ok":true}') as mock_todo:
            result = agent._invoke_tool("todo", {"todos": []}, "task-1")
            mock_todo.assert_called_once()
        assert "ok" in result

    def test_invoke_tool_agent_level_tool_emits_terminal_post_tool_hook(self, agent, monkeypatch):
        """Agent-owned tool paths should close observer tool spans."""
        hook_calls = []
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

        with patch("tools.todo_tool.todo_tool", return_value='{"ok":true}') as mock_todo:
            result = agent._invoke_tool("todo", {"todos": []}, "task-1", tool_call_id="todo-1")

        mock_todo.assert_called_once()
        assert result == '{"ok":true}'
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["tool_name"] == "todo"
        assert post_call[1]["tool_call_id"] == "todo-1"
        assert post_call[1]["status"] == "ok"
        assert post_call[1]["error_type"] is None
        assert isinstance(post_call[1]["duration_ms"], int)

    def test_invoke_tool_blocked_returns_error_and_skips_execution(self, agent, monkeypatch):
        """_invoke_tool should return error JSON when a plugin blocks the tool."""
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked by test policy",
        )
        with patch("tools.todo_tool.todo_tool", side_effect=AssertionError("should not run")) as mock_todo:
            result = agent._invoke_tool("todo", {"todos": []}, "task-1")

        assert json.loads(result) == {"error": "Blocked by test policy"}
        mock_todo.assert_not_called()

    def test_invoke_tool_blocked_skips_handle_function_call(self, agent, monkeypatch):
        """Blocked registry tools should not reach handle_function_call."""
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked",
        )
        with patch("run_agent.handle_function_call", side_effect=AssertionError("should not run")):
            result = agent._invoke_tool("web_search", {"q": "test"}, "task-1")

        assert json.loads(result) == {"error": "Blocked"}

    def test_sequential_blocked_tool_skips_checkpoints_and_callbacks(self, agent, monkeypatch):
        """Sequential path: blocked tool should not trigger checkpoints or start callbacks."""
        tool_call = _mock_tool_call(name="write_file",
                                    arguments='{"path":"test.txt","content":"hello"}',
                                    call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked by policy",
        )
        agent._checkpoint_mgr.enabled = True
        agent._checkpoint_mgr.ensure_checkpoint = MagicMock(
            side_effect=AssertionError("checkpoint should not run")
        )

        starts = []
        agent.tool_start_callback = lambda *a: starts.append(a)

        with patch("run_agent.handle_function_call", side_effect=AssertionError("should not run")):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        agent._checkpoint_mgr.ensure_checkpoint.assert_not_called()
        assert starts == []
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert json.loads(messages[0]["content"]) == {"error": "Blocked by policy"}

    def test_sequential_blocked_tool_emits_terminal_post_tool_hook(self, agent, monkeypatch):
        """Blocked pre_tool_call decisions still terminate observer tool spans."""
        tool_call = _mock_tool_call(name="write_file",
                                    arguments='{"path":"test.txt","content":"hello"}',
                                    call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        hook_calls = []

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked by policy",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

        with patch("run_agent.handle_function_call", side_effect=AssertionError("should not run")):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["tool_name"] == "write_file"
        assert post_call[1]["tool_call_id"] == "c1"
        assert post_call[1]["status"] == "blocked"
        assert post_call[1]["error_type"] == "plugin_block"
        assert post_call[1]["error_message"] == "Blocked by policy"

    def test_sequential_agent_level_tool_emits_terminal_post_tool_hook(self, agent, monkeypatch):
        """Sequential built-in tool paths should also close observer tool spans."""
        tool_call = _mock_tool_call(name="todo", arguments='{"todos":[]}', call_id="todo-1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        hook_calls = []

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

        with patch("tools.todo_tool.todo_tool", return_value='{"ok":true}') as mock_todo:
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        mock_todo.assert_called_once()
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["tool_name"] == "todo"
        assert post_call[1]["tool_call_id"] == "todo-1"
        assert post_call[1]["result"] == '{"ok":true}'
        assert post_call[1]["status"] == "ok"

    def test_sequential_agent_level_tool_execution_middleware_wraps_inline_dispatch(self, agent, monkeypatch):
        """Sequential built-in tool paths should expose the adaptive execution boundary."""
        tool_call = _mock_tool_call(name="todo", arguments='{"todos":[]}', call_id="todo-1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        hook_calls = []
        seen = {}

        def request_middleware(**kwargs):
            return {
                "args": {**kwargs["args"], "request_rewritten": True},
                "source": "request-test",
            }

        def execution_middleware(**kwargs):
            seen["middleware_args"] = kwargs["args"]
            return kwargs["next_call"]({**kwargs["args"], "merge": True})

        manager = SimpleNamespace(_middleware={
            "tool_request": [request_middleware],
            "tool_execution": [execution_middleware],
        })
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_middleware",
            lambda kind, **kwargs: [request_middleware(**kwargs)] if kind == "tool_request" else [],
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

        with patch("tools.todo_tool.todo_tool", return_value='{"ok":true}') as mock_todo:
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert seen["middleware_args"] == {"todos": [], "request_rewritten": True}
        mock_todo.assert_called_once_with(todos=[], merge=True, store=agent._todo_store)
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["tool_name"] == "todo"
        assert post_call[1]["args"] == {"todos": [], "request_rewritten": True, "merge": True}
        assert post_call[1]["middleware_trace"] == [{"source": "request-test"}]

    def test_concurrent_agent_level_tool_preserves_request_middleware_trace(self, agent, monkeypatch):
        tool_call = _mock_tool_call(name="todo", arguments='{"todos":[]}', call_id="todo-1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        hook_calls = []

        def request_middleware(**kwargs):
            return {
                "args": {**kwargs["args"], "request_rewritten": True},
                "source": "request-test",
            }

        manager = SimpleNamespace(_middleware={"tool_request": [request_middleware], "tool_execution": []})
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_middleware",
            lambda kind, **kwargs: [request_middleware(**kwargs)] if kind == "tool_request" else [],
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

        with patch("tools.todo_tool.todo_tool", return_value='{"ok":true}'):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["tool_name"] == "todo"
        assert post_call[1]["args"] == {"todos": [], "request_rewritten": True}
        assert post_call[1]["middleware_trace"] == [{"source": "request-test"}]

    def test_agent_runtime_post_hook_ownership_predicate_covers_agent_tools(self, agent):
        """Sequential and concurrent agent-level paths share post-hook ownership."""
        from agent.agent_runtime_helpers import agent_runtime_owns_post_tool_hook

        for tool_name in ("todo", "session_search", "memory", "clarify", "delegate_task"):
            assert agent_runtime_owns_post_tool_hook(agent, tool_name) is True

        agent._context_engine_tool_names = {"context_query"}
        assert agent_runtime_owns_post_tool_hook(agent, "context_query") is True

        agent._memory_manager = SimpleNamespace(has_tool=lambda name: name == "memory_extra")
        assert agent_runtime_owns_post_tool_hook(agent, "memory_extra") is True
        assert agent_runtime_owns_post_tool_hook(agent, "web_search") is False

    def test_blocked_memory_tool_does_not_reset_counter(self, agent, monkeypatch):
        """Blocked memory tool should not reset the nudge counter."""
        agent._turns_since_memory = 5
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked",
        )
        with patch("tools.memory_tool.memory_tool", side_effect=AssertionError("should not run")):
            result = agent._invoke_tool(
                "memory", {"action": "add", "target": "memory", "content": "x"}, "task-1",
            )

        assert json.loads(result) == {"error": "Blocked"}
        assert agent._turns_since_memory == 5

    def test_concurrent_blocked_write_skips_checkpoint(self, agent, monkeypatch):
        """Concurrent path: blocked write_file should not trigger checkpoint."""
        tc1 = _mock_tool_call(name="write_file",
                              arguments='{"path":"test.txt","content":"hello"}',
                              call_id="c1")
        tc2 = _mock_tool_call(name="read_file",
                              arguments='{"path":"other.py"}',
                              call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked" if args[0] == "write_file" else None,
        )

        agent._checkpoint_mgr.enabled = True

        def fake_handle(name, args, task_id, **kwargs):
            return f"result_{name}"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            with patch.object(agent._checkpoint_mgr, "ensure_checkpoint") as cp_mock:
                agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        cp_mock.assert_not_called()

    def test_concurrent_blocked_patch_skips_checkpoint(self, agent, monkeypatch):
        """Concurrent path: blocked patch should not trigger checkpoint."""
        tc1 = _mock_tool_call(name="patch",
                              arguments='{"path":"f.py","old":"a","new":"b"}',
                              call_id="c1")
        tc2 = _mock_tool_call(name="read_file",
                              arguments='{"path":"other.py"}',
                              call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked" if args[0] == "patch" else None,
        )

        agent._checkpoint_mgr.enabled = True

        def fake_handle(name, args, task_id, **kwargs):
            return f"result_{name}"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            with patch.object(agent._checkpoint_mgr, "ensure_checkpoint") as cp_mock:
                agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        cp_mock.assert_not_called()

    def test_concurrent_blocked_terminal_skips_checkpoint(self, agent, monkeypatch):
        """Concurrent path: blocked terminal should not trigger checkpoint."""
        tc1 = _mock_tool_call(name="terminal",
                              arguments='{"command":"rm -rf /tmp/foo"}',
                              call_id="c1")
        tc2 = _mock_tool_call(name="read_file",
                              arguments='{"path":"other.py"}',
                              call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *args, **kwargs: "Blocked" if args[0] == "terminal" else None,
        )

        agent._checkpoint_mgr.enabled = True

        def fake_handle(name, args, task_id, **kwargs):
            return f"result_{name}"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            with patch.object(agent._checkpoint_mgr, "ensure_checkpoint") as cp_mock:
                with patch("agent.tool_executor._is_destructive_command", return_value=True):
                    agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        cp_mock.assert_not_called()

    def test_concurrent_blocked_write_does_not_steal_slot_from_allowed_write(self, agent, monkeypatch):
        """When write_file is blocked, its dedup slot must not be consumed,
        so a subsequent allowed write_file for the same path still checkpoints."""
        tc1 = _mock_tool_call(name="write_file",
                              arguments='{"path":"dup.txt","content":"blocked"}',
                              call_id="c1")
        tc2 = _mock_tool_call(name="write_file",
                              arguments='{"path":"dup.txt","content":"allowed"}',
                              call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        call_count = {"n": 0}
        def block_first_only(*args, **kwargs):
            call_count["n"] += 1
            return "Blocked" if call_count["n"] == 1 else None

        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            block_first_only,
        )

        agent._checkpoint_mgr.enabled = True

        def fake_handle(name, args, task_id, **kwargs):
            return f"result_{name}"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            with patch.object(agent._checkpoint_mgr, "ensure_checkpoint") as cp_mock:
                agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        # Second (allowed) write must checkpoint even though first was blocked.
        cp_mock.assert_called_once()


class TestAgentRuntimePostHookOwnershipSync:
    """Pin the inline-dispatch tool list against the post-hook ownership set.

    The post_tool_call hook fires from two places: the inline dispatcher in
    agent/tool_executor.py:execute_tool_calls_sequential (for agent-runtime
    tools that never reach handle_function_call) and
    model_tools.handle_function_call itself (for registry-dispatched tools).
    To prevent the executor from silently dropping or double-emitting,
    AGENT_RUNTIME_POST_HOOK_TOOL_NAMES has to match exactly the static
    `function_name == "..."` branches in the inline dispatch chain.

    The chain is the if/elif tower anchored on `_block_msg is not None`.
    Pre-dispatch `function_name == "..."` checks (counter resets, checkpoint
    triggers) live outside the dispatch chain and are explicitly skipped.
    """

    _DISPATCH_ANCHOR_LEFT = "_block_msg"

    @classmethod
    def _is_dispatch_anchor(cls, test_node) -> bool:
        # Looking for `_block_msg is not None`.
        if not isinstance(test_node, ast.Compare):
            return False
        if not (isinstance(test_node.left, ast.Name) and test_node.left.id == cls._DISPATCH_ANCHOR_LEFT):
            return False
        if not (len(test_node.ops) == 1 and isinstance(test_node.ops[0], ast.IsNot)):
            return False
        comparator = test_node.comparators[0]
        return isinstance(comparator, ast.Constant) and comparator.value is None

    @staticmethod
    def _function_name_literal(test_node) -> str | None:
        """Return the string literal X for `function_name == "X"`, else None."""
        if not isinstance(test_node, ast.Compare):
            return None
        if not (isinstance(test_node.left, ast.Name) and test_node.left.id == "function_name"):
            return None
        if not (len(test_node.ops) == 1 and isinstance(test_node.ops[0], ast.Eq)):
            return None
        comparator = test_node.comparators[0]
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
            return comparator.value
        return None

    @classmethod
    def _extract_dispatch_chain_names(cls, func) -> set[str]:
        """Find the if/elif chain anchored on `_block_msg is not None`, return its
        `function_name == "..."` literals."""
        source = inspect.cleandoc("\n" + inspect.getsource(func))
        tree = ast.parse(source)
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not cls._is_dispatch_anchor(node.test):
                continue
            current = node
            while current is not None:
                literal = cls._function_name_literal(current.test)
                if literal is not None:
                    names.add(literal)
                if current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    current = current.orelse[0]
                else:
                    current = None
            break
        return names

    @classmethod
    def _extract_invoke_tool_names(cls, func) -> set[str]:
        """invoke_tool uses a flat if/elif on function_name directly; walk every
        Compare in the function body (no other static `function_name == "..."`
        checks live there)."""
        source = inspect.cleandoc("\n" + inspect.getsource(func))
        tree = ast.parse(source)
        names: set[str] = set()
        for node in ast.walk(tree):
            literal = cls._function_name_literal(node)
            if literal is not None:
                names.add(literal)
        return names

    def test_frozenset_matches_inline_dispatch_chain(self):
        from agent import tool_executor
        from agent.agent_runtime_helpers import AGENT_RUNTIME_POST_HOOK_TOOL_NAMES

        inline_names = self._extract_dispatch_chain_names(
            tool_executor.execute_tool_calls_sequential
        )
        assert inline_names, (
            "Could not find the dispatch chain (anchored on "
            "`_block_msg is not None`) in execute_tool_calls_sequential. "
            "If the dispatcher was refactored, update _DISPATCH_ANCHOR_LEFT "
            "and the walker in this test."
        )
        assert inline_names == set(AGENT_RUNTIME_POST_HOOK_TOOL_NAMES), (
            "Inline dispatch chain in "
            "agent/tool_executor.py:execute_tool_calls_sequential has drifted "
            "from AGENT_RUNTIME_POST_HOOK_TOOL_NAMES in "
            "agent/agent_runtime_helpers.py.\n"
            f"  Inline branches:     {sorted(inline_names)}\n"
            f"  Ownership frozenset: {sorted(AGENT_RUNTIME_POST_HOOK_TOOL_NAMES)}\n"
            "Update both together so post_tool_call fires exactly once per "
            "tool execution."
        )

    def test_invoke_tool_dispatch_matches_inline_dispatch_chain(self):
        """invoke_tool (concurrent path) and the inline dispatcher (sequential
        path) must cover the same set of agent-runtime tools — otherwise
        post_tool_call fires inconsistently depending on which executor ran
        the tool."""
        from agent import agent_runtime_helpers, tool_executor

        invoke_tool_names = self._extract_invoke_tool_names(
            agent_runtime_helpers.invoke_tool
        )
        inline_names = self._extract_dispatch_chain_names(
            tool_executor.execute_tool_calls_sequential
        )
        assert invoke_tool_names == inline_names, (
            "Static `function_name == \"...\"` branches diverged between "
            "agent/agent_runtime_helpers.py:invoke_tool (concurrent path) "
            "and agent/tool_executor.py:execute_tool_calls_sequential "
            "(sequential path).\n"
            f"  invoke_tool:                   {sorted(invoke_tool_names)}\n"
            f"  execute_tool_calls_sequential: {sorted(inline_names)}"
        )


class TestPathsOverlap:
    """Unit tests for the _paths_overlap helper."""

    def test_same_path_overlaps(self):
        from run_agent import _paths_overlap
        assert _paths_overlap(Path("src/a.py"), Path("src/a.py"))

    def test_siblings_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path("src/a.py"), Path("src/b.py"))

    def test_parent_child_overlap(self):
        from run_agent import _paths_overlap
        assert _paths_overlap(Path("src"), Path("src/sub/a.py"))

    def test_different_roots_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path("src/a.py"), Path("other/a.py"))

    def test_nested_vs_flat_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path("src/sub/a.py"), Path("src/a.py"))

    def test_empty_paths_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path(""), Path(""))

    def test_one_empty_path_does_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path(""), Path("src/a.py"))
        assert not _paths_overlap(Path("src/a.py"), Path(""))


class TestParallelScopePathNormalization:
    def test_extract_parallel_scope_path_normalizes_relative_to_cwd(self, tmp_path, monkeypatch):
        from run_agent import _extract_parallel_scope_path

        monkeypatch.chdir(tmp_path)

        scoped = _extract_parallel_scope_path("write_file", {"path": "./notes.txt"})

        assert scoped == tmp_path / "notes.txt"

    def test_extract_parallel_scope_path_treats_relative_and_absolute_same_file_as_same_scope(self, tmp_path, monkeypatch):
        from run_agent import _extract_parallel_scope_path, _paths_overlap

        monkeypatch.chdir(tmp_path)
        abs_path = tmp_path / "notes.txt"

        rel_scoped = _extract_parallel_scope_path("write_file", {"path": "notes.txt"})
        abs_scoped = _extract_parallel_scope_path("write_file", {"path": str(abs_path)})

        assert rel_scoped == abs_scoped
        assert _paths_overlap(rel_scoped, abs_scoped)

    def test_should_parallelize_tool_batch_rejects_same_file_with_mixed_path_spellings(self, tmp_path, monkeypatch):
        from run_agent import _should_parallelize_tool_batch

        monkeypatch.chdir(tmp_path)
        tc1 = _mock_tool_call(name="write_file", arguments='{"path":"notes.txt","content":"one"}', call_id="c1")
        tc2 = _mock_tool_call(name="write_file", arguments=f'{{"path":"{tmp_path / "notes.txt"}","content":"two"}}', call_id="c2")

        assert not _should_parallelize_tool_batch([tc1, tc2])


class TestMcpParallelToolBatch:
    """Integration test: _should_parallelize_tool_batch respects MCP parallel flag."""

    def test_mcp_tools_default_sequential(self):
        """MCP tools without supports_parallel_tool_calls are sequential."""
        from run_agent import _should_parallelize_tool_batch
        tc1 = _mock_tool_call(name="mcp__github__list_repos", arguments='{"org":"openai"}', call_id="c1")
        tc2 = _mock_tool_call(name="mcp__github__search_code", arguments='{"q":"test"}', call_id="c2")
        assert not _should_parallelize_tool_batch([tc1, tc2])

    def test_mcp_tools_parallel_when_server_opted_in(self):
        """MCP tools from a parallel-safe server can run concurrently."""
        from run_agent import _should_parallelize_tool_batch
        from tools.mcp_tool import _mcp_tool_server_names, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.add("github")
            _mcp_tool_server_names["mcp__github__list_repos"] = "github"
            _mcp_tool_server_names["mcp__github__search_code"] = "github"
        try:
            tc1 = _mock_tool_call(name="mcp__github__list_repos", arguments='{"org":"openai"}', call_id="c1")
            tc2 = _mock_tool_call(name="mcp__github__search_code", arguments='{"q":"test"}', call_id="c2")
            assert _should_parallelize_tool_batch([tc1, tc2])
        finally:
            with _lock:
                _parallel_safe_servers.discard("github")
                _mcp_tool_server_names.pop("mcp__github__list_repos", None)
                _mcp_tool_server_names.pop("mcp__github__search_code", None)

    def test_mixed_mcp_and_builtin_parallel(self):
        """MCP parallel tools mixed with built-in parallel-safe tools."""
        from run_agent import _should_parallelize_tool_batch
        from tools.mcp_tool import _mcp_tool_server_names, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.add("docs")
            _mcp_tool_server_names["mcp__docs__search"] = "docs"
        try:
            tc1 = _mock_tool_call(name="mcp__docs__search", arguments='{"query":"api"}', call_id="c1")
            tc2 = _mock_tool_call(name="web_search", arguments='{"query":"test"}', call_id="c2")
            assert _should_parallelize_tool_batch([tc1, tc2])
        finally:
            with _lock:
                _parallel_safe_servers.discard("docs")
                _mcp_tool_server_names.pop("mcp__docs__search", None)

    def test_mixed_parallel_and_serial_mcp_servers(self):
        """One parallel MCP server + one non-parallel MCP server = sequential."""
        from run_agent import _should_parallelize_tool_batch
        from tools.mcp_tool import _mcp_tool_server_names, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.add("docs")
            # "github" is NOT in _parallel_safe_servers
            _mcp_tool_server_names["mcp__docs__search"] = "docs"
            _mcp_tool_server_names["mcp__github__list_repos"] = "github"
        try:
            tc1 = _mock_tool_call(name="mcp__docs__search", arguments='{"query":"api"}', call_id="c1")
            tc2 = _mock_tool_call(name="mcp__github__list_repos", arguments='{"org":"openai"}', call_id="c2")
            assert not _should_parallelize_tool_batch([tc1, tc2])
        finally:
            with _lock:
                _parallel_safe_servers.discard("docs")
                _mcp_tool_server_names.pop("mcp__docs__search", None)
                _mcp_tool_server_names.pop("mcp__github__list_repos", None)


class TestNormalizeCodexDictArguments:
    """_normalize_codex_response must produce valid JSON strings for tool
    call arguments, even when the Responses API returns them as dicts."""

    def _make_codex_response(self, item_type, arguments, item_status="completed"):
        """Build a minimal Responses API response with a single tool call."""
        item = SimpleNamespace(
            type=item_type,
            status=item_status,
        )
        if item_type == "function_call":
            item.name = "web_search"
            item.arguments = arguments
            item.call_id = "call_abc123"
            item.id = "fc_abc123"
        elif item_type == "custom_tool_call":
            item.name = "web_search"
            item.input = arguments
            item.call_id = "call_abc123"
            item.id = "fc_abc123"
        return SimpleNamespace(
            output=[item],
            status="completed",
        )

    def test_function_call_dict_arguments_produce_valid_json(self, agent):
        """dict arguments from function_call must be serialised with
        json.dumps, not str(), so downstream json.loads() succeeds."""
        args_dict = {"query": "weather in NYC", "units": "celsius"}
        response = self._make_codex_response("function_call", args_dict)
        msg, _ = _normalize_codex_response(response)
        tc = msg.tool_calls[0]
        parsed = json.loads(tc.function.arguments)
        assert parsed == args_dict

    def test_custom_tool_call_dict_arguments_produce_valid_json(self, agent):
        """dict arguments from custom_tool_call must also use json.dumps."""
        args_dict = {"path": "/tmp/test.txt", "content": "hello"}
        response = self._make_codex_response("custom_tool_call", args_dict)
        msg, _ = _normalize_codex_response(response)
        tc = msg.tool_calls[0]
        parsed = json.loads(tc.function.arguments)
        assert parsed == args_dict

    def test_string_arguments_unchanged(self, agent):
        """String arguments must pass through without modification."""
        args_str = '{"query": "test"}'
        response = self._make_codex_response("function_call", args_str)
        msg, _ = _normalize_codex_response(response)
        tc = msg.tool_calls[0]
        assert tc.function.arguments == args_str
