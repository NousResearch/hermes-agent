"""test_run_agent_tool_exec: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from agent.codex_responses_adapter import _normalize_codex_response
from agent.memory_manager import MemoryManager
from tests.agent._run_agent_helpers import _mock_assistant_msg, _mock_response, _mock_tool_call


class TestExecuteToolCalls:
    def test_single_tool_executed(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch(
            "model_tools.handle_function_call", return_value="search result"
        ) as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            # enabled_tools passes the agent's own valid_tool_names
            args, kwargs = mock_hfc.call_args
            assert args[:3] == ("web_search", {"q": "test"}, "task-1")
            assert set(kwargs.get("enabled_tools", [])) == agent.valid_tool_names
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "search result" in messages[0]["content"]

    def test_sequential_tool_calls_run_without_delay(self, agent):
        """Two sequential tool calls execute back-to-back with no sleep between them."""
        tc1 = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments="{}", call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        # ``agent.tool_executor`` does a plain ``import time``, so patching
        # ``agent.tool_executor.time.sleep`` rebinds the attribute on the shared
        # ``time`` MODULE -- every thread in the process sees the mock, not only
        # the dispatch loop under test. ``AIAgent.__init__`` starts a daemon
        # ``env-probe`` thread (``tools.env_probe.warm_environment_probe_async``)
        # that shells out to python3/pip, and ``subprocess.Popen.wait()``'s poll
        # loop charges thousands of ``time.sleep(0.002..0.05)`` calls to that
        # mock. A bare ``assert_not_called()`` therefore fails on unrelated
        # background traffic whenever the probe is still running -- which is
        # whenever no earlier test in the process happened to quiesce it.
        # Attribute the sleeps to their calling thread so the assertion gates
        # the between-call delay it actually names.
        dispatch_thread = threading.get_ident()
        dispatch_sleeps: list[float] = []

        def _record_sleep(seconds=0.0, *args, **kwargs):
            if threading.get_ident() == dispatch_thread:
                dispatch_sleeps.append(seconds)

        with (
            patch("model_tools.handle_function_call", return_value="ok") as mock_hfc,
            patch("agent.tool_executor.time.sleep", side_effect=_record_sleep),
        ):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")
        assert mock_hfc.call_count == 2
        assert dispatch_sleeps == []
        tool_results = [m for m in messages if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_results] == ["c1", "c2"]

    def test_sequential_memory_remove_notifies_provider_with_tool_result(self, agent):
        old_text = "stale preference entry"
        tc = _mock_tool_call(
            name="memory",
            arguments=json.dumps({
                "action": "remove",
                "target": "memory",
                "old_text": old_text,
            }),
            call_id="mem-1",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        calls = []

        class FakeMemoryManager(MemoryManager):
            def has_tool(self, tool_name):
                return False

            def on_memory_write(self, action, target, content, metadata=None):
                calls.append((action, target, content, metadata or {}))

        agent._memory_manager = FakeMemoryManager()
        agent._memory_store = object()

        with patch("tools.memory_tool.memory_tool", return_value=json.dumps({"success": True})):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert len(calls) == 1
        action, target, content, metadata = calls[0]
        assert (action, target, content) == ("remove", "memory", "")
        assert metadata["old_text"] == old_text
        assert metadata["tool_call_id"] == "mem-1"
        assert messages[-1]["tool_call_id"] == "mem-1"

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

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _capture_hook)
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

        with (
            patch("model_tools.handle_function_call", side_effect=KeyboardInterrupt),
            patch("run_agent._set_interrupt"),
            patch("agent.interrupt_control._set_interrupt"),
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

    def test_interrupt_skips_remaining(self, agent, monkeypatch):
        tc1 = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments="{}", call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        hook_calls = []

        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        with patch("run_agent._set_interrupt"), patch("agent.interrupt_control._set_interrupt"):
            agent.interrupt()

        agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Both calls should be skipped with cancellation messages
        assert len(messages) == 2
        assert (
            "cancelled" in messages[0]["content"].lower()
            or "interrupted" in messages[0]["content"].lower()
        )
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert [call["tool_call_id"] for call in post_calls] == ["c1", "c2"]
        assert all(call["status"] == "cancelled" for call in post_calls)

    def test_invalid_json_args_are_rejected_without_dispatch(self, agent, monkeypatch):
        tc = _mock_tool_call(
            name="web_search", arguments="not valid json", call_id="c1"
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        hook_calls = []
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        with patch("model_tools.handle_function_call", return_value="ok") as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            mock_hfc.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "c1"
        assert "tool was not executed" in messages[0]["content"].lower()
        [post_call] = [
            kwargs for name, kwargs in hook_calls if name == "post_tool_call"
        ]
        assert post_call["tool_call_id"] == "c1"
        assert post_call["status"] == "error"
        assert post_call["error_type"] == "invalid_tool_arguments"

    def test_concurrent_invalid_json_args_emit_terminal_hook(self, agent, monkeypatch):
        tc = _mock_tool_call(
            name="web_search", arguments="not valid json", call_id="c1"
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        hook_calls = []
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        [post_call] = [
            kwargs for name, kwargs in hook_calls if name == "post_tool_call"
        ]
        assert post_call["tool_call_id"] == "c1"
        assert post_call["status"] == "error"
        assert post_call["error_type"] == "invalid_tool_arguments"

    def test_none_args_rejected_without_dispatch(self, agent):
        """None arguments must not crash the dispatch path. Current contract:
        malformed (non-string, non-JSON-object) args are rejected without
        executing the tool — same as invalid JSON strings. The mainline
        run_conversation path normalizes None to "{}" BEFORE dispatch (see
        test_tool_call_none_args_verbose_logging_does_not_crash), so this
        direct-dispatch path only needs to degrade gracefully, not coerce."""
        tc = _mock_tool_call(name="web_search", arguments=None, call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch("model_tools.handle_function_call", return_value="ok") as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            mock_hfc.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "c1"
        assert "tool was not executed" in messages[0]["content"].lower()

    def test_result_truncation_over_100k(self, agent, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        big_result = "x" * 150_000
        with patch("model_tools.handle_function_call", return_value=big_result):
            agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Content should be replaced with persisted-output or truncation
        assert len(messages[0]["content"]) < 150_000
        assert ("Truncated" in messages[0]["content"] or "<persisted-output>" in messages[0]["content"])

    def test_quiet_tool_output_suppressed_when_progress_callback_present(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        agent.tool_progress_callback = lambda *args, **kwargs: None

        with patch("model_tools.handle_function_call", return_value="search result"), \
             patch.object(agent, "_safe_print") as mock_print:
            agent._execute_tool_calls(mock_msg, messages, "task-1")

        mock_print.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"


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

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
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

    def test_concurrent_none_args_rejected_without_crash(self, agent):
        """Concurrent executor must not crash on arguments=None. Current
        contract (_parse_tool_arguments): non-object args are rejected with
        a structured error result and the tool is not executed; the valid
        sibling still runs. One result per call, in order."""
        tc1 = _mock_tool_call(name="web_search", arguments=None, call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"ok"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        seen_args = []

        def fake_handle(name, args, task_id, **kwargs):
            seen_args.append((kwargs["tool_call_id"], args))
            return "ok"

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        # Only the valid call executed; the None-args call was rejected.
        assert seen_args == [("c2", {"q": "ok"})]
        assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]
        assert "tool was not executed" in messages[0]["content"].lower()

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

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert messages[0]["tool_call_id"] == "c1"
        assert "result_slow" in messages[0]["content"]
        assert messages[1]["tool_call_id"] == "c2"
        assert "result_fast" in messages[1]["content"]


    def test_concurrent_submit_shutdown_error_returns_tool_errors(self, agent):
        """Submit-time interpreter shutdown should not escape the outer loop."""

        class ShutdownExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, *args, **kwargs):
                raise RuntimeError("cannot schedule new futures after interpreter shutdown")

            def shutdown(self, *args, **kwargs):
                pass

        tc1 = _mock_tool_call(name="web_search", arguments='{"q": "alpha"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q": "beta"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("tools.daemon_pool.DaemonThreadPoolExecutor", ShutdownExecutor):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 2
        assert messages[0]["tool_call_id"] == "c1"
        assert messages[1]["tool_call_id"] == "c2"
        assert all("Python interpreter is shutting down" in m["content"] for m in messages)


    def test_sequential_tool_callbacks_fire_in_order(self, agent):
        tool_call = _mock_tool_call(name="web_search", arguments='{"query":"hello"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        starts = []
        completes = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))

        with patch("model_tools.handle_function_call", return_value='{"success": true}'):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert starts == [("c1", "web_search", {"query": "hello"})]
        assert completes == [("c1", "web_search", {"query": "hello"}, '{"success": true}')]


    def test_sequential_browser_type_callbacks_redact_api_key(self, agent):
        secret = "sk-proj-ABCD1234567890EFGH"
        tool_call = _mock_tool_call(
            name="browser_type",
            arguments=json.dumps({"ref": "@apikey", "text": secret}),
            call_id="c-secret",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        starts = []
        completes = []
        progress = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))
        agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress.append((event, name, preview, args))

        with patch("model_tools.handle_function_call", return_value='{"success": true, "typed": "sk-pro...EFGH"}'):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert starts[0][2]["text"].startswith("sk-pro")
        assert completes[0][2]["text"].startswith("sk-pro")
        assert progress[0][2].startswith("sk-pro")
        assert secret not in repr(starts + completes + progress)


    def test_sequential_blocked_tool_skips_checkpoints_and_callbacks(self, agent, monkeypatch):
        """Sequential path: blocked tool should not trigger checkpoints or start callbacks."""
        tool_call = _mock_tool_call(name="write_file",
                                    arguments='{"path":"test.txt","content":"hello"}',
                                    call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []

        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *args, **kwargs: ("Blocked by policy", None),
        )
        agent._checkpoint_mgr.enabled = True
        agent._checkpoint_mgr.ensure_checkpoint = MagicMock(
            side_effect=AssertionError("checkpoint should not run")
        )

        starts = []
        agent.tool_start_callback = lambda *a: starts.append(a)

        with patch("model_tools.handle_function_call", side_effect=AssertionError("should not run")):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        agent._checkpoint_mgr.ensure_checkpoint.assert_not_called()
        assert starts == []
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert json.loads(messages[0]["content"]) == {"error": "Blocked by policy"}


    @pytest.mark.parametrize("concurrent", [False, True])
    def test_tool_execution_middleware_replacement_emits_one_terminal_hook(
        self,
        agent,
        monkeypatch,
        concurrent,
    ):
        """A middleware replacement owns the result but not lifecycle closure."""
        tool_call = _mock_tool_call(
            name="terminal",
            arguments='{"command":"must-not-run"}',
            call_id="terminal-1",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        hook_calls = []

        def execution_middleware(**kwargs):
            return '{"intercepted":true}'

        manager = SimpleNamespace(_middleware={
            "tool_request": [],
            "tool_execution": [execution_middleware],
        })
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

        with patch(
            "model_tools.handle_function_call",
            side_effect=AssertionError("middleware replacement must not dispatch"),
        ):
            if concurrent:
                agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")
            else:
                agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        post_calls = [
            payload for name, payload in hook_calls if name == "post_tool_call"
        ]
        assert len(post_calls) == 1
        assert post_calls[0]["tool_name"] == "terminal"
        assert post_calls[0]["tool_call_id"] == "terminal-1"
        assert post_calls[0]["status"] == "ok"
        assert post_calls[0]["result"] == '{"intercepted":true}'

    def test_blocked_memory_tool_does_not_reset_counter(self, agent, monkeypatch):
        """Blocked memory tool should not reset the nudge counter."""
        agent._turns_since_memory = 5
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *args, **kwargs: ("Blocked", None),
        )
        with patch("tools.memory_tool.memory_tool", side_effect=AssertionError("should not run")):
            result = agent._invoke_tool(
                "memory", {"action": "add", "target": "memory", "content": "x"}, "task-1",
            )

        assert json.loads(result) == {"error": "Blocked"}
        assert agent._turns_since_memory == 5


    def test_managed_tool_pipeline_rejects_second_dispatch(self, agent, monkeypatch):
        from agent import relay_tools, tool_executor

        dispatched = []
        duplicate_errors = []
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: SimpleNamespace(
                payload=args,
                trace=[],
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            lambda _name, args, callback, **_kwargs: callback(args),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *_args, **_kwargs: (None, None),
        )
        monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)

        def invoke_twice(name, args, callback, **kwargs):
            del name, kwargs
            result = callback(args)
            try:
                callback(args)
            except RuntimeError as exc:
                duplicate_errors.append(str(exc))
            return result, args

        monkeypatch.setattr(relay_tools, "execute", invoke_twice)

        outcome = tool_executor._run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "true"},
            effective_task_id="task-1",
            tool_call_id="call-1",
            execute=lambda args: dispatched.append(args) or "ok",
        )

        assert outcome.result == "ok"
        assert dispatched == [{"command": "true"}]
        assert len(duplicate_errors) == 1
        assert outcome.blocked is False

    def test_managed_tool_pipeline_allows_one_concurrent_dispatch(
        self,
        agent,
        monkeypatch,
    ):
        from agent import relay_tools, tool_executor

        dispatched = []
        results = []
        errors = []
        barrier = threading.Barrier(2)
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: SimpleNamespace(
                payload=args,
                trace=[],
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            lambda _name, args, callback, **_kwargs: callback(args),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *_args, **_kwargs: (None, None),
        )
        monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)

        def invoke_concurrently(name, args, callback, **kwargs):
            del name, kwargs

            def invoke():
                barrier.wait(timeout=2)
                try:
                    results.append(callback(args))
                except RuntimeError as exc:
                    errors.append(str(exc))

            threads = [threading.Thread(target=invoke) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
            return results[0], args

        monkeypatch.setattr(relay_tools, "execute", invoke_concurrently)

        outcome = tool_executor._run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "true"},
            effective_task_id="task-1",
            tool_call_id="call-1",
            execute=lambda args: dispatched.append(args) or "ok",
        )

        assert outcome.result == "ok"
        assert dispatched == [{"command": "true"}]
        assert len(errors) == 1
        assert outcome.blocked is False

class TestAgentRuntimePostHookOwnershipSync:
    """Exercise post-hook ownership through both agent-runtime tool paths."""

    _CASES = (
        ("todo_list", {"todos": []}),
        ("session_search", {"query": "needle"}),
        ("memory", {"action": "view", "target": "memory"}),
        ("clarify", {"question": "Continue?"}),
        ("read_terminal", {}),
        ("desktop_preview", {"action": "read"}),
        ("drive_preview", {"action": "elements"}),
        ("annotate_preview", {"action": "clear"}),
        ("read_window_below", {}),
        ("manage_connections", {"action": "install", "connectors": [{"name": "linear", "mcp": True}]}),
        ("setup_mcp", {"server": "linear", "action": "install"}),
        ("manage_catalog", {"action": "search", "query": "blender"}),
        ("gui_tour", {"action": "stop"}),
        ("delegate_task", {"goal": "Check the child path"}),
    )

    @pytest.mark.parametrize(("tool_name", "tool_args"), _CASES)
    def test_agent_runtime_tools_emit_once_per_executor_path(
        self,
        agent,
        monkeypatch,
        tool_name,
        tool_args,
    ):
        from agent.agent_runtime_helpers import AGENT_RUNTIME_POST_HOOK_TOOL_NAMES

        hook_calls = []
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *args, **kwargs: (None, None),
        )
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "tools.todo_tool.todo_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.memory_tool.memory_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.clarify_tool.clarify_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.read_terminal_tool.read_terminal_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.read_preview_tool.read_preview_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.drive_preview_tool.drive_preview_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.annotate_preview_tool.annotate_preview_tool",
            lambda **kwargs: '{"ok":true}',
        )
        monkeypatch.setattr(
            "tools.read_window_tool.read_window_below_tool",
            lambda **kwargs: '{"ok":true}',
        )
        # manage_connections / setup_mcp shim: no card on this fake agent, so the MCP leg runs the
        # backend at once; pin the catalog and the backend so the run is hermetic.
        monkeypatch.setattr("tools.connectors.mcp._catalog_names", lambda: ["linear"])
        monkeypatch.setattr("tools.connectors.mcp._configured_names", lambda: [])

        class _NoInstallBackend:
            def required_env(self, name):
                return [{"name": "LINEAR_API_KEY", "prompt": "API key", "required": True}]

        monkeypatch.setattr("tools.connectors.mcp._default_backend", _NoInstallBackend)
        monkeypatch.setattr(agent, "_get_session_db_for_recall", lambda: None)
        monkeypatch.setattr(
            agent,
            "_dispatch_delegate_task",
            lambda args: '{"ok":true}',
        )
        agent._memory_manager = None

        assert tool_name in AGENT_RUNTIME_POST_HOOK_TOOL_NAMES
        with patch(
            "model_tools.handle_function_call",
            side_effect=AssertionError("agent-runtime tools must stay inline"),
        ):
            agent._invoke_tool(
                tool_name,
                dict(tool_args),
                "task-concurrent",
                tool_call_id=f"{tool_name}-concurrent",
            )
            tool_call = _mock_tool_call(
                name=tool_name,
                arguments=json.dumps(tool_args),
                call_id=f"{tool_name}-sequential",
            )
            agent._execute_tool_calls_sequential(
                _mock_assistant_msg(content="", tool_calls=[tool_call]),
                [],
                "task-sequential",
            )

        post_calls = [
            kwargs
            for hook_name, kwargs in hook_calls
            if hook_name == "post_tool_call"
        ]
        assert [call["tool_call_id"] for call in post_calls] == [
            f"{tool_name}-concurrent",
            f"{tool_name}-sequential",
        ]
        assert all(call["tool_name"] == tool_name for call in post_calls)

class TestRuntimeToolTransformToolResult:
    """A registered ``transform_tool_result`` replaces what the model sees for an
    agent-runtime tool, on both the sequential and the concurrent executor path."""

    @staticmethod
    def _install_rewriting_transform(agent, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *args, **kwargs: (None, None),
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: (
                [f'REWRITTEN[{kwargs["tool_name"]}]{kwargs["result"]}']
                if hook_name == "transform_tool_result"
                else []
            ),
        )
        monkeypatch.setattr("tools.todo_tool.todo_tool", lambda **kwargs: '{"ok":true}')
        agent._memory_manager = None

    def test_concurrent_path_applies_transform(self, agent, monkeypatch):
        self._install_rewriting_transform(agent, monkeypatch)
        messages = []

        agent._execute_tool_calls_concurrent(
            _mock_assistant_msg(
                content="",
                tool_calls=[
                    _mock_tool_call(
                        name="todo_list", arguments=json.dumps({"todos": []}), call_id=call_id
                    )
                    for call_id in ("todo-c1", "todo-c2")
                ],
            ),
            messages,
            "task-concurrent",
        )

        tool_results = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_results] == ["todo-c1", "todo-c2"]
        # Exactly once per call: a second invocation would nest the prefix.
        assert [str(m["content"]) for m in tool_results] == ['REWRITTEN[todo_list]{"ok":true}'] * 2

    def test_sequential_path_applies_transform(self, agent, monkeypatch):
        self._install_rewriting_transform(agent, monkeypatch)
        messages = []

        agent._execute_tool_calls_sequential(
            _mock_assistant_msg(
                content="",
                tool_calls=[
                    _mock_tool_call(
                        name="todo_list",
                        arguments=json.dumps({"todos": []}),
                        call_id="todo-sequential",
                    )
                ],
            ),
            messages,
            "task-sequential",
        )

        tool_results = [m for m in messages if m.get("role") == "tool"]
        assert tool_results, "sequential path appended no tool result"
        # Exactly once: a second invocation would nest the prefix.
        assert str(tool_results[-1]["content"]) == 'REWRITTEN[todo_list]{"ok":true}'

class TestParallelScopePathNormalization:


    def test_should_parallelize_tool_batch_rejects_same_file_with_mixed_path_spellings(self, tmp_path, monkeypatch):
        from agent.tool_dispatch_helpers import _should_parallelize_tool_batch
        monkeypatch.chdir(tmp_path)
        tc1 = _mock_tool_call(name="write_file", arguments='{"path":"notes.txt","content":"one"}', call_id="c1")
        tc2 = _mock_tool_call(name="write_file", arguments=f'{{"path":"{tmp_path / "notes.txt"}","content":"two"}}', call_id="c2")

        assert not _should_parallelize_tool_batch([tc1, tc2])

class TestMcpParallelToolBatch:
    """Integration test: _should_parallelize_tool_batch respects MCP parallel flag."""

    def test_mcp_tools_default_sequential(self):
        """MCP tools without supports_parallel_tool_calls are sequential."""
        from agent.tool_dispatch_helpers import _should_parallelize_tool_batch
        tc1 = _mock_tool_call(name="mcp__github__list_repos", arguments='{"org":"openai"}', call_id="c1")
        tc2 = _mock_tool_call(name="mcp__github__search_code", arguments='{"q":"test"}', call_id="c2")
        assert not _should_parallelize_tool_batch([tc1, tc2])

    def test_mcp_tools_parallel_when_server_opted_in(self):
        """MCP tools from a parallel-safe server can run concurrently."""
        from agent.tool_dispatch_helpers import _should_parallelize_tool_batch
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


    def test_string_arguments_unchanged(self, agent):
        """String arguments must pass through without modification."""
        args_str = '{"query": "test"}'
        response = self._make_codex_response("function_call", args_str)
        msg, _ = _normalize_codex_response(response)
        tc = msg.tool_calls[0]
        assert tc.function.arguments == args_str
