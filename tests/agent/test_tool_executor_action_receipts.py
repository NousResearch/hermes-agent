"""Tool-executor action-receipt boundary and terminalization tests."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent import test_run_agent as _run_agent_tests

agent = _run_agent_tests.agent
_mock_assistant_msg = _run_agent_tests._mock_assistant_msg
_mock_tool_call = _run_agent_tests._mock_tool_call


class TestToolExecutorActionReceipts:
    def test_concurrent_timeout_after_dispatch_records_effective_metadata(
        self, agent, monkeypatch
    ):
        from hermes_cli.middleware import RequestMiddlewareResult

        callback_started = threading.Event()
        release = threading.Event()
        receipts = []
        hook_calls = []
        effective_args = {}
        trace = [{"source": "test-middleware"}]

        def _dispatch(_name, args, *_positional, **_kwargs):
            assert args == effective_args
            callback_started.set()
            release.wait()
            return "late result"

        tc = _mock_tool_call(name="web_search", arguments='{"q":"slow"}', call_id="c1")
        messages = []
        monkeypatch.setattr(
            "agent.tool_executor._resolve_concurrent_tool_timeout", lambda: 1.0
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: RequestMiddlewareResult(
                payload=effective_args,
                original_payload=args,
                changed=True,
                trace=trace,
            ),
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        try:
            with (
                patch("model_tools.handle_function_call", side_effect=_dispatch),
                patch(
                    "agent.tool_executor._maybe_record_action_receipt",
                    side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
                ),
            ):
                agent._execute_tool_calls_concurrent(
                    _mock_assistant_msg(content="", tool_calls=[tc]), messages, "task-1"
                )
        finally:
            release.set()

        assert callback_started.is_set()
        assert "timed out after 1.0s" in messages[0]["content"]
        assert len(receipts) == 1
        assert receipts[0]["tool_call_id"] == "c1"
        assert receipts[0]["exit_status"] == "timeout"
        assert receipts[0]["function_args"] == effective_args
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["args"] == effective_args
        assert post_calls[0]["middleware_trace"] == trace
        assert post_calls[0]["status"] == "timeout"

    def test_concurrent_timeout_suppresses_late_success_hook(self, agent, monkeypatch):
        from hermes_cli import lifecycle
        import model_tools

        callback_started = threading.Event()
        release = threading.Event()
        callback_returned = threading.Event()
        hook_calls = []

        def _dispatch(_name, _args, **_kwargs):
            callback_started.set()
            release.wait()
            callback_returned.set()
            return "late result"

        tc = _mock_tool_call(
            name="web_search",
            arguments='{"q":"slow"}',
            call_id="c1",
        )
        monkeypatch.setattr(
            "agent.tool_executor._resolve_concurrent_tool_timeout", lambda: 1.0
        )
        monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        try:
            with patch.object(model_tools.registry, "dispatch", side_effect=_dispatch):
                agent._execute_tool_calls_concurrent(
                    _mock_assistant_msg(content="", tool_calls=[tc]), [], "task-1"
                )
                assert callback_started.is_set()
                release.set()
                assert callback_returned.wait(timeout=1)
        finally:
            release.set()

        post_calls = [
            kwargs for name, kwargs in hook_calls if name == "post_tool_call"
        ]
        assert len(post_calls) == 1
        assert post_calls[0]["tool_call_id"] == "c1"
        assert post_calls[0]["status"] == "timeout"

    def test_concurrent_blocked_call_emits_one_terminal_hook(self, agent, monkeypatch):
        hook_calls = []
        tc = _mock_tool_call(
            name="web_search",
            arguments='{"q":"blocked"}',
            call_id="c1",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *_args, **_kwargs: ("Blocked by policy", None),
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        with patch(
            "model_tools.handle_function_call",
            side_effect=AssertionError("blocked tool must not dispatch"),
        ):
            agent._execute_tool_calls_concurrent(
                _mock_assistant_msg(content="", tool_calls=[tc]), [], "task-1"
            )

        post_calls = [
            kwargs for name, kwargs in hook_calls if name == "post_tool_call"
        ]
        assert len(post_calls) == 1
        assert post_calls[0]["tool_call_id"] == "c1"
        assert post_calls[0]["status"] == "blocked"
        assert post_calls[0]["error_type"] == "plugin_block"

    def test_concurrent_timeout_before_dispatch_prevents_late_callback(self, agent, monkeypatch):
        preflight_started = threading.Event()
        release = threading.Event()
        callback_calls = []
        callback_invoked = threading.Event()
        receipts = []

        def _preflight(*_args, **_kwargs):
            preflight_started.set()
            release.wait()
            return SimpleNamespace(allows_execution=True)

        agent._tool_guardrails.before_call = _preflight
        tc = _mock_tool_call(name="web_search", arguments='{"q":"slow"}', call_id="c1")
        monkeypatch.setattr(
            "agent.tool_executor._resolve_concurrent_tool_timeout", lambda: 1.0
        )
        try:
            with (
                patch(
                    "model_tools.handle_function_call",
                    side_effect=lambda *_args, **_kwargs: (
                        callback_calls.append("called"), callback_invoked.set()
                    )[0],
                ),
                patch(
                    "agent.tool_executor._maybe_record_action_receipt",
                    side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
                ),
            ):
                agent._execute_tool_calls_concurrent(
                    _mock_assistant_msg(content="", tool_calls=[tc]), [], "task-1"
                )
        finally:
            release.set()

        assert preflight_started.is_set()
        assert not callback_invoked.wait(timeout=0.1)
        assert callback_calls == []
        assert receipts == []

    def test_concurrent_keyboard_interrupt_records_effective_metadata(
        self, agent, monkeypatch
    ):
        from hermes_cli.middleware import RequestMiddlewareResult

        receipts = []
        hook_calls = []
        effective_args = {}
        trace = [{"source": "test-middleware"}]
        tc = _mock_tool_call(name="web_search", arguments='{"q":"stop"}', call_id="c1")
        messages = []
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: RequestMiddlewareResult(
                payload=effective_args,
                original_payload=args,
                changed=True,
                trace=trace,
            ),
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        def _dispatch(_name, args, *_positional, **_kwargs):
            assert args == effective_args
            raise KeyboardInterrupt

        with (
            patch("model_tools.handle_function_call", side_effect=_dispatch),
            patch(
                "agent.tool_executor._maybe_record_action_receipt",
                side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
            ),
        ):
            agent._execute_tool_calls_concurrent(
                _mock_assistant_msg(content="", tool_calls=[tc]), messages, "task-1"
            )

        assert "cancelled" in messages[0]["content"].lower()
        assert len(receipts) == 1
        assert receipts[0]["tool_call_id"] == "c1"
        assert receipts[0]["exit_status"] == "cancelled"
        assert receipts[0]["function_args"] == effective_args
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["args"] == effective_args
        assert post_calls[0]["middleware_trace"] == trace
        assert post_calls[0]["status"] == "cancelled"

    def test_concurrent_started_exception_records_effective_metadata(
        self, agent, monkeypatch
    ):
        from hermes_cli.middleware import RequestMiddlewareResult

        receipts = []
        hook_calls = []
        effective_args = {}
        trace = [{"source": "test-middleware"}]
        tc = _mock_tool_call(name="web_search", arguments='{"q":"fail"}', call_id="c1")
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: RequestMiddlewareResult(
                payload=effective_args,
                original_payload=args,
                changed=True,
                trace=trace,
            ),
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        def _dispatch(_name, args, *_positional, **_kwargs):
            assert args == effective_args
            raise RuntimeError("boom")

        with (
            patch("model_tools.handle_function_call", side_effect=_dispatch),
            patch(
                "agent.tool_executor._maybe_record_action_receipt",
                side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
            ),
        ):
            agent._execute_tool_calls_concurrent(
                _mock_assistant_msg(content="", tool_calls=[tc]), [], "task-1"
            )

        assert len(receipts) == 1
        assert receipts[0]["tool_call_id"] == "c1"
        assert receipts[0]["exit_status"] == "error"
        assert receipts[0]["function_args"] == effective_args
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["tool_call_id"] == "c1"
        assert post_calls[0]["args"] == effective_args
        assert post_calls[0]["middleware_trace"] == trace
        assert post_calls[0]["status"] == "error"

    def test_sequential_started_exception_records_error_receipt(self, agent):
        receipts = []
        tc = _mock_tool_call(name="web_search", arguments='{"q":"fail"}', call_id="c1")

        with (
            patch("model_tools.handle_function_call", side_effect=RuntimeError("boom")),
            patch(
                "agent.tool_executor._maybe_record_action_receipt",
                side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
            ),
        ):
            agent._execute_tool_calls_sequential(
                _mock_assistant_msg(content="", tool_calls=[tc]), [], "task-1"
            )

        assert len(receipts) == 1
        assert receipts[0]["tool_call_id"] == "c1"
        assert receipts[0]["exit_status"] == "error"

    def test_sequential_inline_started_exception_terminalizes_once(
        self, agent, monkeypatch
    ):
        receipts = []
        hook_calls = []
        tc = _mock_tool_call(
            name="todo_list",
            arguments='{"todos":[]}',
            call_id="c1",
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        with (
            patch("tools.todo_tool.todo_tool", side_effect=RuntimeError("boom")),
            patch(
                "agent.tool_executor._maybe_record_action_receipt",
                side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
            ),
        ):
            messages = []
            agent._execute_tool_calls_sequential(
                _mock_assistant_msg(content="", tool_calls=[tc]),
                messages,
                "task-1",
            )

        assert "boom" in messages[0]["content"]
        assert len(receipts) == 1
        assert receipts[0]["exit_status"] == "error"
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["status"] == "error"

    @pytest.mark.parametrize(
        ("tool_name", "error_prefix"),
        (
            ("context_lookup", "Context engine tool 'context_lookup' failed: boom"),
            ("memory_lookup", "Memory tool 'memory_lookup' failed: boom"),
        ),
    )
    def test_sequential_specialized_started_exception_preserves_error_contract(
        self,
        agent,
        monkeypatch,
        tool_name,
        error_prefix,
    ):
        receipts = []
        hook_calls = []
        tool_call = _mock_tool_call(name=tool_name, arguments="{}", call_id="c1")
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        if tool_name == "context_lookup":
            agent._context_engine_tool_names = {tool_name}
            agent.context_compressor.handle_tool_call = MagicMock(
                side_effect=RuntimeError("boom")
            )
        else:
            agent._context_engine_tool_names = set()
            agent._memory_manager = SimpleNamespace(
                has_tool=lambda name: name == tool_name,
                handle_tool_call=MagicMock(side_effect=RuntimeError("boom")),
            )

        with patch(
            "agent.tool_executor._maybe_record_action_receipt",
            side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
        ):
            messages = []
            agent._execute_tool_calls_sequential(
                _mock_assistant_msg(content="", tool_calls=[tool_call]),
                messages,
                "task-1",
            )

        assert json.loads(messages[0]["content"])["error"] == error_prefix
        assert len(receipts) == 1
        assert receipts[0]["exit_status"] == "error"
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert json.loads(post_calls[0]["result"])["error"] == error_prefix

    @pytest.mark.parametrize("mode", ["sequential", "concurrent"])
    def test_heartbeat_setup_failure_has_no_execution_receipt(self, agent, mode):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        with (
            patch("agent.tool_executor._run_with_activity_heartbeat", side_effect=RuntimeError("thread setup failed")),
            patch("model_tools.handle_function_call") as callback,
            patch("agent.tool_executor._maybe_record_action_receipt") as receipt,
        ):
            getattr(agent, "_execute_tool_calls_" + mode)(
                _mock_assistant_msg(content="", tool_calls=[tc]), [], "task-1",
            )
        callback.assert_not_called()
        receipt.assert_not_called()

    @pytest.mark.parametrize("quiet_mode", [True, False])
    def test_sequential_preflight_keyboard_interrupt_has_no_execution_receipt(
        self, agent, monkeypatch, quiet_mode
    ):
        receipts = []
        hook_calls = []
        agent.quiet_mode = quiet_mode
        agent._tool_guardrails.before_call = MagicMock(side_effect=KeyboardInterrupt)
        tc = _mock_tool_call(
            name="web_search",
            arguments='{"q":"stop"}',
            call_id="c1",
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        with (
            patch("model_tools.handle_function_call") as callback,
            patch(
                "agent.tool_executor._maybe_record_action_receipt",
                side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            agent._execute_tool_calls_sequential(
                _mock_assistant_msg(content="", tool_calls=[tc]),
                [],
                "task-1",
            )

        callback.assert_not_called()
        assert receipts == []
        assert [
            kwargs for name, kwargs in hook_calls if name == "post_tool_call"
        ] == []

    @pytest.mark.parametrize("quiet_mode", [True, False])
    def test_sequential_started_interrupt_uses_effective_middleware_metadata(
        self, agent, monkeypatch, quiet_mode
    ):
        from hermes_cli.middleware import RequestMiddlewareResult

        receipts = []
        hook_calls = []
        trace = [{"source": "test-middleware"}]
        effective_args = {}
        agent.quiet_mode = quiet_mode
        tc = _mock_tool_call(
            name="web_search",
            arguments='{"q":"original"}',
            call_id="c1",
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: RequestMiddlewareResult(
                payload=effective_args,
                original_payload=args,
                changed=True,
                trace=trace,
            ),
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )

        def interrupting_callback(_name, args, *_positional, **_kwargs):
            assert args == effective_args
            raise KeyboardInterrupt

        with (
            patch("model_tools.handle_function_call", side_effect=interrupting_callback),
            patch(
                "agent.tool_executor._maybe_record_action_receipt",
                side_effect=lambda *_args, **kwargs: receipts.append(kwargs),
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            agent._execute_tool_calls_sequential(
                _mock_assistant_msg(content="", tool_calls=[tc]),
                [],
                "task-1",
            )

        assert len(receipts) == 1
        assert receipts[0]["function_args"] == effective_args
        assert receipts[0]["exit_status"] == "cancelled"
        post_calls = [kwargs for name, kwargs in hook_calls if name == "post_tool_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["args"] == effective_args
        assert post_calls[0]["middleware_trace"] == trace
        assert post_calls[0]["status"] == "cancelled"

