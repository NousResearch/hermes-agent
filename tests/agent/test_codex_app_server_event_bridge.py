"""Regression tests for the codex_app_server → Hermes UI event bridge.

Pin the translation of codex JSON-RPC notifications into agent callbacks
(`tool_progress_callback`, `_fire_stream_delta`, `_fire_reasoning_delta`,
`_emit_interim_assistant_message`) so Discord/Telegram/TUI continue to
surface live tool-progress bubbles and interim assistant commentary when
the active provider runs on `openai_runtime: codex_app_server` (#33200).

Each test drives `make_codex_app_server_event_bridge(agent)` directly with
fixture notifications that mirror codex 0.130.0's `item/*` shape and
asserts the right agent callback fired with the right arguments. The
bridge is deliberately small (~150 lines of pure dict mapping) so the
tests can stay focused on the wire format ↔ callback contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import (
    _codex_item_completion_payload,
    _codex_item_to_args,
    _codex_item_to_preview,
    _codex_item_to_tool_name,
    make_codex_app_server_event_bridge,
)


def _make_stub_agent() -> SimpleNamespace:
    """Minimal stand-in for AIAgent that records every callback fire."""
    return SimpleNamespace(
        tool_progress_callback=MagicMock(name="tool_progress_callback"),
        _fire_stream_delta=MagicMock(name="_fire_stream_delta"),
        _fire_reasoning_delta=MagicMock(name="_fire_reasoning_delta"),
        _emit_interim_assistant_message=MagicMock(
            name="_emit_interim_assistant_message"
        ),
    )


def _item_started(item: dict) -> dict:
    return {"method": "item/started", "params": {"item": item}}


def _item_completed(item: dict) -> dict:
    return {"method": "item/completed", "params": {"item": item}}


# ---------- name / args / preview / result mapping ----------


class TestCodexItemToToolName:
    def test_command_execution_maps_to_exec_command(self):
        assert _codex_item_to_tool_name(
            {"type": "commandExecution"}
        ) == "exec_command"

    def test_file_change_maps_to_apply_patch(self):
        assert _codex_item_to_tool_name(
            {"type": "fileChange"}
        ) == "apply_patch"

    def test_mcp_tool_call_includes_server_and_tool(self):
        assert _codex_item_to_tool_name(
            {"type": "mcpToolCall", "server": "fs", "tool": "read_file"}
        ) == "mcp.fs.read_file"

    def test_mcp_tool_call_falls_back_when_fields_missing(self):
        assert _codex_item_to_tool_name(
            {"type": "mcpToolCall"}
        ) == "mcp.mcp.unknown"

    def test_dynamic_tool_call_uses_tool_field(self):
        assert _codex_item_to_tool_name(
            {"type": "dynamicToolCall", "tool": "web_search"}
        ) == "web_search"

    def test_hermes_tools_mcp_server_emits_bare_tool_name(self):
        """The hermes-tools MCP server wraps Hermes' own tools for codex;
        the inner dispatch subprocess can't fire native progress events,
        so the codex-level event IS the display event — shown without the
        mcp.hermes-tools.* namespacing (from #26541 by @simpolism)."""
        assert _codex_item_to_tool_name(
            {"type": "mcpToolCall", "server": "hermes-tools", "tool": "web_search"}
        ) == "web_search"
        assert _codex_item_to_tool_name(
            {"type": "mcpToolCall", "server": "hermes-tools", "tool": "browser_navigate"}
        ) == "browser_navigate"

    def test_web_search_builtin_maps_to_web_search(self):
        """Codex's built-in webSearch tool gets a bubble too (#26541)."""
        assert _codex_item_to_tool_name({"type": "webSearch"}) == "web_search"

    def test_unknown_type_returns_type_string(self):
        assert _codex_item_to_tool_name(
            {"type": "plan"}
        ) == "plan"

    def test_missing_type_returns_unknown_sentinel(self):
        assert _codex_item_to_tool_name({}) == "unknown"


class TestCodexItemToArgs:
    def test_command_execution_args_carry_cwd_and_command(self):
        args = _codex_item_to_args({
            "type": "commandExecution",
            "command": "ls -la",
            "cwd": "/tmp",
        })
        assert args == {"command": "ls -la", "cwd": "/tmp"}

    def test_file_change_args_normalize_changes(self):
        args = _codex_item_to_args({
            "type": "fileChange",
            "changes": [
                {"path": "/a.py", "kind": {"type": "add"}},
                {"path": "/b.py", "kind": {"type": "update"}},
                "not-a-dict",
            ],
        })
        assert args == {
            "changes": [
                {"kind": "add", "path": "/a.py"},
                {"kind": "update", "path": "/b.py"},
            ]
        }

    def test_mcp_tool_call_returns_arguments_dict(self):
        args = _codex_item_to_args({
            "type": "mcpToolCall", "arguments": {"q": "x"}
        })
        assert args == {"q": "x"}

    def test_non_dict_arguments_get_wrapped(self):
        args = _codex_item_to_args({
            "type": "dynamicToolCall", "arguments": ["a", "b"],
        })
        assert args == {"arguments": ["a", "b"]}


class TestCodexItemToPreview:
    def test_command_preview_truncated(self):
        long_cmd = "echo " + "x" * 500
        preview = _codex_item_to_preview({
            "type": "commandExecution", "command": long_cmd
        })
        assert preview is not None
        assert len(preview) <= 120

    def test_file_change_preview_lists_first_three_paths(self):
        preview = _codex_item_to_preview({
            "type": "fileChange",
            "changes": [
                {"path": f"/p{i}.py", "kind": {"type": "update"}}
                for i in range(5)
            ],
        })
        assert "/p0.py" in preview and "/p2.py" in preview
        assert "+2 more" in preview

    def test_file_change_no_paths_returns_none(self):
        assert _codex_item_to_preview({
            "type": "fileChange", "changes": [{}]
        }) is None

    def test_mcp_args_preview_is_json(self):
        preview = _codex_item_to_preview({
            "type": "mcpToolCall", "arguments": {"q": "hello"},
        })
        assert preview is not None
        assert "hello" in preview

    def test_empty_args_returns_none(self):
        assert _codex_item_to_preview({"type": "mcpToolCall"}) is None


class TestCodexItemCompletionPayload:
    def test_command_success_returns_aggregated_output(self):
        result, is_error = _codex_item_completion_payload({
            "type": "commandExecution",
            "exitCode": 0,
            "aggregatedOutput": "hello\nworld\n",
        })
        assert result == "hello\nworld\n"
        assert is_error is False

    def test_command_nonzero_exit_marks_error(self):
        result, is_error = _codex_item_completion_payload({
            "type": "commandExecution",
            "exitCode": 2,
            "aggregatedOutput": "boom",
        })
        assert "[exit 2]" in result
        assert "boom" in result
        assert is_error is True

    def test_file_change_completed_status_not_error(self):
        result, is_error = _codex_item_completion_payload({
            "type": "fileChange",
            "status": "completed",
            "changes": [{"path": "/a"}],
        })
        assert "completed" in result
        assert is_error is False

    def test_mcp_tool_error_is_error(self):
        result, is_error = _codex_item_completion_payload({
            "type": "mcpToolCall",
            "error": {"message": "nope"},
        })
        assert "[error]" in result
        assert is_error is True

    def test_dynamic_tool_failure_is_error(self):
        result, is_error = _codex_item_completion_payload({
            "type": "dynamicToolCall",
            "success": False,
        })
        assert "False" in result
        assert is_error is True


# ---------- bridge: dispatch contracts ----------


class TestStreamDeltaDispatch:
    def test_agent_message_delta_fires_stream_delta(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge({"method": "item/agentMessage/delta",
                "params": {"delta": "hello "}})
        bridge({"method": "item/agentMessage/delta",
                "params": {"delta": "world"}})
        assert agent._fire_stream_delta.call_count == 2
        assert agent._fire_stream_delta.call_args_list[0].args == ("hello ",)
        assert agent._fire_stream_delta.call_args_list[1].args == ("world",)

    def test_empty_delta_is_skipped(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge({"method": "item/agentMessage/delta", "params": {"delta": ""}})
        bridge({"method": "item/agentMessage/delta", "params": {}})
        agent._fire_stream_delta.assert_not_called()

    def test_text_field_used_when_delta_missing(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge({"method": "item/agentMessage/delta",
                "params": {"text": "fallback"}})
        agent._fire_stream_delta.assert_called_once_with("fallback")

    def test_reasoning_delta_fires_reasoning_callback(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge({"method": "item/reasoning/delta",
                "params": {"delta": "thinking..."}})
        agent._fire_reasoning_delta.assert_called_once_with("thinking...")
        agent._fire_stream_delta.assert_not_called()


class TestToolProgressDispatch:
    def test_command_started_fires_tool_started(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "commandExecution",
            "id": "exec-1",
            "command": "ls /tmp",
            "cwd": "/tmp",
        }))
        agent.tool_progress_callback.assert_called_once()
        call = agent.tool_progress_callback.call_args
        assert call.args[0] == "tool.started"
        assert call.args[1] == "exec_command"
        assert "ls /tmp" in call.args[2]  # preview
        assert call.args[3] == {"command": "ls /tmp", "cwd": "/tmp"}

    def test_command_completed_fires_tool_completed_with_result(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "commandExecution",
            "id": "exec-2",
            "command": "echo hi",
            "cwd": "/tmp",
        }))
        bridge(_item_completed({
            "type": "commandExecution",
            "id": "exec-2",
            "exitCode": 0,
            "aggregatedOutput": "hi\n",
            "durationMs": 42,
        }))
        # tool.started then tool.completed
        assert agent.tool_progress_callback.call_count == 2
        completed = agent.tool_progress_callback.call_args_list[1]
        assert completed.args[0] == "tool.completed"
        assert completed.args[1] == "exec_command"
        assert completed.args[2] is None  # preview unused on completion
        assert completed.args[3] is None  # args unused on completion
        assert completed.kwargs["duration"] == pytest.approx(0.042)
        assert completed.kwargs["is_error"] is False
        assert completed.kwargs["result"] == "hi\n"

    def test_nonzero_exit_marks_completion_error(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "commandExecution",
            "id": "exec-3",
            "exitCode": 127,
            "aggregatedOutput": "not found",
        }))
        call = agent.tool_progress_callback.call_args
        assert call.args[0] == "tool.completed"
        assert call.kwargs["is_error"] is True
        assert "[exit 127]" in call.kwargs["result"]

    def test_apply_patch_started_and_completed(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "fileChange",
            "id": "fc-1",
            "changes": [
                {"path": "/a.py", "kind": {"type": "add"}},
                {"path": "/b.py", "kind": {"type": "update"}},
            ],
        }))
        bridge(_item_completed({
            "type": "fileChange",
            "id": "fc-1",
            "status": "completed",
            "changes": [{"path": "/a.py"}, {"path": "/b.py"}],
        }))
        names = [
            c.args[1] for c in agent.tool_progress_callback.call_args_list
        ]
        assert names == ["apply_patch", "apply_patch"]
        completed = agent.tool_progress_callback.call_args_list[1]
        assert completed.kwargs["is_error"] is False
        assert "2 change(s)" in completed.kwargs["result"]

    def test_mcp_tool_uses_namespaced_tool_name(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "mcpToolCall",
            "id": "mcp-1",
            "server": "fs",
            "tool": "list_dir",
            "arguments": {"path": "/tmp"},
        }))
        call = agent.tool_progress_callback.call_args
        assert call.args[1] == "mcp.fs.list_dir"
        # Preview should be a json render of the args
        assert "/tmp" in call.args[2]

    def test_dynamic_tool_uses_tool_field_as_name(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "dynamicToolCall",
            "id": "dyn-1",
            "tool": "web_search",
            "arguments": {"query": "hermes"},
        }))
        bridge(_item_completed({
            "type": "dynamicToolCall",
            "id": "dyn-1",
            "tool": "web_search",
            "success": True,
            "contentItems": [{"text": "results"}],
        }))
        names = [
            c.args[1] for c in agent.tool_progress_callback.call_args_list
        ]
        assert names == ["web_search", "web_search"]
        completed = agent.tool_progress_callback.call_args_list[1]
        assert completed.kwargs["is_error"] is False
        assert "results" in completed.kwargs["result"]

    def test_web_search_builtin_fires_started_and_completed(self):
        """Codex's built-in webSearch produces a start/complete bubble pair
        with the query as preview and args (#26541)."""
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "webSearch",
            "id": "ws-1",
            "query": "hermes agent docs",
        }))
        bridge(_item_completed({
            "type": "webSearch",
            "id": "ws-1",
            "query": "hermes agent docs",
        }))
        calls = agent.tool_progress_callback.call_args_list
        assert [c.args[0] for c in calls] == ["tool.started", "tool.completed"]
        assert calls[0].args[1] == "web_search"
        assert calls[0].args[2] == "hermes agent docs"
        assert calls[0].args[3] == {"query": "hermes agent docs"}

    def test_duration_falls_back_to_wall_time_when_codex_missing_ms(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "commandExecution",
            "id": "exec-4",
            "command": "sleep 0",
        }))
        bridge(_item_completed({
            "type": "commandExecution",
            "id": "exec-4",
            "exitCode": 0,
            "aggregatedOutput": "",
            # no durationMs
        }))
        completed = agent.tool_progress_callback.call_args_list[1]
        assert completed.kwargs["duration"] is not None
        assert completed.kwargs["duration"] >= 0

    def test_unknown_started_item_type_is_silent(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({"type": "plan", "id": "p-1"}))
        agent.tool_progress_callback.assert_not_called()


class TestAgentMessageInterimDispatch:
    def test_completed_agent_message_emits_interim(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-1",
            "text": "I'll check the config first.",
        }))
        agent._emit_interim_assistant_message.assert_called_once_with(
            {"role": "assistant", "content": "I'll check the config first."}
        )

    def test_empty_text_does_not_emit_interim(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage", "id": "am-2", "text": "   ",
        }))
        bridge(_item_completed({
            "type": "agentMessage", "id": "am-3", "text": ""
        }))
        agent._emit_interim_assistant_message.assert_not_called()

    def test_completed_agent_message_does_not_fire_tool_progress(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage", "id": "am-4", "text": "hi",
        }))
        agent.tool_progress_callback.assert_not_called()

    def test_show_commentary_off_suppresses_interim(self):
        """display.show_commentary=false silences agentMessage interim
        delivery on the app-server runtime (same contract as the
        codex_responses commentary channel)."""
        agent = _make_stub_agent()
        agent.show_commentary = False
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage", "id": "am-5", "text": "I'll check config.",
        }))
        agent._emit_interim_assistant_message.assert_not_called()
        # Tool progress is unaffected by the commentary toggle.
        bridge(_item_started({
            "type": "commandExecution", "id": "cmd-1", "command": "ls",
        }))
        agent.tool_progress_callback.assert_called_once()


class TestBridgeRobustness:
    def test_non_dict_notification_is_ignored(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge("not-a-dict")  # type: ignore[arg-type]
        bridge(None)  # type: ignore[arg-type]
        bridge(123)  # type: ignore[arg-type]
        agent.tool_progress_callback.assert_not_called()
        agent._fire_stream_delta.assert_not_called()
        agent._emit_interim_assistant_message.assert_not_called()

    def test_missing_params_is_ignored(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge({"method": "item/started"})
        bridge({"method": "item/completed"})
        agent.tool_progress_callback.assert_not_called()

    def test_callback_exceptions_do_not_propagate(self):
        # Buggy display callbacks must not tear down the codex turn loop.
        agent = _make_stub_agent()
        agent.tool_progress_callback.side_effect = RuntimeError("boom")
        agent._fire_stream_delta.side_effect = RuntimeError("boom")
        agent._emit_interim_assistant_message.side_effect = RuntimeError("boom")
        bridge = make_codex_app_server_event_bridge(agent)
        # All three paths must swallow exceptions silently.
        bridge(_item_started({
            "type": "commandExecution", "id": "exec-x", "command": "ls",
        }))
        bridge({"method": "item/agentMessage/delta",
                "params": {"delta": "x"}})
        bridge(_item_completed({
            "type": "agentMessage", "id": "am-x", "text": "hi",
        }))

    def test_agent_without_callbacks_is_a_noop(self):
        # Mirrors gateway-less / cron contexts where the agent never had
        # the display callbacks set. Bridge must not raise.
        agent = SimpleNamespace()  # bare — none of the callbacks exist
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_started({
            "type": "commandExecution", "id": "exec-y", "command": "ls",
        }))
        bridge(_item_completed({
            "type": "commandExecution", "id": "exec-y",
            "exitCode": 0, "aggregatedOutput": "",
        }))
        bridge({"method": "item/agentMessage/delta",
                "params": {"delta": "x"}})
        bridge({"method": "item/reasoning/delta",
                "params": {"delta": "x"}})
        bridge(_item_completed({
            "type": "agentMessage", "id": "am", "text": "hi",
        }))

    def test_silent_methods_do_not_fire_anything(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        for method in ("turn/started", "turn/completed", "thread/started",
                       "item/commandExecution/outputDelta"):
            bridge({"method": method, "params": {}})
        agent.tool_progress_callback.assert_not_called()
        agent._fire_stream_delta.assert_not_called()
        agent._fire_reasoning_delta.assert_not_called()
        agent._emit_interim_assistant_message.assert_not_called()


# ---------- end-to-end: bridge is wired in run_codex_app_server_turn ----------


class TestBridgeWiredInRuntime:
    """Verify run_codex_app_server_turn actually constructs the session
    with `on_event=<bridge>`. This is the integration guard that prevents
    a future refactor from dropping the bridge wiring and silently
    regressing Discord/Telegram live progress visibility."""

    def test_session_constructor_receives_on_event(self, monkeypatch):
        from agent import codex_runtime

        captured: dict = {}

        class FakeSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input, **_):
                from agent.transports.codex_app_server_session import TurnResult
                return TurnResult(
                    final_text="done",
                    projected_messages=[],
                    tool_iterations=0,
                    turn_id="t1",
                    thread_id="th1",
                )

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.codex_app_server_session.CodexAppServerSession",
            FakeSession,
        )

        # Minimal stub agent — the runtime only touches a handful of
        # attributes and we mock the heavy ones to keep the test fast.
        agent = SimpleNamespace(
            session_cwd=None,
            _codex_session=None,
            tool_progress_callback=MagicMock(),
            _fire_stream_delta=MagicMock(),
            _fire_reasoning_delta=MagicMock(),
            _emit_interim_assistant_message=MagicMock(),
            _iters_since_skill=0,
            _skill_nudge_interval=0,
            valid_tool_names=set(),
            _sync_external_memory_for_turn=lambda **_: None,
            _spawn_background_review=lambda **_: None,
            # Usage accounting attrs read by _record_codex_app_server_usage.
            session_api_calls=0,
            session_prompt_tokens=0,
            session_completion_tokens=0,
            session_reasoning_tokens=0,
            session_cached_tokens=0,
            session_total_tokens=0,
            context_compressor=None,
            event_callback=None,
            _session_db=None,
        )

        codex_runtime.run_codex_app_server_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t",
        )

        assert "on_event" in captured, (
            "run_codex_app_server_turn must pass on_event=<bridge> to the "
            "session — without it the gateway sees no live progress (#33200)"
        )
        assert callable(captured["on_event"]), (
            "on_event must be the bridge callable, not None or a sentinel"
        )

        # And the bridge must actually drive the agent's callbacks when
        # fed a representative notification.
        captured["on_event"](_item_started({
            "type": "commandExecution",
            "id": "wired-1",
            "command": "ls",
        }))
        agent.tool_progress_callback.assert_called_once()
        assert agent.tool_progress_callback.call_args.args[0] == "tool.started"
        assert agent.tool_progress_callback.call_args.args[1] == "exec_command"
