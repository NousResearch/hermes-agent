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





class TestCodexItemToArgs:

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





class TestCodexItemCompletionPayload:
    def test_command_success_returns_aggregated_output(self):
        result, is_error = _codex_item_completion_payload({
            "type": "commandExecution",
            "exitCode": 0,
            "aggregatedOutput": "hello\nworld\n",
        })
        assert result == "hello\nworld\n"
        assert is_error is False



    def test_mcp_tool_error_is_error(self):
        result, is_error = _codex_item_completion_payload({
            "type": "mcpToolCall",
            "error": {"message": "nope"},
        })
        assert "[error]" in result
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




class TestAgentMessageInterimDispatch:
    def test_completed_agent_message_alone_is_not_emitted_as_interim(self):
        """A lone completed agentMessage is usually the final answer.

        Emitting it through the interim path duplicates the gateway's normal
        final send on Discord when codex_app_server is active.
        """
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-1",
            "text": "I'll check the config first.",
        }))
        agent._emit_interim_assistant_message.assert_not_called()

    def test_cleared_completed_agent_message_does_not_flush_next_turn(self):
        """A reused app-server bridge must not carry a final answer into the
        next turn's first stream/tool event."""
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-1",
            "text": "Final answer from the previous turn.",
        }))

        bridge.clear_pending_agent_message()
        bridge({"method": "item/agentMessage/delta",
                "params": {"delta": "New turn starts."}})

        agent._emit_interim_assistant_message.assert_not_called()
        agent._fire_stream_delta.assert_called_once_with("New turn starts.")

    def test_completed_agent_message_flushes_before_tool_event(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-1",
            "text": "I'll check the config first.",
        }))
        bridge(_item_started({
            "type": "commandExecution",
            "id": "exec-1",
            "command": "ls",
        }))
        agent._emit_interim_assistant_message.assert_called_once_with(
            {"role": "assistant", "content": "I'll check the config first."}
        )

    def test_completed_agent_message_flushes_before_stream_delta(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-1",
            "text": "I'll check the config first.",
        }))
        bridge({"method": "item/agentMessage/delta",
                "params": {"delta": "final"}})
        agent._emit_interim_assistant_message.assert_called_once_with(
            {"role": "assistant", "content": "I'll check the config first."}
        )
        agent._fire_stream_delta.assert_called_once_with("final")

    def test_second_completed_agent_message_flushes_previous_only(self):
        agent = _make_stub_agent()
        bridge = make_codex_app_server_event_bridge(agent)
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-1",
            "text": "First interim.",
        }))
        bridge(_item_completed({
            "type": "agentMessage",
            "id": "am-2",
            "text": "Final answer.",
        }))
        agent._emit_interim_assistant_message.assert_called_once_with(
            {"role": "assistant", "content": "First interim."}
        )


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
        bridge(_item_started({
            "type": "commandExecution", "id": "cmd-1", "command": "ls",
        }))
        agent._emit_interim_assistant_message.assert_not_called()
        # Tool progress is unaffected by the commentary toggle.
        agent.tool_progress_callback.assert_called_once()


class TestBridgeRobustness:

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

    def test_reused_session_clears_buffered_final_answer_between_turns(
        self, monkeypatch
    ):
        from agent import codex_runtime
        from agent.transports.codex_app_server_session import TurnResult

        class FakeSession:
            def __init__(self, **kwargs):
                self._on_event = kwargs["on_event"]
                self.calls = 0

            def run_turn(self, user_input, **_):
                self.calls += 1
                if self.calls == 1:
                    self._on_event(_item_completed({
                        "type": "agentMessage",
                        "id": "am-final-1",
                        "text": "Previous final answer.",
                    }))
                    return TurnResult(
                        final_text="Previous final answer.",
                        projected_messages=[{
                            "role": "assistant",
                            "content": "Previous final answer.",
                        }],
                        turn_id="turn-1",
                        thread_id="thread-1",
                    )

                self._on_event(_item_started({
                    "type": "commandExecution",
                    "id": "exec-2",
                    "command": "pwd",
                }))
                return TurnResult(
                    final_text="Next final answer.",
                    projected_messages=[{
                        "role": "assistant",
                        "content": "Next final answer.",
                    }],
                    turn_id="turn-2",
                    thread_id="thread-1",
                )

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.codex_app_server_session.CodexAppServerSession",
            FakeSession,
        )

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

        for text in ("first", "second"):
            codex_runtime.run_codex_app_server_turn(
                agent,
                user_message=text,
                original_user_message=text,
                messages=[],
                effective_task_id="task",
            )

        agent._emit_interim_assistant_message.assert_not_called()
        agent.tool_progress_callback.assert_called_once()
        assert agent.tool_progress_callback.call_args.args[0] == "tool.started"
        assert agent.tool_progress_callback.call_args.args[1] == "exec_command"
