"""A final-response plugin can rework once without exposing the rejected answer."""
from types import SimpleNamespace
from unittest.mock import Mock

from agent.turn_final_response import finish_text_response
from agent.stream_delivery import StreamDeliveryMixin
from hermes_cli import plugins


def agent():
    a = SimpleNamespace(
        session_id="s", _current_task_id="task", _current_turn_id="turn", platform="cli", model="test",
        _has_content_after_think_block=lambda s: bool(s), _emit_pending_fallback_notice=lambda: None,
        _clear_status_buffer=lambda: None, _stall_guards=False, valid_tool_names=[], api_mode="chat_completions",
        _strip_think_blocks=lambda s: s, _build_assistant_message=lambda m, _: {"role": "assistant", "content": m.content},
        _flush_messages_to_session_db=Mock(), quiet_mode=True, _interrupt_requested=False,
        _interim_content_was_streamed=lambda _: False, max_iterations=10,
        iteration_budget=SimpleNamespace(remaining=8),
    )
    return a


def finish(a, text, messages):
    return finish_text_response(a, assistant_message=SimpleNamespace(content=text, tool_calls=[]),
        response=None, finish_reason="stop", messages=messages, api_messages=[], conversation_history=[],
        api_call_count=1, user_message="do work", active_system_prompt="unchanged", final_response=None,
        _turn_exit_reason=None, _preflight_compression_blocked=False, codex_ack_continuations=0,
        truncated_response_parts=[], length_continue_retries=0, _pending_verification_response=None,
        _pending_verification_response_previewed=False, effective_task_id="task")


def test_hook_blocks_once_with_narrow_callback_and_no_premature_flush(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    calls = []
    def hook(final_response):
        calls.append(final_response)
        return {"action": "block", "message": "Missing result file"}
    assert "before_turn_end" in plugins.VALID_HOOKS, "missing generic continuation hook"
    manager._hooks["before_turn_end"] = [hook]
    a = agent()
    messages = [{"role": "user", "content": "do work"}]
    first = finish(a, "file ready", messages)
    assert first.action == "continue"
    assert first.active_system_prompt == "unchanged"
    assert "Missing result file" in messages[-1]["content"]
    a._flush_messages_to_session_db.assert_not_called()
    second = finish(a, "corrected", messages)
    assert second.action == "break"
    assert second.final_response == "corrected"
    assert calls == ["file ready", "corrected"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_deferred_text_never_reaches_stream_or_interim(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    assert "before_turn_end" in plugins.VALID_HOOKS
    manager._hooks["before_turn_end"] = [lambda **_: None]
    a = StreamDeliveryMixin()
    a.stream_delta_callback = Mock()
    a._stream_callback = Mock()
    a.interim_assistant_callback = Mock()
    a._strip_think_blocks = lambda s: s
    a._deliver_to_stream_callbacks("unverified text")
    a._deliver_interim("unverified", already_streamed=False, record=[])
    a.stream_delta_callback.assert_not_called()
    a._stream_callback.assert_not_called()
    a.interim_assistant_callback.assert_not_called()


def test_hook_receives_current_source_identity_and_turn_messages(monkeypatch):
    from agent.turn_end_hooks import before_turn_end
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    captured = {}
    manager._hooks["before_turn_end"] = [lambda **kwargs: captured.update(kwargs)]
    a = agent()
    a._current_source_identity = {"platform": "slack", "channel_id": "C123", "request_id": "1.2"}
    messages = [
        {"role": "user", "content": "old"},
        {"role": "tool", "content": "old receipt"},
        {"role": "user", "content": "decision"},
        {"role": "tool", "content": "receipt"},
    ]
    assert before_turn_end(a, "done", {}, messages, user_message="decision", can_continue=True) is False
    assert captured["source_identity"] == a._current_source_identity
    assert captured["messages"] == messages[3:]


def test_hook_keeps_same_turn_actions_across_its_synthetic_repair(monkeypatch):
    from agent.turn_end_hooks import before_turn_end
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    captured = {}
    manager._hooks["before_turn_end"] = [lambda **kwargs: captured.update(kwargs)]
    a = agent()
    messages = [
        {"role": "user", "content": "decision"},
        {"role": "tool", "content": "action"},
        {"role": "assistant", "content": "rejected", "_turn_end_synthetic": True},
        {"role": "user", "content": "repair", "_turn_end_synthetic": True},
        {"role": "tool", "content": "receipt"},
    ]
    assert before_turn_end(a, "done", {}, messages, user_message="decision", can_continue=True) is False
    assert captured["messages"] == messages[1:]


def test_hook_keeps_same_turn_evidence_across_all_runtime_user_scaffolding(monkeypatch):
    from agent.turn_end_hooks import before_turn_end
    from agent.prompt_builder import format_steer_marker
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    captured = {}
    manager._hooks["before_turn_end"] = [lambda **kwargs: captured.update(kwargs)]
    a = agent()
    scaffolds = [
        {"role": "user", "content": "todo", "_todo_snapshot_synthetic": True},
        {"role": "user", "content": "recover", "_empty_recovery_synthetic": True},
        {"role": "user", "content": "verify stop", "_verification_stop_synthetic": True},
        {"role": "user", "content": "pre-verify", "_pre_verify_synthetic": True},
        {"role": "user", "content": "repair", "_turn_end_synthetic": True},
        {"role": "user", "content": "retry tool", "_dropped_toolcall_nudge": True},
        {"role": "user", "content": "[System: Your previous tool call did not complete. Retry it.]"},
        {"role": "user", "content": format_steer_marker("change course")},
    ]
    for scaffold in scaffolds:
        messages = [
            {"role": "user", "content": "decision"},
            {"role": "tool", "content": "action and receipt"},
            scaffold,
            {"role": "assistant", "content": "continued"},
        ]
        captured.clear()
        assert before_turn_end(a, "done", {}, messages, user_message="decision", can_continue=True) is False
        assert captured["messages"] == messages[1:]


def test_interrupt_budget_and_hook_crash_never_force_continuation(monkeypatch):
    from agent.turn_end_hooks import before_turn_end
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    hook = Mock(return_value={"action": "block", "message": "mismatch"})
    manager._hooks["before_turn_end"] = [hook]
    a = agent()
    a._interrupt_requested = True
    messages = [{"role": "user", "content": "stop"}]
    assert before_turn_end(a, "draft", {}, messages, user_message="stop", can_continue=True) is False
    hook.assert_not_called()
    a._interrupt_requested = False
    assert before_turn_end(a, "draft", {}, messages, user_message="work", can_continue=False) is False
    assert len(messages) == 1
    hook.side_effect = RuntimeError("checker crashed")
    assert before_turn_end(a, "draft", {}, messages, user_message="work", can_continue=True) is False
    assert len(messages) == 1


def test_reworked_candidate_is_transformed_and_gated_as_delivered(monkeypatch):
    """Stock records one transform outcome per turn; a rework must not reuse the rejected one."""
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    seen = []
    def hook(final_response, **_):
        seen.append(final_response)
        return {"action": "block", "message": "Missing result file"}
    manager._hooks["before_turn_end"] = [hook]
    manager._hooks["transform_llm_output"] = [lambda response_text, **_: f"[{response_text}]"]
    a = agent()
    a._file_mutation_footer = None
    messages = [{"role": "user", "content": "do work"}]
    assert finish(a, "draft", messages).action == "continue"
    second = finish(a, "fixed", messages)
    assert second.action == "break"
    assert seen == ["[draft]", "[fixed]"]
    assert second.final_response == "[fixed]"
    assert messages[-1]["content"] == "[fixed]"
