from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_health_policy import (
    ContextHealthPolicy,
    PreModelIntakePolicy,
)
from agent.conversation_loop import run_conversation
from agent.turn_context import build_turn_context


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeTodoStore:
    def has_items(self):
        return True


class _FakeAgent:
    def __init__(self, tmp_path: Path):
        self.session_id = "sess-intake"
        self.model = "test/model"
        self.provider = "test-provider"
        self.base_url = ""
        self.api_key = ""
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 8
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = True
        self.compression_enabled = False
        self.context_compressor = type("Compressor", (), {"protect_first_n": 2, "protect_last_n": 2})()
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        self._context_health_intake_dir = tmp_path / "intake"
        self._persisted_messages = None
        self._persist_calls = 0
        self._ensure_calls = 0

    def _ensure_db_session(self):
        self._ensure_calls += 1

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, *_a, **_k):
        self._persist_calls += 1
        self._persisted_messages = [dict(m) for m in messages]


def _runtime_policy(*, threshold: int = 40, sensitive_action: str = "hold") -> ContextHealthPolicy:
    return ContextHealthPolicy(
        enabled=True,
        runtime_behavior_enabled=True,
        pre_model_intake=PreModelIntakePolicy(
            enabled=True,
            long_prompt_char_threshold=threshold,
            long_prompt_line_threshold=3,
            pre_history_required=True,
            high_risk_keywords=("token", "password", "secret"),
            sensitive_prompt_action=sensitive_action,
        ),
    )


def _build(agent, user_message: str, *, persist_user_message=None):
    return build_turn_context(
        agent=agent,
        user_message=user_message,
        system_message=None,
        conversation_history=[{"role": "assistant", "content": "previous ok"}],
        task_id="task-1",
        stream_callback=None,
        persist_user_message=persist_user_message,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: type("RA", (), {"_set_interrupt": lambda self, *a, **k: None})(),
    )


def test_disabled_policy_pass_through_without_file_writes(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = ContextHealthPolicy(enabled=False)
    raw = "RAW_NEEDLE_" + ("x" * 200)

    ctx = _build(agent, raw)

    assert ctx.messages[-1] == {"role": "user", "content": raw}
    assert agent._persist_user_message_override is None
    assert agent._persisted_messages[-1]["content"] == raw
    assert not agent._context_health_intake_dir.exists()


def test_long_prompt_replaced_before_append_and_persist(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)
    raw = "LONG_RAW_NEEDLE_" + ("alpha\n" * 20)

    ctx = _build(agent, raw)

    replacement = ctx.messages[-1]["content"]
    assert "LONG_RAW_NEEDLE_" not in replacement
    assert "Context Health Intake" in replacement
    assert str(agent._context_health_intake_dir) in replacement
    assert "LONG_RAW_NEEDLE_" not in repr(agent._persisted_messages)
    assert agent._persist_user_message_override == replacement


def test_raw_long_prompt_needle_absent_from_ctx_and_persisted_messages(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)
    raw = "ABSENT_NEEDLE_" + ("body line\n" * 20)

    ctx = _build(agent, raw)

    assert "ABSENT_NEEDLE_" not in repr(ctx.messages)
    assert "ABSENT_NEEDLE_" not in repr(agent._persisted_messages)


def test_intake_packet_contains_required_phase2_artifacts(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)

    ctx = _build(agent, "PACKET_NEEDLE_" + ("work item\n" * 20))

    assert ctx.context_health_intake is not None
    packet_dir = Path(ctx.context_health_intake.packet_dir)
    assert (packet_dir / "intake.md").exists()
    assert (packet_dir / "summary.md").exists()
    assert (packet_dir / "task-state.md").exists()
    assert "PACKET_NEEDLE_" not in (packet_dir / "summary.md").read_text()
    assert "PACKET_NEEDLE_" not in (packet_dir / "task-state.md").read_text()


def test_sensitive_long_prompt_holds_before_append(tmp_path):
    from agent.context_health_intake import PreTurnIntakeHold

    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40, sensitive_action="hold")
    raw = "SECRET_HOLD_NEEDLE token=abc123 " + ("private\n" * 20)

    with pytest.raises(PreTurnIntakeHold) as excinfo:
        _build(agent, raw)

    assert "SECRET_HOLD_NEEDLE" not in excinfo.value.user_response
    assert agent._persist_calls == 0
    assert agent._persisted_messages is None
    assert getattr(agent, "_persist_user_message_idx", None) is None


def test_run_conversation_returns_safe_hold_result_without_provider_or_raw_persist(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40, sensitive_action="hold")
    provider_called = False

    def fail_provider_call(*_args, **_kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider/model call should not happen on intake HOLD")

    agent._get_response = fail_provider_call
    raw = "RUNTIME_HOLD_NEEDLE token=abc123 password=private " + ("body\n" * 20)

    result = run_conversation(
        agent,
        raw,
        conversation_history=[{"role": "assistant", "content": "previous ok"}],
        task_id="task-1",
    )

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["api_calls"] == 0
    assert result["error"].startswith("context_health_hold:")
    assert "Context Health HOLD" in result["final_response"]
    assert "RUNTIME_HOLD_NEEDLE" not in repr(result)
    assert "abc123" not in repr(result)
    assert provider_called is False
    assert agent._persist_calls == 0
    assert agent._persisted_messages is None


def test_disabled_short_prompt_pass_through_without_file_writes(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = ContextHealthPolicy(enabled=False)

    ctx = _build(agent, "short prompt")

    assert ctx.messages[-1] == {"role": "user", "content": "short prompt"}
    assert agent._persisted_messages[-1]["content"] == "short prompt"
    assert not agent._context_health_intake_dir.exists()


def test_persist_user_message_override_updates_consistently_after_replacement(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)
    raw = "OVERRIDE_NEEDLE_" + ("line\n" * 20)

    ctx = _build(agent, raw, persist_user_message="clean original")

    assert agent._persist_user_message_override == ctx.user_message
    assert ctx.original_user_message == ctx.user_message
    assert "OVERRIDE_NEEDLE_" not in repr(agent._persisted_messages)


def test_role_alternation_preserved_after_replacement(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)

    ctx = _build(agent, "ROLE_NEEDLE_" + ("line\n" * 20))

    assert [m["role"] for m in ctx.messages] == ["assistant", "user"]


def test_pre_llm_call_receives_safe_replacement_not_raw_long_prompt(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)
    raw = "PLUGIN_NEEDLE_" + ("line\n" * 20)
    captured = {}

    def fake_invoke_hook(_name, **kwargs):
        captured.update(kwargs)
        return []

    with patch("hermes_cli.plugins.invoke_hook", fake_invoke_hook):
        _build(agent, raw)

    assert "PLUGIN_NEEDLE_" not in repr(captured["conversation_history"])
    assert "PLUGIN_NEEDLE_" not in captured["user_message"]
    assert "Context Health Intake" in captured["user_message"]


def test_no_provider_or_model_call_during_intake_classification(tmp_path):
    agent = _FakeAgent(tmp_path)
    agent._context_health_policy = _runtime_policy(threshold=40)
    agent._get_response = lambda *a, **k: (_ for _ in ()).throw(AssertionError("provider called"))

    ctx = _build(agent, "NO_PROVIDER_NEEDLE_" + ("line\n" * 20))

    assert ctx.context_health_intake is not None
