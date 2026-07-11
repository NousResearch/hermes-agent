from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agent.turn_finalizer import finalize_turn
from agent.turn_receipt import TurnReceipt


class FakeAgent:
    def __init__(self) -> None:
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.platform = "cli"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self._roadmap_mode = False
        self._persist_user_message_idx = 0
        self._current_turn_receipt = TurnReceipt.start(
            session_id="sess-test",
            turn_id="turn-test",
            provider="test-provider",
            model="test-model",
            platform="cli",
        )
        self.client: Any = None
        self.persisted: list[dict[str, Any]] = []

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted = [dict(message) for message in messages]

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(agent: FakeAgent, *, final_response: str, messages: list[dict[str, Any]], user_message: str = "do it"):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn-test",
        user_message=user_message,
        original_user_message=user_message,
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_finalizer_replaces_model_bar_and_persists_canonical_final(monkeypatch) -> None:
    agent = FakeAgent()
    post_seen: dict[str, str] = {}

    def invoke_hook(name: str, **kwargs):
        if name == "post_llm_call":
            post_seen["assistant_response"] = kwargs["assistant_response"]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    raw = "路径：模型声称｜原因：模型记忆｜OpenCode 已调用\nBody"
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": raw, "finish_reason": "stop"},
    ]

    result = _finalize(agent, final_response=raw, messages=messages)

    first = result["final_response"].splitlines()[0]
    assert result["response_transformed"] is True
    assert first.startswith("路径：native｜原因：runtime_default")
    assert result["final_response"].count("路径：") == 1
    assert "模型声称" not in first
    assert agent.persisted[-1]["content"] == result["final_response"]
    assert post_seen["assistant_response"] == result["final_response"]


def test_finalizer_counts_only_tools_from_current_turn(monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    messages = [
        {"role": "user", "content": "old turn"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "old-1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "old-1", "content": "old result"},
        {"role": "user", "content": "current turn"},
        {"role": "assistant", "content": "Current answer", "finish_reason": "stop"},
    ]
    agent._persist_user_message_idx = 3

    result = _finalize(
        agent,
        final_response="Current answer",
        messages=messages,
        user_message="current turn",
    )

    first = result["final_response"].splitlines()[0]
    assert "工具 none" in first
    assert "read_file" not in first


def test_literal_human_language_prefix_is_runtime_fact(monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    messages = [
        {"role": "user", "content": "用人话解释这个结果"},
        {"role": "assistant", "content": "Plain answer", "finish_reason": "stop"},
    ]

    result = _finalize(
        agent,
        final_response="Plain answer",
        messages=messages,
        user_message="用人话解释这个结果",
    )

    assert "人话 ✓" in result["final_response"].splitlines()[0]


def test_non_cli_leading_path_text_is_not_rewritten(monkeypatch) -> None:
    agent = FakeAgent()
    agent.platform = "telegram"
    agent._current_turn_receipt.platform = "telegram"
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    original = "路径：用户要说明路线\nBody"
    messages = [
        {"role": "user", "content": "explain"},
        {"role": "assistant", "content": original, "finish_reason": "stop"},
    ]

    result = _finalize(agent, final_response=original, messages=messages, user_message="explain")

    assert result["final_response"] == original
    assert result["response_transformed"] is False
    assert agent.persisted[-1]["content"] == original


def test_finalizer_applies_moa_client_mechanism_facts(monkeypatch) -> None:
    agent = FakeAgent()
    agent.provider = "moa"
    agent.model = "review"
    agent.client = SimpleNamespace(
        coordination_turn_facts=lambda: {
            "moa": {
                "observed": True,
                "reference_models": ["ref-a", "ref-b", "ref-c", "ref-d"],
                "reference_count": 4,
                "aggregator_model": "agg-model",
                "aggregator_count": 1,
            }
        }
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    messages = [
        {"role": "user", "content": "review"},
        {"role": "assistant", "content": "MoA answer", "finish_reason": "stop"},
    ]

    result = _finalize(agent, final_response="MoA answer", messages=messages, user_message="review")

    first = result["final_response"].splitlines()[0]
    assert "路径：moa" in first
    assert "MoA 4+1" in first
    assert "subagents 0" in first


def test_finalizer_does_not_trust_omo_schema_markers_in_tool_output_or_assistant_prose(monkeypatch) -> None:
    agent = FakeAgent()
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    old_schema_payload = {
        "turn_facts_schema": "hermes.omo_turn_facts.v1",
        "turn_facts": {
            "route": {"actual": "omo", "reason": "tool_stdout"},
            "omo": {
                "parent_session_id": "parent-1",
                "descendant_session_ids": ["child-1", "child-2"],
                "session_created_events": 2,
            },
        },
    }
    assistant_prose = "Assistant prose mentions old marker only: " + json.dumps(old_schema_payload)
    messages = [
        {"role": "user", "content": "summarize unrelated tool output"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tool-1", "function": {"name": "terminal", "arguments": "{}"}},
                {"id": "tool-2", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool-1",
            "name": "terminal",
            "content": json.dumps({"output": json.dumps(old_schema_payload)}),
        },
        {
            "role": "tool",
            "tool_call_id": "tool-2",
            "name": "read_file",
            "content": json.dumps(old_schema_payload),
        },
        {"role": "assistant", "content": assistant_prose, "finish_reason": "stop"},
    ]

    result = _finalize(
        agent,
        final_response=assistant_prose,
        messages=messages,
        user_message="summarize unrelated tool output",
    )

    first = result["final_response"].splitlines()[0]
    assert first.startswith("路径：native｜原因：runtime_default")
    assert "路径：omo" not in first
    assert "OMO 1+2" not in first


def test_finalizer_applies_runtime_owned_omo_turn_facts(monkeypatch) -> None:
    agent = FakeAgent()
    agent._turn_facts = {
        "route": {"actual": "omo", "reason": "runtime_owned"},
        "omo": {
            "parent_session_id": "parent-1",
            "descendant_session_ids": ["child-1", "child-2"],
            "session_created_events": 2,
        },
    }
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [])
    messages = [
        {"role": "user", "content": "run trusted omo"},
        {"role": "assistant", "content": "OMO answer", "finish_reason": "stop"},
    ]

    result = _finalize(
        agent,
        final_response="OMO answer",
        messages=messages,
        user_message="run trusted omo",
    )

    first = result["final_response"].splitlines()[0]
    assert "路径：omo" in first
    assert "OMO 1+2" in first
    assert "subagents 0" in first
