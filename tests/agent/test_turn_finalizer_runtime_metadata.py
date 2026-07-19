from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.turn_finalizer import finalize_turn


class _MetadataAgent:
    def __init__(self, reasoning_config):
        self.max_iterations = 10
        self.iteration_budget = SimpleNamespace(remaining=9, used=1, max_total=10)
        self.quiet_mode = True
        self.model = "openai/gpt-5.6-sol"
        self.provider = "openai"
        self.base_url = ""
        self.session_id = "metadata-session"
        self.reasoning_config = reasoning_config
        self.context_compressor = SimpleNamespace(last_prompt_tokens=42)
        self.session_input_tokens = 100
        self.session_output_tokens = 20
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 5
        self.session_prompt_tokens = 100
        self.session_completion_tokens = 20
        self.session_total_tokens = 120
        self.session_estimated_cost_usd = 0.1234
        self.session_cost_status = "estimated"
        self.session_cost_source = "official_docs_snapshot"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

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


def _finalize(monkeypatch, reasoning_config):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    return finalize_turn(
        _MetadataAgent(reasoning_config),
        final_response="done",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "assistant", "content": "done"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )


@pytest.mark.parametrize(
    ("reasoning_config", "expected"),
    [
        ({"enabled": True, "effort": "high"}, "high"),
        ({"enabled": False}, "none"),
        (None, "provider-default"),
    ],
)
def test_finalize_turn_exposes_effective_reasoning_effort(monkeypatch, reasoning_config, expected):
    result = _finalize(monkeypatch, reasoning_config)

    assert result["reasoning_effort"] == expected
    assert result["estimated_cost_usd"] == 0.1234
    assert result["cost_status"] == "estimated"
    assert result["cost_source"] == "official_docs_snapshot"
