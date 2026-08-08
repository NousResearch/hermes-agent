"""Regression test for #57626.

Sub-agent sessions spawned via ``delegate_task`` must never receive the skill
library update injection.  The skill-review nudge is a foreground-only concern;
letting it fire in sub-agents hijacks their final turns with irrelevant
``skill_manage`` calls and pollutes the skill library with unreviewed content.

The fix adds a ``_delegate_depth == 0`` guard to both the iteration counter
(``conversation_loop.py``) and the trigger check (``turn_finalizer.py``).
"""

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 100
    remaining = 50


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface for ``finalize_turn``."""

    def __init__(self, *, delegate_depth=0):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._delegate_depth = delegate_depth
        self._skill_nudge_interval = 10
        self._iters_since_skill = 10  # exactly at threshold
        self.valid_tool_names = {"read_file", "skill_manage"}
        self._background_review_spawned = False
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _spawn_background_review(self, **kwargs):
        self._background_review_spawned = True

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "stub"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


_MESSAGES = [
    {"role": "user", "content": "do a thing"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
]


def _run_finalize(agent):
    return finalize_turn(
        agent,
        final_response="done",
        api_call_count=5,
        interrupted=False,
        failed=False,
        messages=_MESSAGES,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="stop",
    )


class TestSubagentSkillNudgeExclusion:
    """Sub-agents must not trigger the background skill review."""

    def test_foreground_agent_triggers_skill_review(self):
        """Foreground agent (delegate_depth=0) at threshold triggers review."""
        agent = _StubAgent(delegate_depth=0)
        _run_finalize(agent)
        assert agent._background_review_spawned is True

    def test_subagent_does_not_trigger_skill_review(self):
        """Sub-agent (delegate_depth=1) at threshold does NOT trigger review."""
        agent = _StubAgent(delegate_depth=1)
        _run_finalize(agent)
        assert agent._background_review_spawned is False

    def test_deeply_nested_subagent_does_not_trigger(self):
        """Nested sub-agent (delegate_depth=3) also excluded."""
        agent = _StubAgent(delegate_depth=3)
        _run_finalize(agent)
        assert agent._background_review_spawned is False
