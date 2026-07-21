from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.turn_finalizer import finalize_turn


class _StubAgent:
    def __init__(self, **overrides):
        self.max_iterations = 3
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=3)
        self.quiet_mode = True
        self.model = "stub-model"
        self.provider = "stub-provider"
        self.base_url = ""
        self.session_id = "sess-1"
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
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []

        self._delegate_depth = 0
        self._parent_session_id = None
        self._user_id = None
        self._chat_id = None
        self._chat_type = None
        self._thread_id = None
        self._gateway_session_key = None

        for key, value in overrides.items():
            setattr(self, key, value)

    def _handle_max_iterations(self, _messages, _api_call_count):
        return "iteration summary"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, _messages, _conversation_history):
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


@pytest.fixture
def captured_hooks(monkeypatch):
    captured = []

    def _invoke_hook(name, **kwargs):
        if name == "post_llm_call":
            captured.append(kwargs)
            return []
        if name == "transform_llm_output":
            return []
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)
    return captured


def _run_finalize(
    agent: _StubAgent,
    *,
    final_response: str | None = "final",
    interrupted: bool = False,
    failed: bool = False,
    api_call_count: int = 1,
    turn_exit_reason: str = "text_response(finish_reason=stop)",
):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "final"}]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )


def test_post_llm_call_includes_eligibility_fields(captured_hooks):
    agent = _StubAgent()

    _run_finalize(agent)

    assert len(captured_hooks) == 1
    payload = captured_hooks[0]
    assert payload["completed"] is True
    assert payload["failed"] is False
    assert payload["interrupted"] is False
    assert payload["turn_exit_reason"] == "text_response(finish_reason=stop)"


@pytest.mark.parametrize(
    "agent_overrides, expected",
    [
        (
            {"platform": "cli"},
            {
                "is_subagent": False,
                "delegate_depth": 0,
                "speaker_id": None,
                "conversation_id": None,
                "chat_id": None,
                "thread_id": None,
                "chat_type": None,
                "kanban_task_id": None,
            },
        ),
        (
            {
                "platform": "telegram",
                "_user_id": "u-1",
                "_chat_id": "chat-1",
                "_thread_id": "thread-1",
                "_chat_type": "group",
                "_gateway_session_key": "agent:main:telegram:group:chat-1:thread-1",
            },
            {
                "is_subagent": False,
                "delegate_depth": 0,
                "speaker_id": "u-1",
                "conversation_id": "agent:main:telegram:group:chat-1:thread-1",
                "chat_id": "chat-1",
                "thread_id": "thread-1",
                "chat_type": "group",
                "kanban_task_id": None,
            },
        ),
        (
            {
                "platform": "subagent",
                "_delegate_depth": 1,
                "_parent_session_id": "parent-1",
            },
            {
                "is_subagent": True,
                "delegate_depth": 1,
                "speaker_id": None,
                "conversation_id": None,
                "chat_id": None,
                "thread_id": None,
                "chat_type": None,
                "kanban_task_id": None,
            },
        ),
    ],
)
def test_post_llm_call_includes_origin_and_identity_fields(captured_hooks, agent_overrides, expected):
    agent = _StubAgent(**agent_overrides)

    _run_finalize(agent)

    payload = captured_hooks[0]
    for field, value in expected.items():
        assert payload[field] == value


def test_post_llm_call_includes_kanban_task_when_present(captured_hooks, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_card1")
    agent = _StubAgent(platform="cli")

    _run_finalize(agent)

    payload = captured_hooks[0]
    assert payload["kanban_task_id"] == "t_card1"


def test_post_llm_call_not_emitted_for_interrupted_turn(captured_hooks):
    agent = _StubAgent()

    _run_finalize(agent, interrupted=True)

    assert captured_hooks == []


def test_post_llm_call_marks_iteration_limit_as_not_completed(captured_hooks):
    agent = _StubAgent()

    _run_finalize(
        agent,
        final_response=None,
        api_call_count=agent.max_iterations,
        turn_exit_reason="budget_exhausted",
    )

    payload = captured_hooks[0]
    assert payload["completed"] is False
    assert payload["failed"] is False
    assert payload["interrupted"] is False
    assert payload["turn_exit_reason"].startswith("max_iterations_reached(")
