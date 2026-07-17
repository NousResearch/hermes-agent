"""Focused tests for per-turn memory disposition behavior."""

from run_agent import AIAgent
from agent.memory_manager import MemoryManager
from agent.turn_provenance import TURN_MEMORY_DISPOSITION_KEY
from hermes_state import SessionDB

from agent.turn_finalizer import finalize_turn
from agent.turn_provenance import (
    ASYNC_DELEGATION_COMPLETION_TURN,
    NORMAL_USER_TURN,
)


class _Budget:
    used = 0
    max_total = 10
    remaining = 10


class _Agent:
    def __init__(self):
        self.max_iterations = 10
        self.iteration_budget = _Budget()
        self.context_compressor = None
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
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.synced = []
        self.review_calls = []
        self.persisted_messages = None
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"
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

    def _save_trajectory(self, *args, **kwargs):
        pass

    def _cleanup_task_resources(self, *args, **kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *args, **kwargs):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _emit_status(self, *args, **kwargs):
        pass

    def _safe_print(self, *args, **kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **kwargs):
        self.synced.append(kwargs)

    def _spawn_background_review(self, **kwargs):
        self.review_calls.append(kwargs)


class _SyncOnlyAgent:
    def __init__(self):
        self._memory_manager = _MemoryManagerStub()
        self.session_id = "sess-1"


class _MemoryManagerStub:
    def __init__(self):
        self.sync_calls = []
        self.prefetch_calls = []

    def sync_all(self, *args, **kwargs):
        self.sync_calls.append((args, kwargs))

    def queue_prefetch_all(self, *args, **kwargs):
        self.prefetch_calls.append((args, kwargs))


def _run_finalizer(turn_provenance, should_review_memory):
    agent = _Agent()
    messages = [{"role": "user", "content": "synthetic"}]
    result = finalize_turn(
        agent,
        final_response="reply",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="synthetic",
        original_user_message="synthetic",
        _should_review_memory=should_review_memory,
        turn_provenance=turn_provenance,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )
    return agent, result


def test_normal_turn_syncs_memory_and_runs_review_when_requested():
    agent, result = _run_finalizer(NORMAL_USER_TURN, should_review_memory=True)

    assert result["final_response"] == "reply"
    assert agent.synced and agent.synced[0]["turn_provenance"] == NORMAL_USER_TURN
    assert agent.review_calls == [
        {
            "messages_snapshot": [
                {"role": "user", "content": "synthetic"},
                {"role": "assistant", "content": "reply"},
            ],
            "review_memory": True,
            "review_skills": False,
        }
    ]


def test_sync_external_memory_for_turn_skips_non_retain_turns():
    agent = _SyncOnlyAgent()

    AIAgent._sync_external_memory_for_turn(
        agent,
        original_user_message="synthetic",
        final_response="reply",
        interrupted=False,
        messages=[{"role": "user", "content": "synthetic"}],
        turn_provenance=ASYNC_DELEGATION_COMPLETION_TURN,
    )

    assert agent._memory_manager.sync_calls == []
    assert agent._memory_manager.prefetch_calls == []


def test_sync_external_memory_for_turn_keeps_default_retain_behavior():
    agent = _SyncOnlyAgent()

    AIAgent._sync_external_memory_for_turn(
        agent,
        original_user_message="hello",
        final_response="reply",
        interrupted=False,
        messages=[{"role": "user", "content": "hello"}],
        turn_provenance=NORMAL_USER_TURN,
    )

    assert len(agent._memory_manager.sync_calls) == 1
    assert len(agent._memory_manager.prefetch_calls) == 1


def test_non_retain_turn_still_persists_response_but_skips_memory_review():
    agent, result = _run_finalizer(
        ASYNC_DELEGATION_COMPLETION_TURN,
        should_review_memory=True,
    )

    assert result["final_response"] == "reply"
    assert agent.synced and agent.synced[0]["turn_provenance"] == ASYNC_DELEGATION_COMPLETION_TURN
    assert agent.review_calls == []
    assert agent.persisted_messages[-1] == {"role": "assistant", "content": "reply"}


def test_persist_reload_then_session_end_filters_entire_synthetic_turn(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-1", source="cli")
    db.replace_messages(
        "sess-1",
        [
            {"role": "user", "content": "ordinary question"},
            {
                "role": "user",
                "content": "delegated completion",
                TURN_MEMORY_DISPOSITION_KEY: "do_not_retain",
            },
            {"role": "assistant", "content": "synthetic answer"},
            {"role": "user", "content": "ordinary follow-up"},
            {"role": "assistant", "content": "ordinary reply"},
        ],
    )
    db.close()

    reloaded = SessionDB(db_path=tmp_path / "state.db")
    provider = type("Provider", (), {})()
    provider.name = "test"
    provider.get_tool_schemas = lambda: []
    provider.on_session_end = lambda messages: setattr(provider, "received", messages)
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.on_session_end(reloaded.get_messages_as_conversation("sess-1"))

    contents = [message["content"] for message in provider.received]
    assert contents == ["ordinary question", "ordinary follow-up", "ordinary reply"]
    assert all(TURN_MEMORY_DISPOSITION_KEY not in message for message in provider.received)
    reloaded.close()
