"""Host integration tests for ContextEngine lifecycle hooks (PR-2).

Covers the three host integration points:
- ``run_agent.AIAgent._compress_context`` — ``on_pre_compress`` snapshot
  (capability-gated, fail-open) + ``_turn_compressed`` marker semantics
- ``agent.turn_finalizer.finalize_turn`` — ``on_turn_complete`` observation
  (capability-gated, fires before external-memory sync, fail-open)
- capability gating: undeclared engines see zero new calls

Uses the bare-agent pattern established in
``tests/run_agent/test_memory_sync_interrupted.py``.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextEngineCapabilities, TurnInfo


# ── helpers ──────────────────────────────────────────────────────────────────


def _bare_compress_agent(caps: ContextEngineCapabilities):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._context_engine_caps = caps
    agent.context_compressor = MagicMock()
    agent.context_compressor.name = "mock-engine"
    agent._turn_compressed = False
    return agent


def _finalize_agent(caps: ContextEngineCapabilities):
    """MagicMock agent with the real-typed attrs finalize_turn computes on."""
    agent = MagicMock()
    agent.max_iterations = 10
    agent.iteration_budget.remaining = 5
    agent.iteration_budget.used = 1
    agent.iteration_budget.max_total = 10
    agent.session_id = "sess-1"
    agent.quiet_mode = True
    agent.model = "test-model"
    agent.provider = "test"
    agent.base_url = ""
    agent.platform = "cli"
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "ok"
    agent.session_cost_source = "none"
    agent._tool_guardrail_halt_decision = None
    agent._drain_pending_steer.return_value = None
    agent._response_was_previewed = False
    agent._interrupt_message = None
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = set()
    agent._turn_failed_file_mutations = {}
    agent._file_mutation_verifier_enabled.return_value = False
    agent._turn_completion_explainer_enabled.return_value = False
    agent._context_engine_caps = caps
    agent._turn_compressed = False
    agent.context_compressor = MagicMock()
    agent.context_compressor.name = "mock-engine"
    agent.context_compressor.last_prompt_tokens = 1234
    agent.context_compressor.last_completion_tokens = 56
    agent.context_compressor.last_total_tokens = 1290
    return agent


def _run_finalize(agent, *, interrupted=False, final_response="All done."):
    from agent.turn_finalizer import finalize_turn

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": final_response or "x"},
    ]
    result = finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=2,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="default",
        turn_id="turn-42",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="text_response",
    )
    return result, messages


# ── _compress_context forwarder: on_pre_compress + marker ───────────────────


class TestCompressForwarder:
    def _patch_compress(self, new_list):
        return patch(
            "agent.conversation_compression.compress_context",
            return_value=(new_list, "sys"),
        )

    def test_snapshot_called_before_compress(self):
        agent = _bare_compress_agent(
            ContextEngineCapabilities(lossless_snapshot=True)
        )
        msgs = [{"role": "user", "content": "hi"}]
        order = []
        agent.context_compressor.on_pre_compress.side_effect = (
            lambda m: order.append("snapshot")
        )
        with patch(
            "agent.conversation_compression.compress_context",
            side_effect=lambda *a, **k: (order.append("compress"), ([], "sys"))[1],
        ):
            agent._compress_context(msgs, "sys")
        assert order == ["snapshot", "compress"]
        agent.context_compressor.on_pre_compress.assert_called_once_with(msgs)

    def test_snapshot_gated_off_by_default(self):
        agent = _bare_compress_agent(ContextEngineCapabilities())
        msgs = [{"role": "user", "content": "hi"}]
        with self._patch_compress([]):
            agent._compress_context(msgs, "sys")
        agent.context_compressor.on_pre_compress.assert_not_called()

    def test_snapshot_failure_does_not_block_compression(self):
        agent = _bare_compress_agent(
            ContextEngineCapabilities(lossless_snapshot=True)
        )
        agent.context_compressor.on_pre_compress.side_effect = RuntimeError("boom")
        msgs = [{"role": "user", "content": "hi"}]
        with self._patch_compress([]):
            out, _ = agent._compress_context(msgs, "sys")
        assert out == []  # compression still ran

    def test_marker_set_only_when_list_replaced(self):
        agent = _bare_compress_agent(ContextEngineCapabilities())
        msgs = [{"role": "user", "content": "hi"}]
        # Abort path: compress_context returns the SAME object → no marker.
        with patch(
            "agent.conversation_compression.compress_context",
            return_value=(msgs, "sys"),
        ):
            agent._compress_context(msgs, "sys")
        assert agent._turn_compressed is False
        # Real compaction: new list → marker set.
        with self._patch_compress([{"role": "user", "content": "summary"}]):
            agent._compress_context(msgs, "sys")
        assert agent._turn_compressed is True


# ── finalize_turn: on_turn_complete observation ──────────────────────────────


class TestTurnObservation:
    def test_fires_with_observation_capability(self):
        agent = _finalize_agent(ContextEngineCapabilities(observation=True))
        _, messages = _run_finalize(agent)
        hook = agent.context_compressor.on_turn_complete
        hook.assert_called_once()
        called_messages, turn = hook.call_args.args
        assert called_messages is messages
        assert isinstance(turn, TurnInfo)
        assert turn.session_id == "sess-1"
        assert turn.turn_id == "turn-42"
        assert turn.usage["prompt_tokens"] == 1234
        assert turn.interrupted is False
        assert turn.completed is True

    def test_gated_off_by_default(self):
        agent = _finalize_agent(ContextEngineCapabilities())
        _run_finalize(agent)
        agent.context_compressor.on_turn_complete.assert_not_called()

    def test_fires_on_interrupted_turn_with_flag(self):
        agent = _finalize_agent(ContextEngineCapabilities(observation=True))
        _run_finalize(agent, interrupted=True)
        hook = agent.context_compressor.on_turn_complete
        hook.assert_called_once()
        _, turn = hook.call_args.args
        assert turn.interrupted is True

    def test_runs_before_external_memory_sync(self):
        agent = _finalize_agent(ContextEngineCapabilities(observation=True))
        order = []
        agent.context_compressor.on_turn_complete.side_effect = (
            lambda *a, **k: order.append("engine")
        )
        agent._sync_external_memory_for_turn.side_effect = (
            lambda **k: order.append("memory")
        )
        _run_finalize(agent)
        assert order == ["engine", "memory"]

    def test_hook_failure_does_not_break_turn(self):
        agent = _finalize_agent(ContextEngineCapabilities(observation=True))
        agent.context_compressor.on_turn_complete.side_effect = RuntimeError("boom")
        result, _ = _run_finalize(agent)
        assert result["final_response"] == "All done."
        agent._sync_external_memory_for_turn.assert_called_once()

    def test_compressed_during_turn_flag_propagates(self):
        agent = _finalize_agent(ContextEngineCapabilities(observation=True))
        agent._turn_compressed = True
        _run_finalize(agent)
        _, turn = agent.context_compressor.on_turn_complete.call_args.args
        assert turn.compressed_during_turn is True

    def test_no_caps_attribute_is_safe(self):
        """Agents built before agent_init ran (bare paths) must not crash."""
        agent = _finalize_agent(ContextEngineCapabilities())
        del agent._context_engine_caps
        # MagicMock would auto-create the attr; emulate absence with None.
        agent._context_engine_caps = None
        result, _ = _run_finalize(agent)
        assert result["final_response"] == "All done."


# ── background-review fork isolation (B6) ────────────────────────────────────


class TestReviewForkIsolation:
    def test_review_fork_disables_context_engine_lifecycle(self, monkeypatch):
        """The internal review fork shares the parent session_id, so an
        observation-capable engine would ingest the review's internal
        monologue into the parent conversation's store. The fork must zero
        out the capability snapshot before running."""
        from agent.background_review import _run_review_in_thread

        created = []

        def _factory(*args, **kwargs):
            inst = MagicMock()
            inst.run_conversation.side_effect = RuntimeError("stop after setup")
            created.append(inst)
            return inst

        monkeypatch.setattr("run_agent.AIAgent", _factory)
        parent = MagicMock()
        parent._current_main_runtime.return_value = {
            "api_mode": "chat_completions", "base_url": "", "api_key": "k",
        }
        parent.session_id = "parent-sess"
        parent.model = "m"
        parent.platform = "cli"
        parent.provider = "p"

        _run_review_in_thread(parent, [], "review prompt")

        assert created, "review agent was never constructed"
        caps = created[0]._context_engine_caps
        assert isinstance(caps, ContextEngineCapabilities)
        assert caps.observation is False
        assert caps.request_assembly is False
        assert caps.lossless_snapshot is False
