"""Actual >500 prologue compaction and sticky post-prologue diagnostics."""
from copy import deepcopy
from unittest.mock import patch
import pytest
from tests.run_agent.test_413_compression import agent  # noqa: F401
from agent.model_metadata import estimate_request_tokens_rough
import agent.turn_context as turn_context


@pytest.mark.parametrize("mode", ["replacement", "inplace", "system", "partial", "nochange", "equalcopy", "unknown"])
def test_real_prologue_compaction_mutation_survives_loop_reset(agent, mode):
    a = agent
    a.tools = []
    a.max_tokens = 1
    a.max_compression_attempts = 1
    a.context_compressor.context_length = 1000
    a.context_compressor.threshold_tokens = 500
    before = [{"role": "user", "content": "x" * 6000}, {"role": "assistant", "content": "previous"}]
    assert estimate_request_tokens_rough(before, system_prompt=a._cached_system_prompt) > 500
    contexts, compress_tokens = [], []
    def compact(rows, *args, **kwargs):
        compress_tokens.append(kwargs["approx_tokens"])
        system = a._cached_system_prompt
        a.context_compressor.last_prompt_tokens = -1
        a.context_compressor.awaiting_real_usage_after_compression = True
        if mode == "equalcopy":
            return deepcopy(rows), system
        if mode == "unknown":
            # Non-serializable already-present auxiliary state prevents proof.
            rows[0]["opaque_test_state"] = object()
        if mode == "replacement":
            return [{"role": "user", "content": "p" * 4400}], system
        if mode == "inplace":
            rows[0]["content"] = "p" * 4400
        if mode == "system":
            system += " changed"
        if mode == "partial":
            rows.pop(1)
        return rows, system
    original = turn_context.build_turn_context
    def build(*args, **kwargs):
        ctx = original(*args, **kwargs)
        contexts.append(ctx)
        return ctx
    with (
        patch.object(a, "_compress_context", side_effect=compact),
        patch.object(a.context_compressor, "get_active_compression_failure_cooldown", return_value=None),
        patch("agent.conversation_loop.build_turn_context", side_effect=build),
        patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"),
        patch.object(a, "_cleanup_task_resources"), patch.object(a, "_emit_status") as status,
    ):
        result = a.run_conversation("hello", conversation_history=deepcopy(before))
    assert len(contexts) == 1
    assert compress_tokens and compress_tokens[0] > 500
    diagnostic = str(result.get("final_response")) + " ".join(str(c.args[0]) for c in status.call_args_list if c.args)
    assert result.get("failed") is True
    assert a.client.chat.completions.create.call_count == 0
    if mode in {"nochange", "equalcopy"}:
        assert contexts[0].preflight_transcript_rewritten is False
        assert "No messages were dropped" in diagnostic
    else:
        assert contexts[0].preflight_transcript_rewritten is True
        assert "No messages were dropped" not in diagnostic
        if mode == "unknown":
            assert contexts[0].preflight_mutation_outcome == "unknown"
            assert "mutation outcome is unknown" in diagnostic
