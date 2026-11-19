"""Issue #6 (hermes-security-suite): inline-transform flag propagation.

``transform_llm_output`` plugins whose rewrite is an in-place substitution of
the streamed text (Privacy-Shield's TOKEN_* restore replaces tokens in the
middle of the reply) must be able to tell the CLI reprint fast-path that the
rewrite is NOT a prefix extension, so the already-streamed copy is not
repainted in full. The plugin signals it by returning a ``str`` subclass with
``transform_is_inline=True``; the core records the flag on the agent and puts
it into the turn result for the render layer.
"""

from types import SimpleNamespace

import pytest

# Import at collection time (home_io_guard arms per-test): the first
# ``agent.conversation_loop`` import runs its bootstrap dependency probe, which
# other test modules already performed the same way at module top level.
import agent.conversation_loop

from agent.turn_finalizer import apply_llm_output_transform, finalize_turn


class _Flagged(str):
    transform_is_inline = False


def _flagged(text: str) -> str:
    out = _Flagged(text)
    out.transform_is_inline = True
    return out


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """finalize_turn consults the skills trigger + plugin discovery: both must
    never read the real hermes home in tests (home_io_guard)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _agent():
    def _noop(*_args, **_kwargs):
        return None

    return SimpleNamespace(
        _llm_output_transform=None,
        _llm_output_transform_inline=False,
        session_id="sess",
        model="test-model",
        platform="cli",
        _hold_stream=False,
        _held_transforms=None,
        # finalize_turn seam stubs (no-ops; the persistence harness pattern)
        _handle_max_iterations=_noop,
        _emit_status=_noop,
        _safe_print=_noop,
        _save_trajectory=_noop,
        _cleanup_task_resources=_noop,
        _drop_trailing_empty_response_scaffolding=_noop,
        _persist_session=_noop,
        _apply_persist_user_message_override=_noop,
        _file_mutation_verifier_enabled=lambda: False,
        _turn_completion_explainer_enabled=lambda: False,
        _drain_pending_steer=_noop,
        clear_interrupt=_noop,
        _sync_external_memory_for_turn=_noop,
    )


def _patch_hooks(monkeypatch, results):
    lambda_hook = lambda name, **_kw: (
        results if name == "transform_llm_output" else []
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda_hook)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda_hook)


def test_apply_records_inline_flag_from_str_subclass(monkeypatch):
    agent = _agent()
    _patch_hooks(monkeypatch, [_flagged("restored reply")])

    final, transformed, pre = apply_llm_output_transform(agent, "TOKEN_X reply", turn_id="t1")

    assert (final, transformed, pre) == ("restored reply", True, "TOKEN_X reply")
    assert agent._llm_output_transform_inline is True


def test_apply_resets_inline_flag_when_no_rewrite(monkeypatch):
    agent = _agent()
    agent._llm_output_transform_inline = True  # stale from an earlier turn
    _patch_hooks(monkeypatch, [])

    final, transformed, _pre = apply_llm_output_transform(agent, "plain reply", turn_id="t2")

    assert (final, transformed) == ("plain reply", False)
    assert agent._llm_output_transform_inline is False


def test_memoized_call_keeps_recorded_inline_flag(monkeypatch):
    agent = _agent()
    _patch_hooks(monkeypatch, [_flagged("restored reply")])
    apply_llm_output_transform(agent, "TOKEN_X reply", turn_id="t1")

    _patch_hooks(monkeypatch, [])  # a replay later in the turn must not re-fire
    _final, transformed, _pre = apply_llm_output_transform(agent, "restored reply", turn_id="t1")

    assert transformed is True
    assert agent._llm_output_transform_inline is True


def test_finalize_result_carries_inline_flag(monkeypatch, isolated_hermes_home):
    agent = _agent()
    agent.max_iterations = 90
    agent.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
    agent.quiet_mode = True
    agent.provider = "test-provider"
    agent.base_url = ""
    agent.context_compressor = SimpleNamespace(last_prompt_tokens=0)
    for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                  "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
        setattr(agent, f"session_{field}", 0)
    agent.session_estimated_cost_usd = 0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "test"
    agent._tool_guardrail_halt_decision = None
    agent._interrupt_message = None
    agent._response_was_previewed = False
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = []

    _patch_hooks(monkeypatch, [_flagged("restored reply")])
    messages = [{"role": "user", "content": "do it", "timestamp": 1.0}]

    result = finalize_turn(
        agent,
        final_response="TOKEN_X reply",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="t-inline",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert result["response_transformed"] is True
    assert result["transform_is_inline"] is True
    assert result["final_response"] == "restored reply"


def test_finalize_result_defaults_inline_flag_to_false(monkeypatch, isolated_hermes_home):
    agent = _agent()
    agent.max_iterations = 90
    agent.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
    agent.quiet_mode = True
    agent.provider = "test-provider"
    agent.base_url = ""
    agent.context_compressor = SimpleNamespace(last_prompt_tokens=0)
    for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                  "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
        setattr(agent, f"session_{field}", 0)
    agent.session_estimated_cost_usd = 0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "test"
    agent._tool_guardrail_halt_decision = None
    agent._interrupt_message = None
    agent._response_was_previewed = False
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = []

    _patch_hooks(monkeypatch, [])
    messages = [{"role": "user", "content": "do it", "timestamp": 1.0}]

    result = finalize_turn(
        agent,
        final_response="plain reply",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="t-plain",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert result["response_transformed"] is False
    assert result["transform_is_inline"] is False
