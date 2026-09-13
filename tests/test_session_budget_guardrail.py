from pathlib import Path
from types import SimpleNamespace

from agent import session_budget_guardrail as sbg
from agent.session_budget_guardrail import DEFAULT_SESSION_BUDGET_GUARDRAIL
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _agent(config=None):
    agent = SimpleNamespace()
    sbg.initialize_agent(agent, config or {"agent": {"session_budget_guardrail": {"enabled": True}}})
    return agent


def test_config_defaults_include_session_budget_guardrail():
    guardrail = DEFAULT_CONFIG["agent"]["session_budget_guardrail"]
    assert guardrail == DEFAULT_SESSION_BUDGET_GUARDRAIL
    assert guardrail["enabled"] is False
    assert guardrail["soft_prompt_tokens"] == 180_000
    assert guardrail["hard_prompt_tokens"] == 240_000
    assert guardrail["hard_prompt_tokens"] > guardrail["soft_prompt_tokens"]
    assert guardrail["hard_consecutive_soft_hits"] == 3
    assert guardrail["hard_projected_cost_usd"] == 25.0
    assert guardrail["pause_and_ask"] is True


def test_disabled_config_is_noop():
    agent = _agent({"agent": {"session_budget_guardrail": {"enabled": False}}})
    state = sbg.record_usage(agent, prompt_tokens=999_999, projected_cost_usd=999)
    assert state.pending_hard_breach is False
    assert sbg.has_pending_breach(agent) is False


def test_soft_warning_records_without_hard_pause():
    agent = _agent({"agent": {"session_budget_guardrail": {
        "enabled": True,
        "soft_prompt_tokens": 100,
        "hard_prompt_tokens": 500,
        "hard_consecutive_soft_hits": 3,
        "hard_projected_cost_usd": 99,
    }}})
    state = sbg.record_usage(agent, prompt_tokens=120, projected_cost_usd=1)
    assert state.consecutive_soft_hits == 1
    assert state.pending_hard_breach is False


def test_hard_prompt_token_threshold_latches():
    agent = _agent({"agent": {"session_budget_guardrail": {
        "enabled": True,
        "soft_prompt_tokens": 100,
        "hard_prompt_tokens": 200,
        "hard_consecutive_soft_hits": 3,
        "hard_projected_cost_usd": 99,
    }}})
    state = sbg.record_usage(agent, prompt_tokens=250, projected_cost_usd=1)
    assert state.pending_hard_breach is True
    assert "hard_prompt_tokens" in state.reasons[0]
    assert sbg.has_pending_breach(agent) is True


def test_consecutive_soft_hits_latch():
    agent = _agent({"agent": {"session_budget_guardrail": {
        "enabled": True,
        "soft_prompt_tokens": 100,
        "hard_prompt_tokens": 999,
        "hard_consecutive_soft_hits": 3,
        "hard_projected_cost_usd": 99,
    }}})
    for _ in range(3):
        state = sbg.record_usage(agent, prompt_tokens=120, projected_cost_usd=1)
    assert state.consecutive_soft_hits == 3
    assert state.pending_hard_breach is True
    assert any("consecutive soft" in reason for reason in state.reasons)


def test_projected_cost_threshold_latches():
    agent = _agent({"agent": {"session_budget_guardrail": {
        "enabled": True,
        "soft_prompt_tokens": 1000,
        "hard_prompt_tokens": 2000,
        "hard_consecutive_soft_hits": 3,
        "hard_projected_cost_usd": 2.5,
    }}})
    state = sbg.record_usage(agent, prompt_tokens=10, projected_cost_usd=3.0)
    assert state.pending_hard_breach is True
    assert any("projected session cost" in reason for reason in state.reasons)


def test_continue_once_clears_pending_latch():
    agent = _agent({"agent": {"session_budget_guardrail": {
        "enabled": True,
        "hard_prompt_tokens": 10,
    }}})
    sbg.record_usage(agent, prompt_tokens=11, projected_cost_usd=0)
    sbg.clear_pending_for_continue_once(agent)
    assert agent.session_budget_guardrail_state.pending_hard_breach is False
    assert agent.session_budget_guardrail_state.continue_once_armed is True


def test_compress_then_continue_choice_and_clear_path():
    agent = _agent({"agent": {"session_budget_guardrail": {"enabled": True}}})
    agent.clarify_callback = lambda question, choices, multi_select=False: "Compress then continue"
    agent.session_budget_guardrail_state.pending_hard_breach = True
    agent.session_budget_guardrail_state.consecutive_soft_hits = 3
    assert sbg.ask_for_decision(agent) == sbg.CHOICE_COMPRESS_THEN_CONTINUE
    sbg.clear_pending_after_compress(agent)
    assert agent.session_budget_guardrail_state.pending_hard_breach is False
    assert agent.session_budget_guardrail_state.consecutive_soft_hits == 0


def test_stop_is_fail_safe_when_no_clarify_callback():
    agent = _agent({"agent": {"session_budget_guardrail": {"enabled": True}}})
    agent.clarify_callback = None
    agent.session_budget_guardrail_state.pending_hard_breach = True
    assert sbg.ask_for_decision(agent) == sbg.CHOICE_STOP


def test_reset_session_state_clears_guardrail_latch(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.context_compressor = None
    agent.session_id = "new-session"
    agent._transition_context_engine_session = lambda **kwargs: None
    sbg.initialize_agent(agent, {"agent": {"session_budget_guardrail": {"enabled": True}}})
    sbg.record_usage(agent, prompt_tokens=999_999, projected_cost_usd=999)
    assert agent.session_budget_guardrail_state.pending_hard_breach is True

    AIAgent.reset_session_state(agent)

    assert agent.session_budget_guardrail_state.pending_hard_breach is False
    assert agent.session_budget_guardrail_state.consecutive_soft_hits == 0



def test_clarify_callback_exception_fails_safe_to_stop():
    agent = _agent({"agent": {"session_budget_guardrail": {"enabled": True}}})
    def boom(*args, **kwargs):
        raise RuntimeError("callback failed")
    agent.clarify_callback = boom
    agent.session_budget_guardrail_state.pending_hard_breach = True
    assert sbg.ask_for_decision(agent) == sbg.CHOICE_STOP


def test_reset_session_state_clears_every_guardrail_field(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.context_compressor = None
    agent.session_id = "new-session"
    agent._transition_context_engine_session = lambda **kwargs: None
    sbg.initialize_agent(agent, {"agent": {"session_budget_guardrail": {"enabled": True}}})
    state = sbg.record_usage(agent, prompt_tokens=999_999, projected_cost_usd=999)
    state.continue_once_armed = True
    assert state.pending_hard_breach is True
    assert state.fired is True
    assert state.reasons

    AIAgent.reset_session_state(agent)
    state = agent.session_budget_guardrail_state
    assert state.pending_hard_breach is False
    assert state.consecutive_soft_hits == 0
    assert state.last_prompt_tokens == 0
    assert state.last_projected_cost_usd == 0.0
    assert state.reasons == []
    assert state.fired is False
    assert state.continue_once_armed is False


def test_guardrail_gate_runs_before_iteration_budget_is_consumed():
    source = Path("agent/conversation_loop.py").read_text(encoding="utf-8")
    gated = source.index("_sbg = _run_phase(run_session_budget_guardrail_gate")
    begun = source.index("if _run_phase(begin_iteration", gated)
    assert gated < begun


def test_compression_latch_only_clears_after_successful_compression(monkeypatch):
    agent = _agent({"agent": {"session_budget_guardrail": {"enabled": True}}})
    agent.session_budget_guardrail_state.pending_hard_breach = True
    agent.clarify_callback = lambda question, choices, multi_select=False: "Compress then continue"
    agent.context_compressor = SimpleNamespace(compress_context=lambda **kwargs: False)

    result = sbg.enforce_before_provider_call(agent, messages=[], conversation_history=[])

    assert result and "compression was unavailable or deferred" in result
    assert agent.session_budget_guardrail_state.pending_hard_breach is True

    agent.context_compressor = SimpleNamespace(compress_context=lambda **kwargs: True)
    result = sbg.enforce_before_provider_call(agent, messages=[], conversation_history=[])

    assert result is None
    assert agent.session_budget_guardrail_state.pending_hard_breach is False

def test_guardrail_stop_returns_run_conversation_envelope():
    from agent.turn_session_budget_guardrail import run_session_budget_guardrail_gate

    agent = _agent({"agent": {"session_budget_guardrail": {"enabled": True}}})
    agent.session_budget_guardrail_state.pending_hard_breach = True
    agent.clarify_callback = None
    agent.api_call_count = 2
    agent.model = "test-model"
    agent.provider = "test-provider"
    agent.base_url = None
    agent.session_id = "session-1"
    for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
        "estimated_cost_usd", "cost_status", "cost_source",
    ):
        setattr(agent, f"session_{key}", 0)

    verdict = run_session_budget_guardrail_gate(
        agent,
        messages=[{"role": "user", "content": "next"}],
        conversation_history=[],
    )

    assert verdict.action == "return"
    assert isinstance(verdict.result, dict)
    assert verdict.result["completed"] is False
    assert verdict.result["failed"] is True
    assert verdict.result["api_calls"] == 2
    assert verdict.result["messages"][-1]["role"] == "assistant"
    assert verdict.result["failure_reason"] == "session_budget_guardrail_stop"
