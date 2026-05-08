def _agent_with_state(**kwargs):
    from agent.skill_nudge_signals import SignalEvaluator
    from run_agent import AIAgent

    a = AIAgent.__new__(AIAgent)
    a.valid_tool_names = {"skill_manage"}
    a._skill_nudge_interval = kwargs.get("interval", 50)
    a._iters_since_skill = kwargs.get("iters", 0)
    a._nudge_disabled = kwargs.get("disabled", False)
    a._nudge_signals_enabled = kwargs.get("signals_enabled", True)
    a._signal_evaluator = SignalEvaluator(
        repeated_threshold=3,
        error_threshold=2,
        common_clis_suppressed=[],
        cli_window_days=30,
        user_phrases=[],
    )
    if kwargs.get("fired"):
        a._signal_evaluator.fired_signals.update(kwargs["fired"])
    return a


def test_gate_fires_when_signal_present():
    from run_agent import AIAgent

    a = _agent_with_state(fired={"S1"})
    res = AIAgent._compute_should_review_skills(a)

    assert res == (True, {"S1"})


def test_gate_falls_back_to_time_when_no_signal():
    from run_agent import AIAgent

    a = _agent_with_state(iters=50, interval=50)
    res = AIAgent._compute_should_review_skills(a)

    assert res == (True, set())


def test_gate_blocked_by_disable():
    from run_agent import AIAgent

    a = _agent_with_state(fired={"S1"}, disabled=True)
    res = AIAgent._compute_should_review_skills(a)

    assert res == (False, set())


def test_gate_blocked_when_skill_manage_unavailable():
    from run_agent import AIAgent

    a = _agent_with_state(fired={"S1"})
    a.valid_tool_names = set()
    res = AIAgent._compute_should_review_skills(a)

    assert res == (False, set())


def test_gate_does_not_fire_disabled_signal_mode_before_fallback():
    from run_agent import AIAgent

    a = _agent_with_state(fired={"S1"}, signals_enabled=False, iters=2, interval=50)
    res = AIAgent._compute_should_review_skills(a)

    assert res == (False, set())
