def test_env_var_disables_nudge_at_init(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_SKILL_NUDGE_DISABLE", "1")
    a = AIAgent.__new__(AIAgent)
    AIAgent._init_nudge_state(a, {"skills": {"nudge_signals": {"enabled": True}}})

    assert a._nudge_disabled is True


def test_disabled_session_blocks_signal_gate():
    from run_agent import AIAgent

    a = AIAgent.__new__(AIAgent)
    AIAgent._init_nudge_state(a, {"skills": {"nudge_signals": {"enabled": True}}})
    a._nudge_disabled = True
    a.valid_tool_names = {"skill_manage"}
    a._iters_since_skill = 999
    a._skill_nudge_interval = 50
    a._signal_evaluator.fired_signals.add("S1")

    res = AIAgent._compute_should_review_skills(a)

    assert res == (False, set())
