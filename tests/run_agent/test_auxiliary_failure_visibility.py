from run_agent import AIAgent


def _make_agent():
    agent = AIAgent.__new__(AIAgent)
    warnings = []
    agent._emit_warning = warnings.append
    agent._summarize_api_error = lambda exc: str(exc)
    return agent, warnings


def test_title_generation_aux_failure_not_user_visible():
    agent, warnings = _make_agent()
    agent._emit_auxiliary_failure("title generation", TypeError("NoneType object is not iterable"))
    assert warnings == []


def test_non_title_aux_failure_still_user_visible():
    agent, warnings = _make_agent()
    agent._emit_auxiliary_failure("vision", RuntimeError("provider timeout"))
    assert len(warnings) == 1
    assert warnings[0].startswith("⚠ Auxiliary vision failed:")
