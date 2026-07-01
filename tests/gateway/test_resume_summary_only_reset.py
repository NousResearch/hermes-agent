from gateway.run import _clear_resume_summary_only_for_human_turn


def test_next_human_turn_clears_resume_summary_only_before_tools_run():
    class Agent:
        _resume_summary_only = True

    agent = Agent()

    _clear_resume_summary_only_for_human_turn(agent, is_resume_pending=False, message="go")

    assert agent._resume_summary_only is False
