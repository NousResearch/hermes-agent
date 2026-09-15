"""Legacy positional construction must not be shifted by ACP-only options."""

import pytest


@pytest.mark.parametrize("entrypoint", ["constructor", "init_agent"])
def test_legacy_positional_model_and_iteration_budget(entrypoint):
    from agent.agent_init import init_agent
    from run_agent import AIAgent

    # The public constructor and extracted initializer shared these slots before
    # acp_cwd was added. Exercise real initialization, not just signature binding.
    args = (
        "http://127.0.0.1:9/v1", "test-only", "custom", "chat_completions",
        None, None, None, None, "legacy-model", 3,
    )
    flags = dict(
        quiet_mode=True, enabled_toolsets=[], skip_context_files=True,
        skip_memory=True, skip_background_review=True,
    )
    if entrypoint == "constructor":
        agent = AIAgent(*args, **flags)
    else:
        agent = AIAgent.__new__(AIAgent)
        init_agent(agent, *args, **flags)
    try:
        assert agent.model == args[-2]
        assert agent.max_iterations == args[-1]
        assert agent.iteration_budget.remaining == args[-1]
        assert agent.acp_cwd is None
    finally:
        agent.close()
