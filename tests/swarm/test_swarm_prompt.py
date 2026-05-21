from agent.swarm_prompt import build_swarm_operator_prompt


def test_disabled_config_does_not_alter_prompt():
    assert build_swarm_operator_prompt({"enabled": False}) == ""
    assert build_swarm_operator_prompt({}) == ""


def test_enabled_config_includes_operator_rules_and_limits():
    prompt = build_swarm_operator_prompt({"enabled": True, "max_children": 2})

    assert "Jeeves Swarm Operator Contract" in prompt
    assert "single parent operator" in prompt
    assert "max_children=2" in prompt
    assert "Permission-required subtasks stay blocked" in prompt
    assert "Child agents must never send messages" in prompt


def test_prompt_includes_script_threshold_contract():
    prompt = build_swarm_operator_prompt({"enabled": True})

    assert "scripts for >3 procedural steps/source-of-truth/validation/eval" in prompt


def test_prompt_includes_dumb_pipe_policy():
    prompt = build_swarm_operator_prompt({"enabled": True})

    assert "n8n/docker are dumb pipes" in prompt
    assert "They are not reasoning engines" in prompt
