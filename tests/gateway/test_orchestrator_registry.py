from gateway.orchestrator.registry import AgentKind, KNOWN_AGENTS, get_agent_spec


EXPECTED_OPERATIONAL_AGENTS = {"ccd", "codex", "ccg", "ccm"}


def test_known_agents_are_limited_to_operational_parallel_lanes():
    specs = {spec.name: spec for spec in KNOWN_AGENTS}

    assert set(specs) == EXPECTED_OPERATIONAL_AGENTS
    assert specs["ccd"].kind is AgentKind.SHELL_FUNCTION
    assert specs["codex"].kind is AgentKind.BINARY
    assert specs["codex"].sandbox is True
    assert specs["ccg"].kind is AgentKind.SHELL_FUNCTION
    assert specs["ccm"].kind is AgentKind.SHELL_FUNCTION
    assert specs["ccm"].secrets is True


def test_get_agent_spec_returns_only_operational_parallel_specs():
    assert get_agent_spec("codex").name == "codex"
    assert get_agent_spec("ccd").name == "ccd"
    assert get_agent_spec("claude") is None
    assert get_agent_spec("emd") is None
    assert get_agent_spec("missing") is None
