from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE, TOOL_USE_ENFORCEMENT_GUIDANCE


def test_tool_use_guidance_prioritizes_main_request_over_housekeeping():
    assert "execute the main request before side-channel housekeeping" in TOOL_USE_ENFORCEMENT_GUIDANCE
    assert "Memory, todo, and skill updates must not displace" in TOOL_USE_ENFORCEMENT_GUIDANCE


def test_execution_guidance_requires_project_status_baseline():
    assert "Project status / continue-work requests" in OPENAI_MODEL_EXECUTION_GUIDANCE
    assert "current disk and git evidence as higher authority than memory" in OPENAI_MODEL_EXECUTION_GUIDANCE
