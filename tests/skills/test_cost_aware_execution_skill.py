import pytest
from agent.skill_commands import SkillCommandProcessor
from agent.registry import SkillRegistry

def test_cost_aware_execution_registration():
    """Ensure the cost-aware-execution skill is properly discovered and registered."""
    registry = SkillRegistry()
    registry.load_skills()
    
    # Verify the skill is registered under the correct normalized name
    assert "cost-aware-execution" in registry.get_available_skills()
    
    skill = registry.get_skill("cost-aware-execution")
    assert skill.name == "cost-aware-execution"
    # Verify the description complies with the shortened length rule
    assert len(skill.description) <= 60

def test_cost_aware_execution_activation():
    """Test that invoking the slash command correctly activates the meta-skill context."""
    processor = SkillCommandProcessor()
    context = processor.create_mock_context()
    
    # Simulate user sending the registered slash command
    response = processor.execute_command("/cost-aware-execution", context=context)
    
    # Assert execution state flags are correctly updated for tool budget constraint
    assert context.agent_state.active_skills.contains("cost-aware-execution")
    assert context.agent_state.prefer_direct_answers is True
    assert "Activated cost-aware-execution" in response

def test_tool_suppression_logic():
    """Ensure that general knowledge queries do not trigger native tools when the skill is active."""
    processor = SkillCommandProcessor()
    context = processor.create_mock_context()
    
    # Activate the skill context
    processor.execute_command("/cost-aware-execution", context=context)
    
    # Mock a simple arithmetic task evaluation
    should_use_tool = context.agent_state.evaluate_tool_necessity(
        query="What is 25 * 17?", 
        native_tools=["web_search", "python_interpreter"]
    )
    
    # The meta-skill must suppress tool usage for simple tasks
    assert should_use_tool is False
