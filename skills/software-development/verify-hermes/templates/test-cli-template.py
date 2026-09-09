# Test Template: CLI Feature

Template for testing CLI surface features.

## Basic Slash Command Test

```python
import pytest
from cli import HermesCLI
from pathlib import Path


def test_new_slash_command_help(isolated_hermes_home):
    """Test /newcommand shows help text."""
    cli = HermesCLI(
        config={},
        quiet_mode=True,
        session_id="test-session"
    )
    
    result = cli.process_command("/newcommand --help")
    
    assert "Usage:" in result
    assert "newcommand" in result.lower()


def test_new_slash_command_execution(isolated_hermes_home):
    """Test /newcommand executes correctly."""
    cli = HermesCLI(
        config={},
        quiet_mode=True,
        session_id="test-session"
    )
    
    result = cli.process_command("/newcommand arg1 arg2")
    
    assert "expected output" in result
    # Verify side effects
    assert (isolated_hermes_home / "expected_file").exists()
```

## Agent Conversation Test

```python
def test_agent_conversation_flow(isolated_hermes_home):
    """Test multi-turn conversation."""
    from run_agent import AIAgent
    
    agent = AIAgent(
        model="test-model",
        max_iterations=10,
        quiet_mode=True,
        session_id="test-session",
        skip_memory=True  # Isolate from memory providers
    )
    
    # Turn 1
    response1 = agent.chat("First message")
    assert len(response1) > 0
    
    # Turn 2 (with history)
    response2 = agent.chat("Second message")
    assert len(response2) > 0
    
    # Verify session stored
    from hermes_state import SessionDB
    db = SessionDB(session_id="test-session")
    messages = db.get_messages()
    assert len(messages) >= 4  # 2 user + 2 assistant
```

## Tool Execution Test

```python
def test_tool_execution_via_agent():
    """Test agent can call new tool."""
    from model_tools import discover_builtin_tools, handle_function_call
    import json
    
    # Verify tool discovered
    tools = discover_builtin_tools()
    tool_names = [t["name"] for t in tools]
    assert "new_tool" in tool_names
    
    # Execute tool
    result = handle_function_call(
        "new_tool",
        {"param": "value"},
        task_id="test-task"
    )
    
    parsed = json.loads(result)
    assert parsed["success"] == True
```

## Config Test

```python
def test_config_cli_loading(tmp_path, monkeypatch):
    """Test CLI config loading."""
    # Setup
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
display:
  new_setting: test_value
""")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    # Load
    from cli import load_cli_config
    config = load_cli_config()
    
    # Verify
    assert config["display"]["new_setting"] == "test_value"
```

## Skill Loading Test

```python
def test_skill_loading(isolated_hermes_home):
    """Test skill loads correctly."""
    # Create test skill
    skill_dir = isolated_hermes_home / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("""
---
name: test-skill
description: "Test skill."
---

# Test Skill

Test content.
""")
    
    # Load skill
    from agent.skill_commands import load_skill
    content = load_skill("test-skill")
    
    assert "Test Skill" in content
    assert "Test content" in content
```

## Fixtures

```python
@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Provide isolated HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}")
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


@pytest.fixture
def mock_agent_config():
    """Minimal agent config for testing."""
    return {
        "model": "test-model",
        "provider": "test",
        "max_iterations": 5,
        "quiet_mode": True,
    }
```

## Notes

- Always use `isolated_hermes_home` fixture to avoid polluting real config
- Set `quiet_mode=True` to suppress spinner output in tests
- Use `skip_memory=True` for agent tests that don't need memory
- Check both stdout and side effects (files created, DB entries, etc.)
