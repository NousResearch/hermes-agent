# Feature Map Reference

Quick reference for mapping Hermes features to their implementation paths across surfaces.

## CLI Surface (`cli.py`)

### Entry Points

| Feature | Method | File | Notes |
|---------|--------|------|-------|
| Slash commands | `process_command()` | `cli.py` | Dispatch via `resolve_command()` |
| Interactive loop | `run()` | `cli.py` | Main REPL loop |
| Agent instantiation | `_create_agent()` | `cli.py` | AIAgent setup |
| Config loading | `load_cli_config()` | `cli.py` | Merges defaults + user YAML |
| Skill loading | `load_skill()` | `agent/skill_commands.py` | Injected as user message |
| Tool discovery | `discover_builtin_tools()` | `model_tools.py` | Auto-imports `tools/*.py` |

### Common Test Patterns

```python
# Test slash command
def test_slash_command_cli():
    cli = HermesCLI(config={...}, quiet_mode=True)
    result = cli.process_command("/testcmd arg")
    assert expected_output in result

# Test agent conversation
def test_agent_chat():
    agent = AIAgent(...)
    response = agent.chat("test message")
    assert "expected" in response
```

## TUI Surface (`ui-tui/`)

### Entry Points

| Feature | Component/Method | File | Notes |
|---------|------------------|------|-------|
| JSON-RPC gateway | `TUIGateway` | `tui_gateway/server.py` | Stdio transport |
| Prompt submission | `prompt.submit` | `tui_gateway/server.py` | Triggers agent turn |
| Slash execution | `slash.exec` | `tui_gateway/server.py` | Persistent worker subprocess |
| Session management | `session.list/resume` | `tui_gateway/server.py` | Session ops |
| React components | `<App />` | `ui-tui/src/app.tsx` | Main UI tree |
| State atoms | Nanostores | `ui-tui/src/store/` | Client-side state |

### Common Test Patterns

```python
# Test JSON-RPC method
def test_tui_rpc_method():
    gateway = TUIGateway()
    response = gateway.handle_request({
        "method": "prompt.submit",
        "params": {"text": "test"}
    })
    assert response["type"] == "message.delta"

# Test slash command routing
def test_tui_slash():
    response = gateway.handle_request({
        "method": "slash.exec",
        "params": {"command": "/test arg"}
    })
    assert response["result"]["type"] == "skill"
```

## Gateway Surface (`gateway/`)

### Entry Points

| Feature | Method | File | Notes |
|---------|--------|------|-------|
| Message handling | `_process_message_background()` | `gateway/run.py` | Async message queue |
| Command dispatch | `_handle_command()` | `gateway/run.py` | Slash command routing |
| Platform adapters | `PlatformAdapter` subclasses | `gateway/platforms/*/adapter.py` | Per-platform |
| Multiplex profiles | `_profile_runtime_scope` | `gateway/run.py` | Profile isolation |
| Toolset loading | `_load_enabled_toolsets()` | `gateway/run.py` | Platform-specific tools |

### Common Test Patterns

```python
# Test gateway message
def test_gateway_message():
    gateway = GatewayRunner(config={...})
    event = MockMessageEvent(
        platform="test",
        text="/testcmd arg",
        sender_id="user123"
    )
    result = await gateway.handle_message(event)
    assert result.success

# Test platform adapter
def test_platform_adapter():
    adapter = TestPlatformAdapter(config={...})
    await adapter.start()
    result = await adapter.send_message("test", channel_id="ch123")
    assert result.message_id
```

## Tool Registration

### Core Tools (`toolsets.py`)

Tools in `_HERMES_CORE_TOOLS` are available to all platforms by default:

```python
_HERMES_CORE_TOOLS = [
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "terminal",
    "web_search",
    # ... etc
]
```

### Custom Toolsets

Platform-specific tools live in named toolsets:

```python
TOOLSETS = {
    "messaging": _HERMES_CORE_TOOLS + [
        "send_message",
        "react_to_message",
    ],
    "browser": [
        "browser_navigate",
        "browser_screenshot",
    ],
    # ... etc
}
```

### Tool Discovery

```python
# Auto-discovery imports all tools/*.py files
# Each calls registry.register() at module load
from tools.registry import registry

def test_tool_discovery():
    tools = discover_builtin_tools()
    tool_names = [t["name"] for t in tools]
    assert "my_new_tool" in tool_names
```

## Configuration

### Config Loading Paths

| Context | Loader | Location | Includes CLI defaults? |
|---------|--------|----------|------------------------|
| CLI | `load_cli_config()` | `cli.py` | Yes |
| Subcommands | `load_config()` | `hermes_cli/config.py` | No |
| Gateway | Direct YAML | `gateway/config.py` | No |

### Test Pattern

```python
def test_config_load(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
display:
  skin: test-skin
memory:
  provider: test
""")
    os.environ["HERMES_HOME"] = str(tmp_path)
    config = load_config()
    assert config["display"]["skin"] == "test-skin"
```

## Memory Integration

### Memory Provider Hooks

```python
# Called on each agent turn
memory_provider.sync_turn(turn_messages)

# Called before agent processes message
memory_provider.prefetch(user_message)

# Called on session end
memory_provider.shutdown()
```

### Test Pattern

```python
def test_memory_integration():
    provider = TestMemoryProvider(config={...})
    provider.sync_turn([
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"}
    ])
    
    # Verify memory stored
    memories = provider.search("test")
    assert len(memories) > 0
```

## Skills

### Skill Loading

```python
# Skills loaded at conversation start
# Injected as USER message (preserves cache)
skill_content = load_skill("skill-name")
messages.append({"role": "user", "content": skill_content})
```

### Skill Slash Commands

```python
# Agent scans ~/.hermes/skills/ for SKILL.md files
# Generates slash commands from skill names
# Example: skills/test-skill/ → /test-skill

def test_skill_slash():
    result = cli.process_command("/test-skill")
    assert "skill loaded" in result.lower()
```

## Plugin System

### Plugin Discovery

```python
# Plugins discovered from:
# 1. ~/.hermes/plugins/
# 2. ./.hermes/plugins/ (opt-in)
# 3. pip entry points: hermes_agent.plugins

from hermes_cli.plugins import discover_plugins

def test_plugin_discovery():
    plugins = discover_plugins()
    assert "test-plugin" in [p.name for p in plugins]
```

### Plugin Hooks

```python
# Available hooks:
# - pre_tool_call
# - post_tool_call
# - pre_llm_call
# - post_llm_call
# - on_session_start
# - on_session_end

def test_plugin_hook():
    plugin = TestPlugin()
    plugin.register(ctx)
    
    # Verify hook registered
    assert plugin in plugin_manager.hooks["pre_tool_call"]
```

## Test Utilities

### Isolated HERMES_HOME

```python
@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home

def test_with_isolation(isolated_hermes_home):
    # Test runs with clean slate
    # No pollution from real ~/.hermes/
    pass
```

### Mock Message Events

```python
class MockMessageEvent:
    def __init__(self, text, sender_id="test", platform="test"):
        self.text = text
        self.sender_id = sender_id
        self.platform = platform
        self.timestamp = datetime.now()

def test_with_mock_event():
    event = MockMessageEvent("/test command")
    result = handler.process(event)
    assert result.success
```

## Quick Verification Checklist

- [ ] CLI: Direct `HermesCLI` instantiation works
- [ ] TUI: JSON-RPC methods return expected structure
- [ ] Gateway: Message events route correctly
- [ ] Tools: Discovery finds new/modified tools
- [ ] Config: Loading works across all loaders
- [ ] Memory: Provider hooks called correctly
- [ ] Skills: Loadable and slash commands work
- [ ] Plugins: Discovery and hooks registered

## Common Pitfalls

1. **Config Loader Mismatch**: CLI sees new config key but gateway doesn't → check `DEFAULT_CONFIG` in `hermes_cli/config.py`

2. **Tool Not Discovered**: Tool file exists but not in schema → add to toolset in `toolsets.py`

3. **Surface-Specific Code**: Using `print()` for output (breaks gateway) or checking `HERMES_DESKTOP` env (breaks remote topologies)

4. **Cache Invalidation**: Modifying system prompt mid-conversation → start fresh session or defer change

5. **Test Pollution**: Tests leave session DB / config files → use isolated `HERMES_HOME` fixture
