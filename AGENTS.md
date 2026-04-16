# Hermes Agent - Development Guide

<!-- LEVEL 1: ALWAYS LOADED (~2000 chars) -->
<!-- This section is always loaded. Keep it concise and critical. -->
<!-- Tag: #agents-level1 -->

## Project Identity

Hermes Agent is an AI coding assistant built on NousResearch/hermes-agent. It uses tool calling to execute tasks, with skills stored as Markdown files in `~/.hermes/skills/`.

## Core Principles

1. **Work in context** — Always understand the project structure before making changes
2. **Verify before act** — Check existing code, tests, and documentation first
3. **Incremental progress** — Make small, verifiable changes
4. **Respect project conventions** — Follow existing patterns in the codebase

## Critical Rules

- **Activate venv first**: `source venv/bin/activate` before running Python
- **Use `get_hermes_home()`** — Never hardcode `~/.hermes` paths
- **Tests before push** — Run full test suite: `python -m pytest tests/ -q`
- **No profile hardcoding** — Each profile has its own `HERMES_HOME`
- **Prompt caching** — Never alter context mid-conversation
- **Sudo requires user confirmation** — Never run sudo or elevated commands without asking the user first (harness-5 self-evolving adaptation)

## Known Pitfalls (Must Read)

1. **Path hardcoding breaks profiles** — Use `get_hermes_home()` for state files
2. **Interactive menus break in tmux** — Use `curses` instead of `simple_term_menu`
3. **ANSI erase leaks in prompt_toolkit** — Use space-padding instead of `\033[K`
4. **Global `_last_resolved_tool_names`** — May be stale during subagent runs
5. **Tests write to temp dir** — Never hardcode `~/.hermes/` in tests

---

<!-- LEVEL 2: ON-DEMAND MODULES (~8000 chars) -->
<!-- Loaded when relevant module is being worked on -->
<!-- Tag: #agents-level2 -->

## Project Architecture

```
hermes-agent/
├── run_agent.py           # AIAgent class — core conversation loop
├── model_tools.py         # Tool orchestration & dispatch
├── agent/                 # Agent internals
│   ├── prompt_builder.py  # System prompt assembly
│   ├── context_compressor.py # Context window compression
│   ├── memory_manager.py  # Memory provider orchestration
│   ├── skill_utils.py     # Skill metadata parsing
│   └── skill_commands.py  # Skill slash commands
├── tools/                # Tool implementations (~50 tools)
│   ├── registry.py       # Central tool registry
│   ├── terminal_tool.py   # Terminal orchestration
│   ├── file_tools.py      # File read/write/search/patch
│   └── delegate_tool.py   # Subagent delegation
├── hermes_cli/           # CLI commands & setup
├── gateway/              # Messaging platform gateway
└── tests/               # Pytest suite (~3000 tests)
```

### Key File Patterns

**Tool registration** (`tools/your_tool.py`):
```python
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("YOUR_API_KEY"))

def your_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="your_tool",
    toolset="your_toolset",
    schema={"name": "your_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: your_tool(param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["YOUR_API_KEY"],
)
```

**Memory Provider** (`agent/memory_provider.py`):
```python
class YourProvider(MemoryProvider):
    async def initialize(self) -> None:
        # Connect to your storage
        pass
    
    def system_prompt_block(self) -> Optional[dict]:
        # Return static system prompt addition
        pass
    
    async def prefetch(self, query: str, session_id: str) -> str:
        # Return relevant context for query
        pass
    
    async def sync_turn(self, user: str, assistant: str) -> None:
        # Called after each conversation turn
        pass
    
    def get_tool_schemas(self) -> List[dict]:
        # Return tools exposed by this provider
        return []
```

## Adding New Components

### Adding a Tool

1. Create `tools/your_tool.py` with `registry.register()` call
2. Add to `_HERMES_CORE_TOOLS` or a new toolset in `toolsets.py`
3. Auto-discovery: any `tools/*.py` with `registry.register()` is auto-imported

### Adding a Slash Command

1. Add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`
2. Add handler in `HermesCLI.process_command()` in `cli.py`
3. Add gateway handler in `gateway/run.py` if platform-independent

### Adding Configuration

1. **config.yaml**: Add to `DEFAULT_CONFIG` in `hermes_cli/config.py`, bump `_config_version`
2. **.env**: Add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`

## Context File Loading Order

Context files are loaded in this order (each has max chars limit):

1. SOUL.md (identity, ~5000 chars)
2. .hermes.md / HERMES.md (git-root walk, ~8000 chars)
3. **AGENTS.md** (cwd only, ~15000 chars)
4. CLAUDE.md / claude.md (cwd only, ~6000 chars)
5. .cursorrules + .cursor/rules/*.mdc (cwd only, ~8000 chars)

## Profiles (Multi-Instance)

Hermes supports profiles — fully isolated instances each with own `HERMES_HOME`.

**Profile rules**:
- Use `get_hermes_home()` for all state paths
- Use `display_hermes_home()` for user-facing messages
- Profile operations are anchored to `~/.hermes/profiles`

---

<!-- LEVEL 3: TASK-SPECIFIC DETAILS (~12000 chars) -->
<!-- Loaded when specific task type is detected -->
<!-- Tag: #agents-level3 -->

## Testing Guide

### Run Full Test Suite
```bash
source venv/bin/activate
python -m pytest tests/ -q
```

### Run Specific Test Categories
```bash
python -m pytest tests/test_model_tools.py -q    # Tool resolution
python -m pytest tests/test_cli_init.py -q      # CLI config loading
python -m pytest tests/gateway/ -q             # Gateway tests
python -m pytest tests/tools/ -q                # Tool-level tests
```

### Profile Testing Pattern
```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

## Code Templates

### Adding a New Tool
```python
# tools/example_tool.py
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={
        "name": "example_tool",
        "description": "Does something useful. Requires EXAMPLE_API_KEY.",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "The parameter"}
            },
            "required": ["param"]
        }
    },
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""),
        task_id=kw.get("task_id")
    ),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

### Adding a Memory Provider
```python
# agent/example_memory_provider.py
from agent.memory_provider import MemoryProvider
from typing import List, Optional

class ExampleMemoryProvider(MemoryProvider):
    def __init__(self):
        self.storage = {}
    
    async def initialize(self) -> None:
        pass
    
    def system_prompt_block(self) -> Optional[dict]:
        return {
            "type": "text",
            "text": "You have access to example memory."
        }
    
    async def prefetch(self, query: str, session_id: str) -> str:
        return self.storage.get(session_id, "")
    
    async def sync_turn(self, user: str, assistant: str) -> None:
        pass
    
    def get_tool_schemas(self) -> List[dict]:
        return []
```

## Troubleshooting

### Context File Not Loading
- Check file name exactly: `AGENTS.md` or `agents.md` (cwd only)
- Check encoding: must be UTF-8
- Check size: files are truncated if > `CONTEXT_FILE_MAX_CHARS`

### Prompt Caching Broken
- NEVER alter past context mid-conversation
- NEVER reload memories or rebuild system prompts mid-conversation
- Only context compression modifies context, and only during compression trigger

### Profile Paths Wrong
- Use `get_hermes_home()` not `Path.home() / ".hermes"`
- Restart after changing `HERMES_HOME`

### Tool Not Found
- Check `registry.register()` is called at module import time
- Check tool is in an enabled toolset
- Check `check_fn` returns `True`

## Skill System

Skills are stored in `~/.hermes/skills/[category]/SKILL.md`:

```yaml
---
name: my-skill
description: What this skill does
version: 1.0.0
---
# My Skill

## When to Use
Brief description of when to use this skill.

## Procedure
Step-by-step instructions...

## Pitfalls
Common mistakes to avoid.

## Verification
How to verify the skill worked.
```

### Skill Discovery
- `skills_list()` — returns name + description only
- `skill_view(name)` — returns full skill content
- Progressive disclosure: don't load full skills until needed

---

<!-- END OF AGENTS.md -->
<!-- Progressive Disclosure: Level 1 always loaded, Level 2 on module focus, Level 3 on specific tasks -->
