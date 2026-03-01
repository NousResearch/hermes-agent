# Tools

Tools are functions that extend the agent's capabilities. Each tool is defined with an OpenAI-compatible JSON schema and an async handler function.

## Tool Structure

Each tool module in `tools/` exports:
1. **Schema definitions** - OpenAI function-calling format
2. **Handler functions** - Async functions that execute the tool

```python
# Example: tools/web_tools.py

# Schema definition
WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}

# Handler function
async def web_search(query: str) -> dict:
    """Execute web search and return results."""
    # Implementation...
    return {"results": [...]}
```

## Tool Categories

| Category | Module | Tools |
|----------|--------|-------|
| **Web** | `web_tools.py` | `web_search`, `web_extract`, `web_crawl` |
| **Terminal** | `terminal_tool.py` | `terminal` (local/docker/singularity/modal/ssh backends) |
| **File** | `file_tools.py` | `read_file`, `write_file`, `patch`, `search` |
| **Browser** | `browser_tool.py` | `browser_navigate`, `browser_click`, `browser_type`, etc. |
| **Vision** | `vision_tools.py` | `vision_analyze` |
| **Image Gen** | `image_generation_tool.py` | `image_generate` |
| **TTS** | `tts_tool.py` | `text_to_speech` (Edge TTS free / ElevenLabs / OpenAI) |
| **Reasoning** | `mixture_of_agents_tool.py` | `mixture_of_agents` |
| **Skills** | `skills_tool.py`, `skill_manager_tool.py` | `skills_list`, `skill_view`, `skill_manage` |
| **Todo** | `todo_tool.py` | `todo` (read/write task list for multi-step planning) |
| **Memory** | `memory_tool.py` | `memory` (persistent notes + user profile across sessions) |
| **Session Search** | `session_search_tool.py` | `session_search` (search + summarize past conversations) |
| **Cronjob** | `cronjob_tools.py` | `schedule_cronjob`, `list_cronjobs`, `remove_cronjob` |
| **RL Training** | `rl_training_tool.py` | `rl_list_environments`, `rl_start_training`, `rl_check_status`, etc. |
| **Clarify** | `clarify_tool.py` | `clarify` (interactive multiple-choice / open-ended questions, CLI-only) |
| **Code Execution** | `code_execution_tool.py` | `execute_code` (run Python scripts that call tools via RPC sandbox) |
| **Delegation** | `delegate_tool.py` | `delegate_task` (spawn subagents with isolated context, single + parallel batch) |

## Tool Registration

Each tool file self-registers via `tools/registry.py`:

```python
# tools/example_tool.py
from tools.registry import registry

EXAMPLE_SCHEMA = {
    "name": "example_tool",
    "description": "Does something useful.",
    "parameters": { ... }
}

registry.register(
    name="example_tool",
    toolset="example",
    schema=EXAMPLE_SCHEMA,
    handler=lambda args, **kw: example_tool(args.get("param", "")),
    check_fn=check_example_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

`model_tools.py` is a thin orchestration layer that imports all tool modules (triggering registration), then delegates to the registry for schema collection and dispatch.

## Toolsets

Tools are grouped into **toolsets** for logical organization (see `toolsets.py`). All platforms share a `_HERMES_CORE_TOOLS` list; messaging platforms add `send_message`.

## Adding a New Tool

### Overview

Adding a tool touches 3 files:

1. **`tools/your_tool.py`** -- handler, schema, check function, `registry.register()` call
2. **`toolsets.py`** -- add tool name to `_HERMES_CORE_TOOLS` (or a specific toolset)
3. **`model_tools.py`** -- add `"tools.your_tool"` to the `_discover_tools()` list

### Step 1: Create the tool file

Every tool file follows the same structure: handler function, availability check, schema constant, and registry registration.

```python
# tools/weather_tool.py
"""Weather Tool -- look up current weather for a location."""

import json
import os
import logging

logger = logging.getLogger(__name__)


# --- Availability check ---

def check_weather_requirements() -> bool:
    """Return True if the tool's dependencies are available."""
    return bool(os.getenv("WEATHER_API_KEY"))


# --- Handler ---

def weather_tool(location: str, units: str = "metric") -> str:
    """Fetch weather for a location. Returns JSON string."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "WEATHER_API_KEY not configured"})
    try:
        # ... call weather API ...
        return json.dumps({"location": location, "temp": 22, "units": units})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Schema ---

WEATHER_SCHEMA = {
    "name": "weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates (e.g. 'London' or '51.5,-0.1')"
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Temperature units (default: metric)",
                "default": "metric"
            }
        },
        "required": ["location"]
    }
}


# --- Registration ---

from tools.registry import registry

registry.register(
    name="weather",
    toolset="weather",
    schema=WEATHER_SCHEMA,
    handler=lambda args, **kw: weather_tool(
        location=args.get("location", ""),
        units=args.get("units", "metric")),
    check_fn=check_weather_requirements,
    requires_env=["WEATHER_API_KEY"],
)
```

**Key rules:**

- Handlers MUST return a JSON string (via `json.dumps()`), never raw dicts.
- Errors MUST be returned as `{"error": "message"}`, never raised as exceptions. The registry's `dispatch()` also wraps unexpected exceptions automatically.
- The `check_fn` is called when building tool definitions -- if it returns `False`, the tool is silently excluded from the schema sent to the LLM.
- The `handler` receives `(args: dict, **kwargs)` where `args` is the LLM's tool call arguments and `kwargs` may include `task_id`, `user_task`, `store`, etc. depending on what the caller passes.

### Step 2: Add to a toolset

In `toolsets.py`, add the tool name to the appropriate place:

```python
# If it should be available on all platforms (CLI + messaging):
_HERMES_CORE_TOOLS = [
    ...
    "weather",  # <-- add here
]

# Or create a new standalone toolset:
"weather": {
    "description": "Weather lookup tools",
    "tools": ["weather"],
    "includes": []
},
```

### Step 3: Add discovery import

In `model_tools.py`, add the module to the `_discover_tools()` list:

```python
def _discover_tools():
    _modules = [
        ...
        "tools.weather_tool",  # <-- add here
    ]
```

This import triggers the `registry.register()` call at the bottom of the tool file.

### Async handlers

If your handler needs to call async code (e.g., `aiohttp`, async SDK), mark it with `is_async=True`:

```python
async def weather_tool_async(location: str) -> str:
    async with aiohttp.ClientSession() as session:
        ...
    return json.dumps(result)

registry.register(
    name="weather",
    toolset="weather",
    schema=WEATHER_SCHEMA,
    handler=lambda args, **kw: weather_tool_async(args.get("location", "")),
    check_fn=check_weather_requirements,
    is_async=True,  # <-- registry calls _run_async() automatically
)
```

The registry handles async bridging transparently via `_run_async()` -- you never call `asyncio.run()` yourself. This works correctly in CLI mode (no event loop), the gateway (running async loop), and RL environments (Atropos event loop + thread pool wrapping).

### Handlers that need task_id

Tools that manage per-session state (terminal, browser, file ops) receive `task_id` via `**kwargs`:

```python
def _handle_weather(args, **kw):
    task_id = kw.get("task_id")  # may be None in CLI mode
    return weather_tool(args.get("location", ""), task_id=task_id)

registry.register(
    name="weather",
    ...
    handler=_handle_weather,
)
```

Use a named function instead of a lambda when the arg unpacking is complex.

### Agent-loop intercepted tools

Some tools (todo, memory, session_search, delegate_task) need access to per-session agent state (TodoStore, MemoryStore, etc.) that doesn't flow through `handle_function_call`. These are intercepted by `run_agent.py` before reaching the registry. The registry still holds their schemas (so they appear in the tool list), but `dispatch()` returns a fallback error if the intercept is bypassed. See `todo_tool.py` for the pattern.

### Optional: setup wizard integration

If your tool requires an API key, add it to `hermes_cli/config.py`'s `OPTIONAL_ENV_VARS` dict so the setup wizard can prompt for it:

```python
OPTIONAL_ENV_VARS = {
    ...
    "WEATHER_API_KEY": {
        "description": "Weather API key for weather lookup",
        "prompt": "Weather API key",
        "url": "https://weatherapi.com/",
        "tools": ["weather"],
        "password": True,
    },
}
```

### Optional: batch processing

Add to `toolset_distributions.py` if the tool should be available in specific batch processing distributions.

## Stateful Tools

Some tools maintain state across calls within a session:

- **Terminal**: Keeps container/sandbox running between commands
- **Browser**: Maintains browser session for multi-step navigation

State is managed per `task_id` and cleaned up automatically.

## Terminal Backends

The terminal tool supports multiple execution backends:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `local` | Direct execution on host | Development, simple tasks |
| `ssh` | Remote execution via SSH | Sandboxing (agent can't modify its own code) |
| `docker` | Docker container | Isolation, reproducibility |
| `singularity` | Singularity/Apptainer | HPC clusters, rootless containers |
| `modal` | Modal cloud | Scalable cloud compute, GPUs |

Configure via environment variables or `cli-config.yaml`:

```yaml
# SSH backend example (in cli-config.yaml)
terminal:
  env_type: "ssh"
  ssh_host: "my-server.example.com"
  ssh_user: "myuser"
  ssh_key: "~/.ssh/id_rsa"
  cwd: "/home/myuser/project"
```

The SSH backend uses ControlMaster for connection persistence, making subsequent commands fast.

## Skills Tools (Progressive Disclosure)

Skills are on-demand knowledge documents. They use **progressive disclosure** to minimize tokens:

```
Level 0: skills_categories()     → ["mlops", "devops"]           (~50 tokens)
Level 1: skills_list(category)   → [{name, description}, ...]   (~3k tokens)
Level 2: skill_view(name)        → Full content + metadata       (varies)
Level 3: skill_view(name, path)  → Specific reference file       (varies)
```

All skills live in `~/.hermes/skills/` — a single directory that serves as the source of truth. On fresh install, bundled skills are seeded from the repo's `skills/` directory. Hub-installed and agent-created skills also go here. The agent can modify or delete any skill.

Skill directory structure:
```
~/.hermes/skills/
├── mlops/
│   └── axolotl/
│       ├── SKILL.md             # Main instructions (required)
│       ├── references/          # Additional docs
│       ├── templates/           # Output formats, configs
│       └── assets/              # Supplementary files (agentskills.io)
├── devops/
│   └── deploy-k8s/
│       └── SKILL.md
├── .hub/                        # Skills Hub state
└── .bundled_manifest            # Tracks seeded bundled skills
```

SKILL.md uses YAML frontmatter (agentskills.io compatible):
```yaml
---
name: axolotl
description: Fine-tuning LLMs with Axolotl
metadata:
  hermes:
    tags: [Fine-Tuning, LoRA, DPO]
    category: mlops
---
```

## Skill Management (skill_manage)

The `skill_manage` tool lets the agent create, update, and delete its own skills -- turning successful approaches into reusable procedural knowledge.

**Module:** `tools/skill_manager_tool.py`

**Actions:**
| Action | Description | Required params |
|--------|-------------|-----------------|
| `create` | Create new skill (SKILL.md + directory) | `name`, `content`, optional `category` |
| `patch` | Targeted find-and-replace in SKILL.md or supporting file | `name`, `old_string`, `new_string`, optional `file_path`, `replace_all` |
| `edit` | Full replacement of SKILL.md (major rewrites only) | `name`, `content` |
| `delete` | Remove a user skill entirely | `name` |
| `write_file` | Add/overwrite a supporting file | `name`, `file_path`, `file_content` |
| `remove_file` | Remove a supporting file | `name`, `file_path` |

### Patch vs Edit

`patch` and `edit` both modify skill files, but serve different purposes:

**`patch`** (preferred for most updates):
- Targeted `old_string` → `new_string` replacement, same interface as the `patch` file tool
- Token-efficient: only the changed text appears in the tool call, not the full file
- Requires unique match by default; set `replace_all=true` for global replacements
- Returns match count on ambiguous matches so the model can add more context
- When targeting SKILL.md, validates that frontmatter remains intact after the patch
- Also works on supporting files via `file_path` parameter (e.g., `references/api.md`)
- Returns a file preview on not-found errors for self-correction without extra reads

**`edit`** (for major rewrites):
- Full replacement of SKILL.md content
- Use when the skill's structure needs to change (reorganizing sections, rewriting from scratch)
- The model should `skill_view()` first, then provide the complete updated text

**Constraints:**
- All skills live in `~/.hermes/skills/` and can be modified or deleted
- Skill names must be lowercase, filesystem-safe (`[a-z0-9._-]+`), max 64 chars
- SKILL.md must have valid YAML frontmatter with `name` and `description` fields
- Supporting files must be under `references/`, `templates/`, `scripts/`, or `assets/`
- Path traversal (`..`) in file paths is blocked

**Availability:** Enabled by default in CLI, Telegram, Discord, WhatsApp, and Slack. Not included in batch_runner or RL training environments.

**Behavioral guidance:** The tool description teaches the model when to create skills (after difficult tasks), when to update them (stale/broken instructions), to prefer `patch` over `edit` for targeted fixes, and the feedback loop pattern (ask user after difficult tasks, offer to save as a skill).

## Skills Hub

The Skills Hub enables searching, installing, and managing skills from online registries. It is **user-driven only** — the model cannot search for or install skills.

**Sources:** GitHub repos (openai/skills, anthropics/skills, custom taps), ClawHub, Claude Code marketplaces, LobeHub.

**Security:** Every downloaded skill is scanned by `tools/skills_guard.py` (regex patterns + optional LLM audit) before installation. Trust levels: `builtin` (ships with Hermes), `trusted` (openai/skills, anthropics/skills), `community` (everything else — any findings = blocked unless `--force`).

**Architecture:**
- `tools/skills_guard.py` — Static scanner + LLM audit, trust-aware install policy
- `tools/skills_hub.py` — SkillSource ABC, GitHubAuth (PAT + App), 4 source adapters, lock file, hub state
- `tools/skill_manager_tool.py` — Agent-managed skill CRUD (`skill_manage` tool)
- `hermes_cli/skills_hub.py` — Shared `do_*` functions, CLI subcommands, `/skills` slash command handler

**CLI:** `hermes skills search|install|inspect|list|audit|uninstall|publish|snapshot|tap`
**Slash:** `/skills search|install|inspect|list|audit|uninstall|publish|snapshot|tap`
