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
| **Skills** | `skills_tool.py` | `skills_list`, `skill_view` |
| **Todo** | `todo_tool.py` | `todo` (read/write task list for multi-step planning) |
| **Cronjob** | `cronjob_tools.py` | `schedule_cronjob`, `list_cronjobs`, `remove_cronjob` |
| **RL Training** | `rl_training_tool.py` | `rl_list_environments`, `rl_start_training`, `rl_check_status`, etc. |

## Tool Registration

Tools are registered in `model_tools.py`:

```python
# model_tools.py
TOOL_SCHEMAS = [
    *WEB_TOOL_SCHEMAS,
    *TERMINAL_TOOL_SCHEMAS,
    *BROWSER_TOOL_SCHEMAS,
    # ...
]

TOOL_HANDLERS = {
    "web_search": web_search,
    "terminal": terminal_tool,
    "browser_navigate": browser_navigate,
    # ...
}
```

## Toolsets

Tools are grouped into **toolsets** for logical organization (see `toolsets.py`):

```python
TOOLSETS = {
    "web": {
        "description": "Web search and content extraction",
        "tools": ["web_search", "web_extract", "web_crawl"]
    },
    "terminal": {
        "description": "Command execution",
        "tools": ["terminal", "process"]
    },
    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": ["todo"]
    },
    # ...
}
```

## Adding a New Tool

1. Create handler function in `tools/your_tool.py`
2. Define JSON schema following OpenAI format
3. Register in `model_tools.py` (schemas and handlers)
4. Add to appropriate toolset in `toolsets.py`
5. Update `tools/__init__.py` exports

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

Skill directory structure:
```
skills/
└── mlops/
    └── axolotl/
        ├── SKILL.md           # Main instructions (required)
        ├── references/        # Additional docs
        ├── templates/         # Output formats, configs
        └── assets/            # Supplementary files (agentskills.io)
```

SKILL.md uses YAML frontmatter (agentskills.io compatible):
```yaml
---
name: axolotl
description: Fine-tuning LLMs with Axolotl
metadata:
  hermes:
    tags: [Fine-Tuning, LoRA, DPO]
---
```

## Skills Hub

The Skills Hub enables searching, installing, and managing skills from online registries. It is **user-driven only** — the model cannot search for or install skills.

**Sources:** GitHub repos (openai/skills, anthropics/skills, custom taps), ClawHub, Claude Code marketplaces, LobeHub.

**Security:** Every downloaded skill is scanned by `tools/skills_guard.py` (regex patterns + optional LLM audit) before installation. Trust levels: `builtin` (ships with Hermes), `trusted` (openai/skills, anthropics/skills), `community` (everything else — any findings = blocked unless `--force`).

**Architecture:**
- `tools/skills_guard.py` — Static scanner + LLM audit, trust-aware install policy
- `tools/skills_hub.py` — SkillSource ABC, GitHubAuth (PAT + App), 4 source adapters, lock file, hub state
- `hermes_cli/skills_hub.py` — Shared `do_*` functions, CLI subcommands, `/skills` slash command handler

**CLI:** `hermes skills search|install|inspect|list|audit|uninstall|publish|snapshot|tap`
**Slash:** `/skills search|install|inspect|list|audit|uninstall|publish|snapshot|tap`
