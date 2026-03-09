# Hermes Agent - Development Guide

Instructions for AI coding assistants (GitHub Copilot, Cursor, etc.) and human developers.

Hermes Agent is an AI agent harness with tool-calling capabilities, interactive CLI, messaging integrations, and scheduled tasks.

## Development Environment

**IMPORTANT**: Always use the virtual environment if it exists:
```bash
source venv/bin/activate  # Before running any Python commands
```

## Project Structure

```
hermes-agent/
├── agent/                # Agent internals (extracted from run_agent.py)
│   ├── auxiliary_client.py   # Shared auxiliary OpenAI client (vision, compression, web extract)
│   ├── model_metadata.py     # Model context lengths, token estimation
│   ├── context_compressor.py # Auto context compression
│   ├── prompt_caching.py     # Anthropic prompt caching
│   ├── prompt_builder.py     # System prompt assembly (identity, skills index, context files)
│   ├── display.py            # KawaiiSpinner, tool preview formatting
│   └── trajectory.py         # Trajectory saving helpers
├── hermes_cli/           # CLI implementation
│   ├── main.py           # Entry point, command dispatcher
│   ├── banner.py         # Welcome banner, ASCII art, skills summary
│   ├── commands.py       # Slash command definitions + autocomplete
│   ├── callbacks.py      # Interactive prompt callbacks (clarify, sudo, approval)
│   ├── setup.py          # Interactive setup wizard
│   ├── config.py         # Config management & migration
│   ├── status.py         # Status display
│   ├── doctor.py         # Diagnostics
│   ├── gateway.py        # Gateway management
│   ├── uninstall.py      # Uninstaller
│   ├── cron.py           # Cron job management
│   └── skills_hub.py     # Skills Hub CLI + /skills slash command
├── tools/                # Tool implementations
│   ├── registry.py            # Central tool registry (schemas, handlers, dispatch)
│   ├── approval.py            # Dangerous command detection + per-session approval
│   ├── environments/          # Terminal execution backends
│   │   ├── base.py            # BaseEnvironment ABC
│   │   ├── local.py           # Local execution with interrupt support
│   │   ├── docker.py          # Docker container execution
│   │   ├── ssh.py             # SSH remote execution
│   │   ├── singularity.py     # Singularity/Apptainer + SIF management
│   │   ├── modal.py           # Modal cloud execution
│   │   └── daytona.py         # Daytona cloud sandboxes
│   ├── terminal_tool.py       # Terminal orchestration (sudo, lifecycle, factory)
│   ├── todo_tool.py           # Planning & task management
│   ├── process_registry.py    # Background process management
│   └── ...                    # Other tool files
├── gateway/              # Messaging platform adapters
│   ├── platforms/        # Platform-specific adapters (telegram, discord, slack, whatsapp)
│   └── ...
├── cron/                 # Scheduler implementation
├── environments/         # RL training environments (Atropos integration)
├── skills/               # Bundled skill sources
├── optional-skills/      # Official optional skills (not activated by default)
├── cli.py                # Interactive CLI orchestrator (HermesCLI class)
├── hermes_state.py       # SessionDB — SQLite session store (schema, titles, FTS5 search)
├── run_agent.py          # AIAgent class (core conversation loop)
├── model_tools.py        # Tool orchestration (thin layer over tools/registry.py)
├── toolsets.py           # Tool groupings
├── toolset_distributions.py  # Probability-based tool selection
└── batch_runner.py       # Parallel batch processing
```

**User Configuration** (stored in `~/.hermes/`):
- `~/.hermes/config.yaml` - Settings (model, terminal, toolsets, etc.)
- `~/.hermes/.env` - API keys and secrets
- `~/.hermes/pairing/` - DM pairing data
- `~/.hermes/hooks/` - Custom event hooks
- `~/.hermes/image_cache/` - Cached user images
- `~/.hermes/audio_cache/` - Cached user voice messages
- `~/.hermes/sticker_cache.json` - Telegram sticker descriptions

## File Dependency Chain

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

Each tool file co-locates its schema, handler, and registration. `model_tools.py` is a thin orchestration layer.

---

## AIAgent Class

The main agent is implemented in `run_agent.py`:

```python
class AIAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4.6",
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_iterations: int = 60,        # Max tool-calling loops
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        verbose_logging: bool = False,
        quiet_mode: bool = False,         # Suppress progress output
        tool_progress_callback: callable = None,  # Called on each tool use
    ):
        # Initialize OpenAI client, load tools based on toolsets
        ...
    
    def chat(self, user_message: str, task_id: str = None) -> str:
        # Main entry point - runs the agent loop
        ...
```

### Agent Loop

The core loop in `_run_agent_loop()`:

```
1. Add user message to conversation
2. Call LLM with tools
3. If LLM returns tool calls:
   - Execute each tool
   - Add tool results to conversation
   - Go to step 2
4. If LLM returns text response:
   - Return response to user
```

```python
while turns < max_turns:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,
    )
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = await execute_tool(tool_call)
            messages.append(tool_result_message(result))
        turns += 1
    else:
        return response.content
```

### Conversation Management

Messages are stored as a list of dicts following OpenAI format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Search for Python tutorials"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "Here's what I found..."},
]
```

### Reasoning Model Support

For models that support chain-of-thought reasoning:
- Extract `reasoning_content` from API responses
- Store in `assistant_msg["reasoning"]` for trajectory export
- Pass back via `reasoning_content` field on subsequent turns

---

## CLI Architecture (cli.py)

The interactive CLI uses:
- **Rich** - For the welcome banner and styled panels
- **prompt_toolkit** - For fixed input area with history, `patch_stdout`, slash command autocomplete, and floating completion menus
- **KawaiiSpinner** (in run_agent.py) - Animated kawaii faces during API calls; clean `┊` activity feed for tool execution results

Key components:
- `HermesCLI` class - Main CLI controller with commands and conversation loop
- `SlashCommandCompleter` - Autocomplete dropdown for `/commands` (type `/` to see all)
- `agent/skill_commands.py` - Scans skills and builds invocation messages (shared with gateway)
- `load_cli_config()` - Loads config, sets environment variables for terminal
- `build_welcome_banner()` - Displays ASCII art logo, tools, and skills summary
- `_preload_resumed_session()` - Loads session history early (before banner) for immediate display on resume
- `_display_resumed_history()` - Renders a compact conversation recap in a Rich Panel on session resume

CLI UX notes:
- Thinking spinner (during LLM API call) shows animated kawaii face + verb (`(⌐■_■) deliberating...`)
- When LLM returns tool calls, the spinner clears silently (no "got it!" noise)
- Tool execution results appear as a clean activity feed: `┊ {emoji} {verb} {detail} {duration}`
- "got it!" only appears when the LLM returns a final text response (`⚕ ready`)
- The prompt shows `⚕ ❯` when the agent is working, `❯` when idle
- Pasting 5+ lines auto-saves to `~/.hermes/pastes/` and collapses to a reference
- Multi-line input via Alt+Enter or Ctrl+J
- When resuming a session (`--continue`/`--resume`), a "Previous Conversation" panel shows previous messages before the input prompt (configurable via `display.resume_display`)
- `/commands` - Process user commands like `/help`, `/clear`, `/personality`, etc.
- `/skill-name` - Invoke installed skills directly (e.g., `/axolotl`, `/gif-search`)

CLI uses `quiet_mode=True` when creating AIAgent to suppress verbose logging.

### Skill Slash Commands

Every installed skill in `~/.hermes/skills/` is automatically registered as a slash command.
The skill name (from frontmatter or folder name) becomes the command: `axolotl` → `/axolotl`.

Implementation (`agent/skill_commands.py`, shared between CLI and gateway):
1. `scan_skill_commands()` scans all SKILL.md files at startup, filtering out skills incompatible with the current OS platform (via the `platforms` frontmatter field)
2. `build_skill_invocation_message()` loads the SKILL.md content and builds a user-turn message
3. The message includes the full skill content, a list of supporting files (not loaded), and the user's instruction
4. Supporting files can be loaded on demand via the `skill_view` tool
5. Injected as a **user message** (not system prompt) to preserve prompt caching

### Adding CLI Commands

1. Add to `COMMANDS` dict with description
2. Add handler in `process_command()` method
3. For persistent settings, use `save_config_value()` to update config

---

## Hermes CLI Commands

The unified `hermes` command provides all functionality:

| Command | Description |
|---------|-------------|
| `hermes` | Interactive chat (default) |
| `hermes chat -q "..."` | Single query mode |
| `hermes -c` / `hermes --continue` | Resume the most recent session |
| `hermes -c "my project"` | Resume a session by name (latest in lineage) |
| `hermes --resume <session_id>` | Resume a specific session by ID or title |
| `hermes -w` / `hermes --worktree` | Start in isolated git worktree (for parallel agents) |
| `hermes setup` | Configure API keys and settings |
| `hermes config` | View current configuration |
| `hermes config edit` | Open config in editor |
| `hermes config set KEY VAL` | Set a specific value |
| `hermes config check` | Check for missing config |
| `hermes config migrate` | Prompt for missing config interactively |
| `hermes status` | Show configuration status |
| `hermes doctor` | Diagnose issues |
| `hermes update` | Update to latest (checks for new config) |
| `hermes uninstall` | Uninstall (can keep configs for reinstall) |
| `hermes gateway` | Start gateway (messaging + cron scheduler) |
| `hermes gateway setup` | Configure messaging platforms interactively |
| `hermes gateway install` | Install gateway as system service |
| `hermes sessions list` | List past sessions (title, preview, last active) |
| `hermes sessions rename <id> <title>` | Rename/title a session |
| `hermes cron list` | View scheduled jobs |
| `hermes cron status` | Check if cron scheduler is running |
| `hermes version` | Show version info |
| `hermes pairing list/approve/revoke` | Manage DM pairing codes |

---

## Messaging Gateway

The gateway connects Hermes to Telegram, Discord, Slack, and WhatsApp.

### Setup

The interactive setup wizard handles platform configuration:

```bash
hermes gateway setup      # Arrow-key menu of all platforms, configure tokens/allowlists/home channels
```

This is the recommended way to configure messaging. It shows which platforms are already set up, walks through each one interactively, and offers to start/restart the gateway service at the end.

Platforms can also be configured manually in `~/.hermes/.env`:

### Configuration (in `~/.hermes/.env`):

```bash
# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...      # From @BotFather
TELEGRAM_ALLOWED_USERS=123456789,987654   # Comma-separated user IDs (from @userinfobot)

# Discord  
DISCORD_BOT_TOKEN=MTIz...                 # From Developer Portal
DISCORD_ALLOWED_USERS=123456789012345678  # Comma-separated user IDs

# Agent Behavior
HERMES_MAX_ITERATIONS=60                  # Max tool-calling iterations
MESSAGING_CWD=/home/myuser                # Terminal working directory for messaging

# Tool progress is configured in config.yaml (display.tool_progress: off|new|all|verbose)
```

### Working Directory Behavior

- **CLI (`hermes` command)**: Uses current directory (`.` → `os.getcwd()`)
- **Messaging (Telegram/Discord)**: Uses `MESSAGING_CWD` (default: home directory)

This is intentional: CLI users are in a terminal and expect the agent to work in their current directory, while messaging users need a consistent starting location.

### Security (User Allowlists):

**IMPORTANT**: By default, the gateway denies all users who are not in an allowlist or paired via DM.

The gateway checks `{PLATFORM}_ALLOWED_USERS` environment variables:
- If set: Only listed user IDs can interact with the bot
- If unset: All users are denied unless `GATEWAY_ALLOW_ALL_USERS=true` is set

Users can find their IDs:
- **Telegram**: Message [@userinfobot](https://t.me/userinfobot)
- **Discord**: Enable Developer Mode, right-click name → Copy ID

### DM Pairing System

Instead of static allowlists, users can pair via one-time codes:
1. Unknown user DMs the bot → receives pairing code
2. Owner runs `hermes pairing approve <platform> <code>`
3. User is permanently authorized

Security: 8-char codes, 1-hour expiry, rate-limited (1/10min/user), max 3 pending per platform, lockout after 5 failed attempts, `chmod 0600` on data files.

Files: `gateway/pairing.py`, `hermes_cli/pairing.py`

### Event Hooks

Hooks fire at lifecycle points. Place hook directories in `~/.hermes/hooks/`:

```
~/.hermes/hooks/my-hook/
├── HOOK.yaml    # name, description, events list
└── handler.py   # async def handle(event_type, context): ...
```

Events: `gateway:startup`, `session:start`, `session:reset`, `agent:start`, `agent:step`, `agent:end`, `command:*`

The `agent:step` event fires each iteration of the tool-calling loop with tool names and results.

Files: `gateway/hooks.py`

### Tool Progress Notifications

When `tool_progress` is enabled in `config.yaml`, the bot sends status messages as it works:
- `💻 \`ls -la\`...` (terminal commands show the actual command)
- `🔍 web_search...`
- `📄 web_extract...`
- `🐍 execute_code...` (programmatic tool calling sandbox)
- `🔀 delegate_task...` (subagent delegation)
- `❓ clarify...` (user question, CLI-only)

Modes:
- `new`: Only when switching to a different tool (less spam)
- `all`: Every single tool call

### Typing Indicator

The gateway keeps the "typing..." indicator active throughout processing, refreshing every 4 seconds. This lets users know the bot is working even during long tool-calling sequences.

### Platform Toolsets:

Each platform has a dedicated toolset in `toolsets.py`:
- `hermes-telegram`: Full tools including terminal (with safety checks)
- `hermes-discord`: Full tools including terminal
- `hermes-whatsapp`: Full tools including terminal

---

## Configuration System

Configuration files are stored in `~/.hermes/` for easy user access:
- `~/.hermes/config.yaml` - All settings (model, terminal, compression, etc.)
- `~/.hermes/.env` - API keys and secrets

### Adding New Configuration Options

When adding new configuration variables, you MUST follow this process:

#### For config.yaml options:

1. Add to `DEFAULT_CONFIG` in `hermes_cli/config.py`
2. **CRITICAL**: Bump `_config_version` in `DEFAULT_CONFIG` when adding required fields
3. This triggers migration prompts for existing users on next `hermes update` or `hermes setup`

Example:
```python
DEFAULT_CONFIG = {
    # ... existing config ...
    
    "new_feature": {
        "enabled": True,
        "option": "default_value",
    },
    
    # BUMP THIS when adding required fields
    "_config_version": 2,  # Was 1, now 2
}
```

#### For .env variables (API keys/secrets):

1. Add to `REQUIRED_ENV_VARS` or `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`
2. Include metadata for the migration system:

```python
OPTIONAL_ENV_VARS = {
    # ... existing vars ...
    "NEW_API_KEY": {
        "description": "What this key is for",
        "prompt": "Display name in prompts",
        "url": "https://where-to-get-it.com/",
        "tools": ["tools_it_enables"],  # What tools need this
        "password": True,  # Mask input
    },
}
```

#### Update related files:

- `hermes_cli/setup.py` - Add prompts in the setup wizard
- `cli-config.yaml.example` - Add example with comments
- Update README.md if user-facing

### Config Version Migration

The system uses `_config_version` to detect outdated configs:

1. `check_for_missing_config()` compares user config to `DEFAULT_CONFIG`
2. `migrate_config()` interactively prompts for missing values
3. Called automatically by `hermes update` and optionally by `hermes setup`

---

## Environment Variables

API keys are loaded from `~/.hermes/.env`:
- `OPENROUTER_API_KEY` - Main LLM API access (primary provider)
- `FIRECRAWL_API_KEY` - Web search/extract tools
- `FIRECRAWL_API_URL` - Self-hosted Firecrawl endpoint (optional)
- `BROWSERBASE_API_KEY` / `BROWSERBASE_PROJECT_ID` - Browser automation
- `FAL_KEY` - Image generation (FLUX model)
- `NOUS_API_KEY` - Vision and Mixture-of-Agents tools

Terminal tool configuration (in `~/.hermes/config.yaml`):
- `terminal.backend` - Backend: local, docker, singularity, modal, daytona, or ssh
- `terminal.cwd` - Working directory ("." = host CWD for local only; for remote backends set an absolute path inside the target, or omit to use the backend's default)
- `terminal.docker_image` - Image for Docker backend
- `terminal.singularity_image` - Image for Singularity backend
- `terminal.modal_image` - Image for Modal backend
- `terminal.daytona_image` - Image for Daytona backend
- `DAYTONA_API_KEY` - API key for Daytona backend (in .env)
- SSH: `TERMINAL_SSH_HOST`, `TERMINAL_SSH_USER`, `TERMINAL_SSH_KEY` in .env

Agent behavior (in `~/.hermes/.env`):
- `HERMES_MAX_ITERATIONS` - Max tool-calling iterations (default: 60)
- `MESSAGING_CWD` - Working directory for messaging platforms (default: ~)
- `display.tool_progress` in config.yaml - Tool progress: `off`, `new`, `all`, `verbose`
- `OPENAI_API_KEY` - Voice transcription (Whisper STT)
- `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` - Slack integration (Socket Mode)
- `SLACK_ALLOWED_USERS` - Comma-separated Slack user IDs
- `HERMES_HUMAN_DELAY_MODE` - Response pacing: off/natural/custom
- `HERMES_HUMAN_DELAY_MIN_MS` / `HERMES_HUMAN_DELAY_MAX_MS` - Custom delay range

### Dangerous Command Approval

The terminal tool includes safety checks for potentially destructive commands (e.g., `rm -rf`, `DROP TABLE`, `chmod 777`, etc.):

**Behavior by Backend:**
- **Docker/Singularity/Modal**: Commands run unrestricted (isolated containers)
- **Local/SSH**: Dangerous commands trigger approval flow

**Approval Flow (CLI):**
```
⚠️  Potentially dangerous command detected: recursive delete
    rm -rf /tmp/test

    [o]nce  |  [s]ession  |  [a]lways  |  [d]eny
    Choice [o/s/a/D]: 
```

**Approval Flow (Messaging):**
- Command is blocked with explanation
- Agent explains the command was blocked for safety
- User must add the pattern to their allowlist via `hermes config edit` or run the command directly on their machine

**Configuration:**
- `command_allowlist` in `~/.hermes/config.yaml` stores permanently allowed patterns
- Add patterns via "always" approval or edit directly

**Sudo Handling (Messaging):**
- If sudo fails over messaging, output includes tip to add `SUDO_PASSWORD` to `~/.hermes/.env`

---

## Background Process Management

The `process` tool works alongside `terminal` for managing long-running background processes:

**Starting a background process:**
```python
terminal(command="pytest -v tests/", background=true)
# Returns: {"session_id": "proc_abc123", "pid": 12345, ...}
```

**Managing it with the process tool:**
- `process(action="list")` -- show all running/recent processes
- `process(action="poll", session_id="proc_abc123")` -- check status + new output
- `process(action="log", session_id="proc_abc123")` -- full output with pagination
- `process(action="wait", session_id="proc_abc123", timeout=600)` -- block until done
- `process(action="kill", session_id="proc_abc123")` -- terminate
- `process(action="write", session_id="proc_abc123", data="y")` -- send stdin
- `process(action="submit", session_id="proc_abc123", data="yes")` -- send + Enter

**Key behaviors:**
- Background processes execute through the configured terminal backend (local/Docker/Modal/Daytona/SSH/Singularity) -- never directly on the host unless `TERMINAL_ENV=local`
- The `wait` action blocks the tool call until the process finishes, times out, or is interrupted by a new user message
- PTY mode (`pty=true` on terminal) enables interactive CLI tools (Codex, Claude Code)
- In RL training, background processes are auto-killed when the episode ends (`tool_context.cleanup()`)
- In the gateway, sessions with active background processes are exempt from idle reset
- The process registry checkpoints to `~/.hermes/processes.json` for crash recovery

Files: `tools/process_registry.py` (registry + handler), `tools/terminal_tool.py` (spawn integration)

---

## Adding New Tools

Adding a tool requires changes in **2 files** (the tool file and `toolsets.py`):

1. **Create `tools/your_tool.py`** with handler, schema, check function, and registry call:

```python
# tools/example_tool.py
import json
import os
from tools.registry import registry

def check_example_requirements() -> bool:
    """Check if required API keys/dependencies are available."""
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    """Execute the tool and return JSON string result."""
    try:
        result = {"success": True, "data": "..."}
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

EXAMPLE_SCHEMA = {
    "name": "example_tool",
    "description": "Does something useful.",
    "parameters": {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "The parameter"}
        },
        "required": ["param"]
    }
}

registry.register(
    name="example_tool",
    toolset="example",
    schema=EXAMPLE_SCHEMA,
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_example_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

2. **Add to `toolsets.py`**: Add `"example_tool"` to `_HERMES_CORE_TOOLS` if it should be in all platform toolsets, or create a new toolset entry.

3. **Add discovery import** in `model_tools.py`'s `_discover_tools()` list: `"tools.example_tool"`.

That's it. The registry handles schema collection, dispatch, availability checking, and error wrapping automatically. No edits to `TOOLSET_REQUIREMENTS`, `handle_function_call()`, `get_all_tool_names()`, or any other data structure.

**Optional:** Add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py` for the setup wizard, and to `toolset_distributions.py` for batch processing.

**Special case: tools that need agent-level state** (like `todo`, `memory`):
These are intercepted by `run_agent.py`'s tool dispatch loop *before* `handle_function_call()`. The registry still holds their schemas, but dispatch returns a stub error as a safety fallback. See `todo_tool.py` for the pattern.

All tool handlers MUST return a JSON string. The registry's `dispatch()` wraps all exceptions in `{"error": "..."}` automatically.

### Dynamic Tool Availability

Tools declare their requirements at registration time via `check_fn` and `requires_env`. The registry checks `check_fn()` when building tool definitions -- tools whose check fails are silently excluded.

### Stateful Tools

Tools that maintain state (terminal, browser) require:
- `task_id` parameter for session isolation between concurrent tasks
- `cleanup_*()` function to release resources
- Cleanup is called automatically in run_agent.py after conversation completes

---

## Trajectory Format

Conversations are saved in ShareGPT format for training:
```json
{"from": "system", "value": "System prompt with <tools>...</tools>"}
{"from": "human", "value": "User message"}
{"from": "gpt", "value": "<think>reasoning</think>\n<tool_call>{...}</tool_call>"}
{"from": "tool", "value": "<tool_response>{...}</tool_response>"}
{"from": "gpt", "value": "Final response"}
```

Tool calls use `<tool_call>` XML tags, responses use `<tool_response>` tags, reasoning uses `<think>` tags.

### Trajectory Export

```python
agent = AIAgent(save_trajectories=True)
agent.chat("Do something")
# Saves to trajectories/*.jsonl in ShareGPT format
```

---

## Batch Processing (batch_runner.py)

For processing multiple prompts:
- Parallel execution with multiprocessing
- Content-based resume for fault tolerance (matches on prompt text, not indices)
- Toolset distributions control probabilistic tool availability per prompt
- Output: `data/<run_name>/trajectories.jsonl` (combined) + individual batch files

```bash
python batch_runner.py \
    --dataset_file=prompts.jsonl \
    --batch_size=20 \
    --num_workers=4 \
    --run_name=my_run
```

---

## Skills System

Skills are on-demand knowledge documents the agent can load. Compatible with the [agentskills.io](https://agentskills.io/specification) open standard.

```
skills/
├── mlops/                    # Category folder
│   ├── axolotl/             # Skill folder
│   │   ├── SKILL.md         # Main instructions (required)
│   │   ├── references/      # Additional docs, API specs
│   │   ├── templates/       # Output formats, configs
│   │   └── assets/          # Supplementary files (agentskills.io)
│   └── vllm/
│       └── SKILL.md
├── .hub/                    # Skills Hub state (gitignored)
│   ├── lock.json            # Installed skill provenance
│   ├── quarantine/          # Pending security review
│   ├── audit.log            # Security scan history
│   ├── taps.json            # Custom source repos
│   └── index-cache/         # Cached remote indexes
```

**Progressive disclosure** (token-efficient):
1. `skills_categories()` - List category names (~50 tokens)
2. `skills_list(category)` - Name + description per skill (~3k tokens)
3. `skill_view(name)` - Full content + tags + linked files

SKILL.md files use YAML frontmatter (agentskills.io format):
```yaml
---
name: skill-name
description: Brief description for listing
version: 1.0.0
platforms: [macos]              # Optional — restrict to specific OS (macos/linux/windows)
metadata:
  hermes:
    tags: [tag1, tag2]
    related_skills: [other-skill]
---
# Skill Content...
```

**Platform filtering** — Skills with a `platforms` field are automatically excluded from the system prompt index, `skills_list()`, and slash commands on incompatible platforms. Skills without the field load everywhere (backward compatible). See `skills/apple/` for macOS-only examples (iMessage, Reminders, Notes, FindMy).

**Skills Hub** — user-driven skill search/install from online registries and official optional skills. Sources: official optional skills (shipped with repo, labeled "official"), GitHub (openai/skills, anthropics/skills, custom taps), ClawHub, Claude marketplace, LobeHub. Not exposed as an agent tool — the model cannot search for or install skills. Users manage skills via `hermes skills browse/search/install` CLI commands or the `/skills` slash command in chat.

Key files:
- `tools/skills_tool.py` — Agent-facing skill list/view (progressive disclosure)
- `tools/skills_guard.py` — Security scanner (regex + LLM audit, trust-aware install policy)
- `tools/skills_hub.py` — Source adapters (OptionalSkillSource, GitHub, ClawHub, Claude marketplace, LobeHub), lock file, auth
- `hermes_cli/skills_hub.py` — CLI subcommands + `/skills` slash command handler

---

## Auxiliary Model Configuration

Hermes uses lightweight "auxiliary" models for side tasks that run alongside the main conversation model:

| Task | Tool(s) | Default Model |
|------|---------|---------------|
| **Vision analysis** | `vision_analyze`, `browser_vision` | `google/gemini-3-flash-preview` (via OpenRouter) |
| **Web extraction** | `web_extract`, browser snapshot summarization | `google/gemini-3-flash-preview` (via OpenRouter) |
| **Context compression** | Auto-compression when approaching context limit | `google/gemini-3-flash-preview` (via OpenRouter) |

By default, these auto-detect the best available provider: OpenRouter → Nous Portal → (text tasks only) custom endpoint → Codex → API-key providers.

### Changing the Vision Model

To use a different model for image analysis (e.g., GPT-4o instead of Gemini Flash), add to `~/.hermes/config.yaml`:

```yaml
auxiliary:
  vision:
    provider: "openrouter"        # or "nous", "main", "auto"
    model: "openai/gpt-4o"        # any model slug your provider supports
```

Or set environment variables (in `~/.hermes/.env` or shell):

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
# Optionally force a specific provider:
AUXILIARY_VISION_PROVIDER=openrouter
```

### Changing the Web Extraction Model

```yaml
auxiliary:
  web_extract:
    provider: "auto"
    model: "google/gemini-2.5-flash"
```

### Changing the Compression Model

```yaml
compression:
  summary_model: "google/gemini-2.5-flash"
  summary_provider: "auto"          # "auto", "openrouter", "nous", "main"
```

### Provider Options

| Provider | Description |
|----------|-------------|
| `"auto"` | Best available (default). For vision, only tries OpenRouter + Nous. |
| `"openrouter"` | Force OpenRouter (requires `OPENROUTER_API_KEY`) |
| `"nous"` | Force Nous Portal (requires `hermes login`) |
| `"codex"` | Force Codex OAuth (ChatGPT account). Supports vision via gpt-5.3-codex. |
| `"main"` | Use your custom endpoint (`OPENAI_BASE_URL` + `OPENAI_API_KEY`). Works with OpenAI API, local models, etc. |

**Important:** Vision tasks require a multimodal-capable model. In `auto` mode, OpenRouter, Nous Portal, and Codex OAuth are tried (they all support vision). Setting `provider: "main"` for vision will work only if your endpoint supports multimodal input (e.g. OpenAI with GPT-4o, or a local model with vision).

**Key files:** `agent/auxiliary_client.py` (resolution chain), `tools/vision_tools.py`, `tools/browser_tool.py`, `tools/web_tools.py`

---

## Known Pitfalls

### DO NOT use `simple_term_menu` for interactive menus

`simple_term_menu` has rendering bugs in tmux, iTerm2, and other non-standard terminals. When the user scrolls with arrow keys, previously highlighted items "ghost" — duplicating upward and corrupting the display. This happens because the library uses ANSI cursor-up codes to redraw in place, and tmux/iTerm miscalculate positions when the menu is near the bottom of the viewport.

**Rule:** All interactive menus in `hermes_cli/` must use `curses` (Python stdlib) instead. See `tools_config.py` for the pattern — both `_prompt_choice()` (single-select) and `_prompt_toolset_checklist()` (multi-select with space toggle) use `curses.wrapper()`. The numbered-input fallback handles Windows where curses isn't available.

### DO NOT use `\033[K` (ANSI erase-to-EOL) in spinner/display code

The ANSI escape `\033[K` leaks as literal `?[K` text when `prompt_toolkit`'s `patch_stdout` is active. Use space-padding instead to clear lines: `f"\r{line}{' ' * pad}"`. See `agent/display.py` `KawaiiSpinner`.

### `_last_resolved_tool_names` is a process-global in `model_tools.py`

The `execute_code` sandbox uses `_last_resolved_tool_names` (set by `get_tool_definitions()`) to decide which tool stubs to generate. When subagents run with restricted toolsets, they overwrite this global. After delegation returns to the parent, `execute_code` may see the child's restricted list instead of the parent's full list. This is a known bug — `execute_code` calls after delegation may fail with `ImportError: cannot import name 'patch' from 'hermes_tools'`.

### Tests must not write to `~/.hermes/`

The `autouse` fixture `_isolate_hermes_home` in `tests/conftest.py` redirects `HERMES_HOME` to a temp dir. Every test runs in isolation. If you add a test that creates `AIAgent` instances or writes session logs, the fixture handles cleanup automatically. Never hardcode `~/.hermes/` paths in tests.

---

## Testing Changes

After making changes:

1. Run `hermes doctor` to check setup
2. Run `hermes config check` to verify config
3. Test with `hermes chat -q "test message"`
4. For new config options, test fresh install: `rm -rf ~/.hermes && hermes setup`
