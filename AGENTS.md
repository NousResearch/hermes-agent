# Hermes Agent - Development Guide

Instructions for AI coding assistants (GitHub Copilot, Cursor, etc.) and human developers.

Hermes Agent is an AI agent harness with tool-calling capabilities, interactive CLI, messaging integrations, and scheduled tasks.

## Development Environment

**IMPORTANT**: Always use the virtual environment if it exists:
```bash
source .venv/bin/activate  # Before running any Python commands
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
│   ├── trajectory.py         # Trajectory saving helpers
│   ├── skill_commands.py     # Skill slash command scanning + invocation (shared CLI/gateway)
│   ├── auxiliary_client.py   # Auxiliary LLM client (vision, summarization)
│   ├── insights.py           # Usage analytics and session statistics
│   └── redact.py             # Sensitive data redaction
├── hermes_cli/           # CLI implementation
│   ├── main.py           # Entry point, command dispatcher (all `hermes` subcommands)
│   ├── banner.py         # Welcome banner, ASCII art, skills summary
│   ├── commands.py       # Slash command definitions + SlashCommandCompleter
│   ├── callbacks.py      # Interactive prompt callbacks (clarify, sudo, approval)
│   ├── setup.py          # Interactive setup wizard
│   ├── config.py         # Config management, DEFAULT_CONFIG, migration
│   ├── status.py         # Status display
│   ├── doctor.py         # Diagnostics
│   ├── gateway.py        # Gateway management (start/stop/install)
│   ├── uninstall.py      # Uninstaller
│   ├── cron.py           # Cron job management
│   ├── skills_hub.py     # Skills Hub CLI + /skills slash command
│   ├── tools_config.py   # `hermes tools` command — per-platform tool toggling
│   ├── pairing.py        # DM pairing management CLI
│   ├── auth.py           # Provider OAuth authentication
│   ├── models.py         # Model selection and listing
│   ├── runtime_provider.py # Runtime provider resolution
│   ├── clipboard.py      # Clipboard image paste support
│   ├── colors.py         # Terminal color utilities
│   └── codex_models.py   # Codex/Responses API model definitions
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
│   ├── process_registry.py    # Background process management
│   ├── todo_tool.py           # Planning & task management
│   ├── memory_tool.py         # Persistent memory read/write
│   ├── skills_tool.py         # Agent-facing skill list/view (progressive disclosure)
│   ├── skill_manager_tool.py  # Skill CRUD operations
│   ├── session_search_tool.py # FTS5 session search
│   ├── file_tools.py          # File read/write/search/patch tools
│   ├── file_operations.py     # File operations helpers
│   ├── web_tools.py           # Firecrawl search/extract
│   ├── browser_tool.py        # Browserbase browser automation
│   ├── vision_tools.py        # Image analysis via auxiliary LLM
│   ├── image_generation_tool.py # FLUX image generation via fal.ai
│   ├── tts_tool.py            # Text-to-speech
│   ├── transcription_tools.py # Whisper voice transcription
│   ├── code_execution_tool.py # execute_code sandbox
│   ├── delegate_tool.py       # Subagent delegation
│   ├── clarify_tool.py        # User clarification prompts
│   ├── send_message_tool.py   # Cross-platform message sending
│   ├── cronjob_tools.py       # Scheduled task management
│   ├── mcp_tool.py            # MCP (Model Context Protocol) client
│   ├── mixture_of_agents_tool.py # Mixture-of-Agents orchestration
│   ├── homeassistant_tool.py  # Home Assistant integration
│   ├── honcho_tools.py        # Honcho context management
│   ├── rl_training_tool.py    # RL training environment tools
│   ├── openrouter_client.py   # OpenRouter API helpers
│   ├── patch_parser.py        # V4A patch format parser
│   ├── fuzzy_match.py         # Multi-strategy fuzzy string matching
│   ├── interrupt.py           # Agent interrupt handling
│   ├── debug_helpers.py       # Debug/diagnostic helpers
│   ├── skills_guard.py        # Security scanner (regex + LLM audit)
│   ├── skills_hub.py          # Source adapters for skills marketplace
│   └── skills_sync.py         # Skill synchronization
├── gateway/              # Messaging platform adapters
│   ├── run.py            # Main gateway loop, slash commands, message dispatch
│   ├── session.py        # SessionStore — conversation persistence
│   ├── config.py         # Gateway-specific config helpers
│   ├── delivery.py       # Message delivery (origin, telegram, discord, etc.)
│   ├── hooks.py          # Event hook system
│   ├── pairing.py        # DM pairing system (code generation, verification)
│   ├── mirror.py         # Message mirroring
│   ├── status.py         # Gateway status reporting
│   ├── sticker_cache.py  # Telegram sticker description cache
│   ├── channel_directory.py # Channel/chat directory management
│   └── platforms/        # Platform-specific adapters
│       ├── base.py           # BasePlatform ABC
│       ├── telegram.py       # Telegram bot adapter
│       ├── discord.py        # Discord bot adapter
│       ├── slack.py          # Slack bot adapter (Socket Mode)
│       ├── whatsapp.py       # WhatsApp adapter
│       └── homeassistant.py  # Home Assistant adapter
├── cron/                 # Scheduler implementation
├── environments/         # RL training environments (Atropos integration)
├── honcho_integration/   # Honcho client & session management
├── skills/               # Bundled skill sources
├── optional-skills/      # Official optional skills (not activated by default)
├── scripts/              # Install scripts, utilities
├── tests/                # Full pytest suite (~2300+ tests)
├── cli.py                # Interactive CLI orchestrator (HermesCLI class)
├── hermes_state.py       # SessionDB — SQLite session store (schema, titles, FTS5 search)
├── hermes_constants.py   # OpenRouter URL constants
├── hermes_time.py        # Timezone-aware timestamp utilities
├── run_agent.py          # AIAgent class (core conversation loop)
├── model_tools.py        # Tool orchestration (thin layer over tools/registry.py)
├── toolsets.py           # Tool groupings and platform toolset definitions
├── toolset_distributions.py  # Probability-based tool selection
├── trajectory_compressor.py  # Trajectory post-processing
├── utils.py              # Shared utilities
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
        base_url: str = None,
        api_key: str = None,
        provider: str = None,             # Provider identifier (routing hints)
        api_mode: str = None,             # "chat_completions" or "codex_responses"
        model: str = "anthropic/claude-opus-4.6",  # OpenRouter format
        max_iterations: int = 90,         # Max tool-calling loops
        tool_delay: float = 1.0,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,         # Suppress progress output
        session_id: str = None,
        tool_progress_callback: callable = None,  # Called on each tool use
        clarify_callback: callable = None,
        step_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: dict = None,
        platform: str = None,             # Platform identifier (cli, telegram, etc.)
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db = None,
        iteration_budget: "IterationBudget" = None,
        # ... plus OpenRouter provider routing params
    ):
        # Initialize OpenAI client, load tools based on toolsets
        ...
    
    def chat(self, message: str) -> str:
        # Simple interface — returns just the final response string
        ...
    
    def run_conversation(
        self, user_message: str, system_message: str = None,
        conversation_history: list = None, task_id: str = None
    ) -> dict:
        # Full interface — returns dict with final_response + message history
        ...
```

### Agent Loop

The core loop is inside `run_conversation()` (there is no separate `_run_agent_loop()` method):

```
1. Add user message to conversation
2. Call LLM with tools
3. If LLM returns tool calls:
   - Execute each tool (synchronously)
   - Add tool results to conversation
   - Go to step 2
4. If LLM returns text response:
   - Return response to user
```

```python
while api_call_count < self.max_iterations and self.iteration_budget.remaining > 0:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,
    )
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

Note: The agent is **entirely synchronous** — no async/await anywhere.

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
- **KawaiiSpinner** (in agent/display.py) - Animated kawaii faces during API calls; clean `┊` activity feed for tool execution results

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

1. Add to `COMMANDS` dict in `hermes_cli/commands.py`
2. Add handler in `process_command()` method (in `HermesCLI` class, `cli.py`)
3. For persistent settings, use `save_config_value()` to update config

---

## Hermes CLI Commands

The unified `hermes` command provides all functionality:

| Command | Description |
|---------|-------------|
| `hermes` | Interactive chat (default) |
| `hermes chat -q "..."` | Single query mode |
| `hermes chat -m <model>` | Chat with a specific model |
| `hermes chat --provider <name>` | Chat with a specific provider |
| `hermes -c` / `hermes --continue` | Resume the most recent session |
| `hermes -c "my project"` | Resume a session by name (latest in lineage) |
| `hermes --resume <session_id>` | Resume a specific session by ID or title |
| `hermes -w` / `hermes --worktree` | Start in isolated git worktree (for parallel agents) |
| `hermes model` | Interactive provider and model selection |
| `hermes login <provider>` | OAuth login to inference providers (nous, openai-codex) |
| `hermes logout <provider>` | Clear authentication credentials |
| `hermes setup` | Configure API keys and settings |
| `hermes config` / `hermes config show` | View current configuration |
| `hermes config edit` | Open config in editor |
| `hermes config set KEY VAL` | Set a specific value |
| `hermes config check` | Check for missing config |
| `hermes config migrate` | Prompt for missing config interactively |
| `hermes config path` | Show config file path |
| `hermes config env-path` | Show .env file path |
| `hermes status` | Show configuration status |
| `hermes doctor` | Diagnose issues |
| `hermes update` | Update to latest (checks for new config) |
| `hermes uninstall` | Uninstall (can keep configs for reinstall) |
| `hermes gateway` | Start gateway (messaging + cron scheduler) |
| `hermes gateway setup` | Configure messaging platforms interactively |
| `hermes gateway install` | Install gateway as system service |
| `hermes gateway start/stop/restart` | Manage gateway service |
| `hermes gateway status` | Check gateway service status |
| `hermes gateway uninstall` | Remove gateway service |
| `hermes whatsapp` | WhatsApp setup and QR pairing wizard |
| `hermes tools` | Interactive tool configuration per platform |
| `hermes skills browse/search` | Browse and search skills marketplace |
| `hermes skills install/uninstall` | Install or remove skills |
| `hermes skills list` | List installed skills |
| `hermes skills audit` | Security audit installed skills |
| `hermes skills tap add/remove/list` | Manage custom skill sources |
| `hermes sessions list` | List past sessions (title, preview, last active) |
| `hermes sessions rename <id> <title>` | Rename/title a session |
| `hermes sessions export <id>` | Export a session |
| `hermes sessions delete <id>` | Delete a session |
| `hermes sessions prune` | Remove old sessions |
| `hermes sessions stats` | Session statistics |
| `hermes cron list` | View scheduled jobs |
| `hermes cron status` | Check if cron scheduler is running |
| `hermes insights` | Usage analytics and session statistics |
| `hermes version` | Show version info |
| `hermes pairing list/approve/revoke` | Manage DM pairing codes |

---

## Messaging Gateway

The gateway connects Hermes to Telegram, Discord, Slack, WhatsApp, Signal, and Home Assistant.

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

# Signal
SIGNAL_HTTP_URL=http://127.0.0.1:8080    # signal-cli daemon URL
SIGNAL_ACCOUNT=+1234567890               # Bot phone number (E.164)
SIGNAL_ALLOWED_USERS=+1234567890         # Comma-separated E.164 numbers/UUIDs

# Agent Behavior
HERMES_MAX_ITERATIONS=90                  # Max tool-calling iterations (default: 90)
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

### Gateway Slash Commands

The gateway supports these slash commands in messaging chats:
- `/new` - Start a new conversation
- `/reset` - Reset conversation history
- `/retry` - Retry last message
- `/undo` - Remove the last exchange
- `/compress` - Compress conversation context
- `/stop` - Interrupt the running agent
- `/model` - Show/change model
- `/provider` - Show available providers and auth status
- `/personality` - Set a personality
- `/title` - Set or show session title
- `/resume` - Resume a previously-named session
- `/usage` - Show token usage for this session
- `/insights` - Show usage analytics
- `/sethome` - Set this chat as the home channel
- `/reload-mcp` - Reload MCP servers from config
- `/update` - Update Hermes Agent to latest version
- `/help` - Show command list
- `/status` - Show session info
- Plus dynamic `/skill-name` commands (loaded from agent/skill_commands.py)

### Typing Indicator

The gateway keeps the "typing..." indicator active throughout processing, refreshing every 4 seconds. This lets users know the bot is working even during long tool-calling sequences.

### Platform Toolsets:

Each platform has a dedicated toolset in `toolsets.py` (all share the same `_HERMES_CORE_TOOLS` list):
- `hermes-cli`: CLI-specific toolset
- `hermes-telegram`: Full tools including terminal (with safety checks)
- `hermes-discord`: Full tools including terminal
- `hermes-whatsapp`: Full tools including terminal
- `hermes-slack`: Full tools including terminal
- `hermes-homeassistant`: Home Assistant integration tools
- `hermes-gateway`: Meta-toolset including all platform toolsets

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

1. Add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py` (note: `REQUIRED_ENV_VARS` exists but is intentionally empty — provider setup is handled by the setup wizard)
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
        "category": "tool",  # One of: provider, tool, messaging, setting
    },
}
```

#### Update related files:

- `hermes_cli/setup.py` - Add prompts in the setup wizard
- `cli-config.yaml.example` - Add example with comments
- Update README.md if user-facing

### Config Version Migration

The system uses `_config_version` (currently at version 5) to detect outdated configs:

1. `check_config_version()` compares user config version to `DEFAULT_CONFIG` version
2. `get_missing_env_vars()` identifies missing environment variables
3. `migrate_config()` interactively prompts for missing values and handles version-specific migrations (e.g., v3→4: tool progress, v4→5: timezone)
4. Called automatically by `hermes update` and optionally by `hermes setup`

---

## Environment Variables

API keys are loaded from `~/.hermes/.env`:
- `OPENROUTER_API_KEY` - Main LLM API access (primary provider)
- `FIRECRAWL_API_KEY` - Web search/extract tools
- `FIRECRAWL_API_URL` - Self-hosted Firecrawl endpoint (optional)
- `BROWSERBASE_API_KEY` / `BROWSERBASE_PROJECT_ID` - Browser automation
- `FAL_KEY` - Image generation (FLUX model)
- `VOICE_TOOLS_OPENAI_KEY` - Voice transcription (Whisper STT) and OpenAI TTS

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
- `HERMES_MAX_ITERATIONS` - Max tool-calling iterations (default: 90)
- `MESSAGING_CWD` - Working directory for messaging platforms (default: ~)
- `display.tool_progress` in config.yaml - Tool progress: `off`, `new`, `all`, `verbose`
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

Adding a tool requires changes in **3 files** (the tool file, `model_tools.py`, and `toolsets.py`):

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

2. **Add discovery import** in `model_tools.py`'s `_discover_tools()` list: `"tools.example_tool"`.

3. **Add to `toolsets.py`**: Add `"example_tool"` to `_HERMES_CORE_TOOLS` if it should be in all platform toolsets, or create a new toolset entry.

That's it. The registry handles schema collection, dispatch, availability checking, and error wrapping automatically. No edits to `handle_function_call()`, `get_all_tool_names()`, or any other data structure.

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
# Saves to trajectory_samples.jsonl (or failed_trajectories.jsonl) in ShareGPT format
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
