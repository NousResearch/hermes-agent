# Hermes Agent - Development Guide

Instructions for AI coding assistants (GitHub Copilot, Cursor, etc.) and human developers.

Hermes-Agent is an AI agent harness with tool-calling capabilities, interactive CLI, messaging integrations, and scheduled tasks.

## Development Environment

**IMPORTANT**: Always use the virtual environment if it exists:
```bash
source venv/bin/activate  # Before running any Python commands
```

## Project Structure

```
hermes-agent/
├── hermes_cli/           # Unified CLI commands
│   ├── main.py           # Entry point, command dispatcher
│   ├── setup.py          # Interactive setup wizard
│   ├── config.py         # Config management & migration
│   ├── status.py         # Status display
│   ├── doctor.py         # Diagnostics
│   ├── gateway.py        # Gateway management
│   ├── uninstall.py      # Uninstaller
│   └── cron.py           # Cron job management
├── tools/                # Tool implementations
├── gateway/              # Messaging platform adapters
├── cron/                 # Scheduler implementation
├── skills/               # Knowledge documents
├── cli.py                # Interactive CLI (Rich UI)
├── run_agent.py          # Agent runner with AIAgent class
├── model_tools.py        # Tool schemas and handlers
├── toolsets.py           # Tool groupings
├── toolset_distributions.py  # Probability-based tool selection
└── batch_runner.py       # Parallel batch processing
```

**User Configuration** (stored in `~/.hermes/`):
- `~/.hermes/config.yaml` - Settings (model, terminal, toolsets, etc.)
- `~/.hermes/.env` - API keys and secrets

## File Dependency Chain

```
tools/*.py → tools/__init__.py → model_tools.py → toolsets.py → toolset_distributions.py
                                       ↑
run_agent.py ──────────────────────────┘
cli.py → run_agent.py (uses AIAgent with quiet_mode=True)
batch_runner.py → run_agent.py + toolset_distributions.py
```

Always ensure consistency between tools, model_tools.py, and toolsets.py when changing any of them.

---

## AIAgent Class

The main agent is implemented in `run_agent.py`:

```python
class AIAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_turns: int = 20,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        verbose_logging: bool = False,
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
- **prompt_toolkit** - For fixed input area with history and `patch_stdout`
- **KawaiiSpinner** (in run_agent.py) - Animated feedback during API calls and tool execution

Key components:
- `HermesCLI` class - Main CLI controller with commands and conversation loop
- `load_cli_config()` - Loads config, sets environment variables for terminal
- `build_welcome_banner()` - Displays ASCII art logo, tools, and skills summary
- `/commands` - Process user commands like `/help`, `/clear`, `/personality`, etc.

CLI uses `quiet_mode=True` when creating AIAgent to suppress verbose logging.

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
| `hermes gateway` | Start messaging gateway |
| `hermes cron list` | View scheduled jobs |
| `hermes version` | Show version info |

---

## Messaging Gateway

The gateway connects Hermes to Telegram, Discord, and WhatsApp.

### Configuration (in `~/.hermes/.env`):

```bash
# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...      # From @BotFather
TELEGRAM_ALLOWED_USERS=123456789,987654   # Comma-separated user IDs (from @userinfobot)

# Discord  
DISCORD_BOT_TOKEN=MTIz...                 # From Developer Portal
DISCORD_ALLOWED_USERS=123456789012345678  # Comma-separated user IDs
```

### Security (User Allowlists):

**IMPORTANT**: Without an allowlist, anyone who finds your bot can use it!

The gateway checks `{PLATFORM}_ALLOWED_USERS` environment variables:
- If set: Only listed user IDs can interact with the bot
- If unset: All users are allowed (dangerous with terminal access!)

Users can find their IDs:
- **Telegram**: Message [@userinfobot](https://t.me/userinfobot)
- **Discord**: Enable Developer Mode, right-click name → Copy ID

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
- `BROWSERBASE_API_KEY` / `BROWSERBASE_PROJECT_ID` - Browser automation
- `FAL_KEY` - Image generation (FLUX model)
- `NOUS_API_KEY` - Vision and Mixture-of-Agents tools

Terminal tool configuration (in `~/.hermes/config.yaml`):
- `terminal.backend` - Backend: local, docker, singularity, modal, or ssh
- `terminal.cwd` - Working directory ("." = current directory)
- `terminal.docker_image` - Image for Docker backend
- `terminal.singularity_image` - Image for Singularity backend
- `terminal.modal_image` - Image for Modal backend
- SSH: `TERMINAL_SSH_HOST`, `TERMINAL_SSH_USER`, `TERMINAL_SSH_KEY` in .env

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

## Adding New Tools

Follow this strict order to maintain consistency:

1. Create `tools/your_tool.py` with:
   - Handler function (sync or async) returning a JSON string via `json.dumps()`
   - `check_*_requirements()` function to verify dependencies (e.g., API keys)
   - Schema definition following OpenAI function-calling format

2. Export in `tools/__init__.py`:
   - Import the handler and check function
   - Add to `__all__` list

3. Register in `model_tools.py`:
   - Add to `TOOLSET_REQUIREMENTS` if it needs API keys
   - Create `get_*_tool_definitions()` function or add to existing
   - Add routing in `handle_function_call()` dispatcher
   - Update `get_all_tool_names()` with the tool name
   - Update `get_toolset_for_tool()` mapping
   - Update `get_available_toolsets()` and `check_toolset_requirements()`

4. Add to toolset in `toolsets.py`:
   - Add to existing toolset or create new one in TOOLSETS dict

5. If the tool requires an API key:
   - Add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`
   - The tool will be auto-disabled if the key is missing

6. Optionally add to `toolset_distributions.py` for batch processing

### Tool Implementation Pattern

```python
# tools/example_tool.py
import json
import os

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
```

All tool handlers MUST return a JSON string. Never return raw dicts.

### Dynamic Tool Availability

Tools are automatically disabled when their API keys are missing:

```python
# In model_tools.py
TOOLSET_REQUIREMENTS = {
    "web": {"env_vars": ["FIRECRAWL_API_KEY"]},
    "browser": {"env_vars": ["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"]},
    "creative": {"env_vars": ["FAL_KEY"]},
}
```

The `check_tool_availability()` function determines which tools to include.

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

Skills are on-demand knowledge documents the agent can load. Located in `skills/` directory:

```
skills/
├── mlops/                    # Category folder
│   ├── axolotl/             # Skill folder
│   │   ├── SKILL.md         # Main instructions (required)
│   │   ├── references/      # Additional docs, API specs
│   │   └── templates/       # Output formats, configs
│   └── vllm/
│       └── SKILL.md
└── example-skill/
    └── SKILL.md
```

**Progressive disclosure** (token-efficient):
1. `skills_categories()` - List category names (~50 tokens)
2. `skills_list(category)` - Name + description per skill (~3k tokens)
3. `skill_view(name)` - Full content + tags + linked files

SKILL.md files use YAML frontmatter:
```yaml
---
name: skill-name
description: Brief description for listing
tags: [tag1, tag2]
related_skills: [other-skill]
version: 1.0.0
---
# Skill Content...
```

Tool files: `tools/skills_tool.py` → `model_tools.py` → `toolsets.py`

---

## Testing Changes

After making changes:

1. Run `hermes doctor` to check setup
2. Run `hermes config check` to verify config
3. Test with `hermes chat -q "test message"`
4. For new config options, test fresh install: `rm -rf ~/.hermes && hermes setup`
