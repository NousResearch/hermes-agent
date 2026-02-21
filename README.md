<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ⚕

An AI agent with advanced tool-calling capabilities, featuring a flexible toolsets system, messaging integrations, and scheduled tasks.

## Quick Install

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

The installer will:
- Install [uv](https://docs.astral.sh/uv/) (fast Python package manager) if not present
- Install Python 3.11 via uv if not already available (no sudo needed)
- Clone to `~/.hermes/hermes-agent` (with submodules: mini-swe-agent, tinker-atropos)
- Create a virtual environment with Python 3.11
- Install all dependencies and submodule packages
- Symlink `hermes` into `~/.local/bin` so it works globally (no venv activation needed)
- Run the interactive setup wizard

After installation, reload your shell and run:
```bash
hermes setup    # Configure API keys (if you skipped during install)
hermes          # Start chatting!
```

---

## Getting Started

The installer (`hermes setup`) walks you through selecting a provider and model. Once that's done:

```bash
hermes          # Start chatting!
```

To change your provider or model later:

```bash
hermes model    # Interactive provider & model selector
```

This lets you switch between **Nous Portal** (subscription), **OpenRouter** (100+ models, pay-per-use), or a **custom endpoint** (VLLM, SGLang, any OpenAI-compatible API) at any time.

---

## Updating

**Quick update (installer version):**
```bash
hermes update    # Update to latest version (prompts for new config)
```

**Manual update (if you cloned the repo yourself):**
```bash
cd /path/to/hermes-agent
export VIRTUAL_ENV="$(pwd)/venv"

# Pull latest code and submodules
git pull origin main
git submodule update --init --recursive

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"
uv pip install -e "./mini-swe-agent"
uv pip install -e "./tinker-atropos"

# Check for new config options added since your last update
hermes config check
hermes config migrate   # Interactively add any missing options
```

**Uninstalling:**
```bash
hermes uninstall          # Uninstall (can keep configs for later reinstall)
```

Or manually:
```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # Optional — keep if you plan to reinstall
```

---

## Configuration

All your settings are stored in `~/.hermes/` for easy access:

```
~/.hermes/
├── config.yaml     # Settings (model, terminal, TTS, compression, etc.)
├── .env            # API keys and secrets
├── auth.json       # OAuth provider credentials (Nous Portal, etc.)
├── SOUL.md         # Optional: global persona (agent embodies this personality)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (managed via skill_manage tool)
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs
```

### Messaging Platforms (Telegram, Discord, Slack)

If you configured a messaging bot token during setup, **start the gateway** so Hermes can receive and send messages:

```bash
hermes gateway              # Run in foreground (see output)
hermes gateway install      # Or install as a background service (Linux)
hermes gateway start        # Start the background service
```

The installer will offer to do this automatically if it detects a bot token. See [Messaging Gateway](#messaging-gateway) below for full setup instructions.

### Managing Configuration

```bash
hermes config              # View current configuration
hermes config edit         # Open config.yaml in your editor
hermes config set KEY VAL  # Set a specific value
hermes config check        # Check for missing options (after updates)
hermes config migrate      # Interactively add missing options

# Examples:
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # Saves to .env
```

### Inference Providers

You need at least one way to connect to an LLM. Use `hermes model` to switch providers and models interactively, or configure directly:

| Provider | Setup |
|----------|-------|
| **Nous Portal** | `hermes login` (OAuth, subscription-based) |
| **OpenRouter** | `OPENROUTER_API_KEY` in `~/.hermes/.env` |
| **Custom Endpoint** | `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `~/.hermes/.env` |

**Note:** Even when using Nous Portal or a custom endpoint, some tools (vision, web summarization, MoA) use OpenRouter independently. An `OPENROUTER_API_KEY` enables these tools.

### Optional API Keys

| Feature | Provider | Env Variable |
|---------|----------|--------------|
| Custom OpenAI Endpoint (OAI or VLLM/SGLANG) | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Web scraping | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY` |
| Browser automation | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| Image generation | [FAL](https://fal.ai/) | `FAL_KEY` |
| Premium TTS voices | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS voices | [OpenAI](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| RL Training | [Tinker](https://tinker-console.thinkingmachines.ai/) + [WandB](https://wandb.ai/) | `TINKER_API_KEY`, `WANDB_API_KEY` |
| Voice transcription | [OpenAI](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Slack integration | [Slack](https://api.slack.com/apps) | `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` |
| Messaging | Telegram, Discord | `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN` |

---

## Messaging Gateway

Chat with Hermes from Telegram, Discord, or WhatsApp.

### Starting the Gateway

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as systemd service (Linux)
hermes gateway start        # Start the systemd service
hermes gateway stop         # Stop the systemd service
hermes gateway status       # Check service status
```

### Gateway Commands (inside chat)

| Command | Description |
|---------|-------------|
| `/new` or `/reset` | Start fresh conversation |
| `/status` | Show session info |
| `/hermes` (Discord) | Slash command — ask, reset, status, stop |

### Telegram Setup

1. **Create a bot:** Message [@BotFather](https://t.me/BotFather) on Telegram, use `/newbot`
2. **Get your user ID:** Message [@userinfobot](https://t.me/userinfobot) - it replies with your numeric ID
3. **Configure:**

```bash
# Add to ~/.hermes/.env:
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_USERS=YOUR_USER_ID    # Comma-separated for multiple users
```

4. **Start the gateway:** `hermes gateway`

### Discord Setup

1. **Create a bot:** Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. **Get your user ID:** Enable Developer Mode in Discord settings, right-click your name → Copy ID
3. **Configure:**

```bash
# Add to ~/.hermes/.env:
DISCORD_BOT_TOKEN=MTIz...
DISCORD_ALLOWED_USERS=YOUR_USER_ID
```

### Slack Setup

1. **Create an app:** Go to [Slack API](https://api.slack.com/apps), create a new app
2. **Enable Socket Mode:** In app settings → Socket Mode → Enable
3. **Get tokens:**
   - Bot Token (`xoxb-...`): OAuth & Permissions → Install to Workspace
   - App Token (`xapp-...`): Basic Information → App-Level Tokens → Generate
4. **Configure:**

```bash
# Add to ~/.hermes/.env:
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_ALLOWED_USERS=U01234ABCDE    # Comma-separated Slack user IDs
```

5. **Start the gateway:** `hermes gateway`

### DM Pairing (Alternative to Allowlists)

Instead of manually configuring user IDs in allowlists, you can use the pairing system. When an unknown user DMs your bot, they receive a one-time pairing code:

```bash
# The user sees: "Pairing code: XKGH5N7P"
# You approve them with:
hermes pairing approve telegram XKGH5N7P

# Other pairing commands:
hermes pairing list          # View pending + approved users
hermes pairing revoke telegram 123456789  # Remove access
```

Pairing codes expire after 1 hour, are rate-limited, and use cryptographic randomness.

### Security (Important!)

**Without an allowlist, anyone who finds your bot can use it!**

```bash
# Restrict to specific users (recommended):
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678

# Or allow all users in a specific platform:
# (Leave the variable unset - NOT recommended for bots with terminal access)
```

### Working Directory

- **CLI (`hermes`)**: Uses current directory where you run the command
- **Messaging**: Uses `MESSAGING_CWD` (default: home directory `~`)

```bash
# Set custom messaging working directory in ~/.hermes/.env
MESSAGING_CWD=/home/myuser/projects
```

### Tool Progress Notifications

Get real-time updates as the agent works:

```bash
# Enable in ~/.hermes/.env
HERMES_TOOL_PROGRESS=true
HERMES_TOOL_PROGRESS_MODE=new    # or "all" for every tool call
```

When enabled, you'll see messages like:
```
💻 `ls -la`...
🔍 web_search...
📄 web_extract...
```

See [docs/messaging.md](docs/messaging.md) for WhatsApp and advanced setup.

---

## Commands

```bash
# Chat
hermes                    # Interactive chat (default)
hermes chat -q "Hello"    # Single query mode

# Provider & model management
hermes model              # Switch provider and model interactively
hermes login              # Authenticate with Nous Portal (OAuth)
hermes logout             # Clear stored OAuth credentials

# Configuration
hermes setup              # Full setup wizard (provider, terminal, messaging, etc.)
hermes config             # View/edit configuration
hermes config check       # Check for missing config (useful after updates)
hermes config migrate     # Interactively add missing options
hermes status             # Show configuration status (incl. auth)
hermes doctor             # Diagnose issues

# Maintenance
hermes update             # Update to latest version
hermes uninstall          # Uninstall (can keep configs for later reinstall)

# Messaging, skills, cron
hermes gateway            # Start messaging gateway
hermes skills search k8s  # Search skill registries
hermes skills install ... # Install a skill (with security scan)
hermes skills list        # List installed skills
hermes cron list          # View scheduled jobs
hermes pairing list       # View/manage DM pairing codes
hermes version            # Show version info
```

### CLI Commands (inside chat)

Type `/` to see an autocomplete dropdown of all commands.

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List available tools |
| `/toolsets` | List available toolsets |
| `/model [name]` | Show or change model |
| `/prompt` | View/set custom system prompt |
| `/personality [name]` | Set personality (kawaii, pirate, etc.) |
| `/clear` | Clear screen and reset conversation |
| `/history` | Show conversation history |
| `/reset` | Reset conversation only (keep screen) |
| `/retry` | Retry the last message |
| `/undo` | Remove the last exchange |
| `/save` | Save the current conversation |
| `/config` | Show current configuration |
| `/cron` | Manage scheduled tasks |
| `/skills` | Search, install, inspect, or manage skills from registries |
| `/platforms` | Show gateway/messaging platform status |
| `/quit` | Exit (also: `/exit`, `/q`) |

**Keybindings:**
- `Enter` — send message
- `Alt+Enter` or `Ctrl+J` — new line (multi-line input)
- `Ctrl+C` — interrupt agent (double-press to force exit)
- `Ctrl+D` — exit

---

## Features

### 🛠️ Tools & Toolsets

Tools are organized into logical **toolsets**:

```bash
# Use specific toolsets
hermes --toolsets "web,terminal"

# List all toolsets
hermes --list-tools
```

**Available toolsets:** `web`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `moa`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, and more.

### 🖥️ Terminal & Process Management

The terminal tool can execute commands in different environments, with full background process management via the `process` tool:

**Background processes:** Start with `terminal(command="...", background=true)`, then use `process(action="poll/wait/log/kill/write")` to monitor, wait for completion, read output, terminate, or send input. The `wait` action blocks until the process finishes -- no polling loops needed. PTY mode (`pty=true`) enables interactive CLI tools like Codex and Claude Code.

**Execution environments:**

| Backend | Description | Use Case |
|---------|-------------|----------|
| `local` | Run on your machine (default) | Development, trusted tasks |
| `docker` | Isolated containers | Security, reproducibility |
| `ssh` | Remote server | Sandboxing, keep agent away from its own code |
| `singularity` | HPC containers | Cluster computing, rootless |
| `modal` | Cloud execution | Serverless, scale |

**Configure in `~/.hermes/config.yaml`:**
```yaml
terminal:
  backend: local    # or: docker, ssh, singularity, modal
  cwd: "."          # Working directory ("." = current dir)
  timeout: 180      # Command timeout in seconds
```

**Docker Backend:**
```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**SSH Backend** (recommended for security - agent can't modify its own code):
```yaml
terminal:
  backend: ssh
```
```bash
# Set credentials in ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

**Singularity/Apptainer** (for HPC clusters):
```bash
# Pre-build SIF for parallel workers
apptainer build ~/python.sif docker://python:3.11-slim

# Configure
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

**Modal** (serverless cloud):
```bash
uv pip install "swe-rex[modal]"   # Installs swe-rex + modal + boto3
modal setup                    # Authenticate with Modal
hermes config set terminal.backend modal
```

**Sudo Support:** If a command needs sudo, you'll be prompted for your password (cached for the session). Or set `SUDO_PASSWORD` in `~/.hermes/.env`.

### 🧠 Persistent Memory

Bounded curated memory that persists across sessions:

- **MEMORY.md** — agent's personal notes (environment facts, conventions, things learned). ~800 token budget.
- **USER.md** — user profile (preferences, communication style, expectations). ~500 token budget.

Both are injected into the system prompt as a frozen snapshot at session start. The agent manages its own memory via the `memory` tool (add/replace/remove/read). Character limits keep memory focused — when full, the agent consolidates or replaces entries.

Configure in `~/.hermes/config.yaml`:
```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
```

### 📄 Context Files (SOUL.md, AGENTS.md, .cursorrules)

Drop these files in your project directory and the agent automatically picks them up:

| File | Purpose |
|------|---------|
| `AGENTS.md` | Project-specific instructions, coding conventions, tool usage guidelines |
| `SOUL.md` | Persona definition -- the agent embodies this personality and tone |
| `.cursorrules` | Cursor IDE rules (also detected) |
| `.cursor/rules/*.mdc` | Cursor rule files (also detected) |

- **AGENTS.md** is hierarchical: if subdirectories also have `AGENTS.md`, all are combined (like Codex/Cline).
- **SOUL.md** checks cwd first, then `~/.hermes/SOUL.md` as a global fallback.
- All context files are capped at 20,000 characters with smart truncation.

### 🗜️ Context Compression

Long conversations are automatically summarized when approaching context limits:

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.85    # Compress at 85% of limit
```

### 🗄️ Session Store

All CLI and messaging sessions are stored in a SQLite database (`~/.hermes/state.db`) with full-text search:

- **Full message history** stored per-session with model config and system prompt snapshots
- **FTS5 search** via the `session_search` tool -- search past conversations with Gemini Flash summarization
- **Compression-triggered session splitting** -- when context is compressed, a new session is created linked to the parent, giving clean trajectories
- **Source tagging** -- each session is tagged with its origin (cli, telegram, discord, etc.)
- Batch runner and RL trajectories are NOT stored here (separate systems)

### 📝 Session Logging

Every conversation is logged to `~/.hermes-agent/logs/` for debugging:

```
logs/
├── session_20260201_143052_a1b2c3.json
└── ...
```

### ⏰ Scheduled Tasks (Cron)

Schedule tasks to run automatically:

```bash
# In the CLI
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "0 9 * * *" "Morning briefing"
/cron list
/cron remove <job_id>
```

The agent can also self-schedule using `schedule_cronjob` tool.

**Run the scheduler:**
```bash
hermes cron daemon         # Built-in daemon
# Or add to system cron for reliability
```

### 🛡️ Exec Approval (Messaging Platforms)

When the agent tries to run a potentially dangerous command (rm -rf, chmod 777, etc.) on Telegram/Discord/WhatsApp, instead of blocking it silently, it asks the user for approval:

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

Reply "yes"/"y" to approve or "no"/"n" to deny. In CLI mode, the existing interactive approval prompt (once/session/always/deny) is preserved.

### 🔊 Text-to-Speech

Convert text to speech with three providers:

| Provider | Quality | Cost | API Key |
|----------|---------|------|---------|
| **Edge TTS** (default) | Good | Free | None needed |
| **ElevenLabs** | Excellent | Paid | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Good | Paid | `OPENAI_API_KEY` |

On Telegram, audio plays as native voice bubbles (the round, inline-playable kind). On Discord/WhatsApp, sent as audio file attachments. In CLI mode, saved to `~/voice-memos/`.

**Configure in `~/.hermes/config.yaml`:**
```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai"
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
```

**Telegram voice bubbles & ffmpeg:**

Telegram voice bubbles require Opus/OGG audio format. OpenAI and ElevenLabs produce Opus natively — no extra dependencies needed. Edge TTS (the default free provider) outputs MP3 and needs **ffmpeg** to convert to Opus:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

Without ffmpeg, Edge TTS audio is sent as a regular audio file (playable, but shows as a rectangular player instead of a voice bubble). If you want voice bubbles without installing ffmpeg, switch to the OpenAI or ElevenLabs provider.

### 🎙️ Voice Message Transcription

Voice messages sent on Telegram, Discord, WhatsApp, or Slack are automatically transcribed using OpenAI's Whisper API and injected as text into the conversation. The agent sees the transcript as normal text -- no special handling needed.

| Provider | Model | Quality | Cost |
|----------|-------|---------|------|
| **OpenAI Whisper** | `whisper-1` (default) | Good | Low |
| **OpenAI GPT-4o** | `gpt-4o-mini-transcribe` | Better | Medium |
| **OpenAI GPT-4o** | `gpt-4o-transcribe` | Best | Higher |

Requires `OPENAI_API_KEY` in `~/.hermes/.env`. Configure the model in `~/.hermes/config.yaml`:
```yaml
stt:
  enabled: true
  model: "whisper-1"
```

### 🌐 Browser Automation

Browser tools let the agent navigate websites, fill forms, click buttons, and extract content using [Browserbase](https://browserbase.com/).

**Setup:**
```bash
# 1. Get credentials from browserbase.com
hermes config set BROWSERBASE_API_KEY your_api_key
hermes config set BROWSERBASE_PROJECT_ID your_project_id

# 2. Install Node.js dependencies (if not already)
cd ~/.hermes-agent && npm install
```

**Available tools:** `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`, `browser_close`, `browser_get_images`

**Example:**
```bash
hermes --toolsets browser -q "Go to amazon.com and find the price of the latest Kindle"
```

### 📚 Skills System

Skills are on-demand knowledge documents the agent can load when needed. They follow a **progressive disclosure** pattern to minimize token usage and are compatible with the [agentskills.io](https://agentskills.io/specification) open standard.

All skills live in **`~/.hermes/skills/`** -- a single directory that is the source of truth. On fresh install, bundled skills are copied there from the repo. Hub-installed skills and agent-created skills also go here. The agent can modify or delete any skill. `hermes update` adds only genuinely new bundled skills (via a manifest) without overwriting your changes or re-adding skills you deleted.

**Using Skills:**
```bash
hermes --toolsets skills -q "What skills do you have?"
hermes --toolsets skills -q "Show me the axolotl skill"
```

**Agent-Managed Skills (skill_manage tool):**

The agent can create, update, and delete its own skills via the `skill_manage` tool. This is the agent's **procedural memory** -- when it figures out a non-trivial workflow, it can save the approach as a skill for future reuse.

The agent is encouraged to **create** skills when:
- It completed a complex task (5+ tool calls) successfully
- It hit errors or dead ends and found the working path
- The user corrected its approach and the corrected version worked
- It discovered a non-trivial workflow (deployment, data pipeline, configuration)

The agent is encouraged to **update** skills when:
- Instructions were stale or incorrect (outdated API, changed behavior)
- Steps didn't work on the current OS or environment
- Missing critical steps or pitfalls discovered during use

**Actions:**

| Action | Use for | Key params |
|--------|---------|------------|
| `create` | New skill from scratch | `name`, `content` (full SKILL.md), optional `category` |
| `patch` | Targeted fixes (preferred for updates) | `name`, `old_string`, `new_string` |
| `edit` | Major structural rewrites | `name`, `content` (full SKILL.md replacement) |
| `delete` | Remove a skill entirely | `name` |
| `write_file` | Add/update supporting files | `name`, `file_path`, `file_content` |
| `remove_file` | Remove a supporting file | `name`, `file_path` |

The `patch` action uses the same `old_string`/`new_string` pattern as the `patch` file tool -- find a unique string and replace it. This is more token-efficient than `edit` for small fixes (updating a command, adding a pitfall, fixing a version) because the model doesn't need to rewrite the entire skill. When patching SKILL.md, frontmatter integrity is validated after the replacement. The `patch` action also works on supporting files via the `file_path` parameter.

User-created skills are stored in `~/.hermes/skills/` and can optionally be organized into categories (subdirectories). Each skill has a `SKILL.md` file and may include supporting files under `references/`, `templates/`, `scripts/`, and `assets/`.

The `skill_manage` tool is enabled by default in CLI and all messaging platforms. It is **not** included in batch_runner or RL training environments.

**Skills Hub — Search, install, and manage skills from online registries:**
```bash
hermes skills search kubernetes          # Search all sources (GitHub, ClawHub, LobeHub)
hermes skills install openai/skills/k8s  # Install with security scan
hermes skills inspect openai/skills/k8s  # Preview before installing
hermes skills list --source hub          # List hub-installed skills
hermes skills audit                      # Re-scan all hub skills
hermes skills uninstall k8s              # Remove a hub skill
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json # Export skill config
hermes skills tap add myorg/skills-repo  # Add a custom source
```

All hub-installed skills go through a **security scanner** that checks for data exfiltration, prompt injection, destructive commands, and other threats. Trust levels: `builtin` (ships with Hermes), `trusted` (openai/skills, anthropics/skills), `community` (everything else — any findings = blocked unless `--force`).

**SKILL.md Format:**

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
metadata:
  hermes:
    tags: [python, automation]
    category: devops
---

# Skill Title

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes

## Verification
How to confirm it worked.
```

**Skill Directory Structure:**
```
~/.hermes/skills/                  # Single source of truth for all skills
├── mlops/                         # Category directory
│   ├── axolotl/
│   │   ├── SKILL.md               # Main instructions (required)
│   │   ├── references/            # Additional docs
│   │   ├── templates/             # Output formats
│   │   └── assets/                # Supplementary files (agentskills.io standard)
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Agent-created skill
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub state
│   ├── lock.json                  # Installed skill provenance
│   ├── quarantine/                # Pending security review
│   └── audit.log                  # Security scan history
└── .bundled_manifest              # Tracks which bundled skills have been offered
```

### 🐍 Code Execution (Programmatic Tool Calling)

The `execute_code` tool lets the agent write Python scripts that call Hermes tools programmatically, collapsing multi-step workflows into a single LLM turn. The script runs in a sandboxed child process on the agent host, communicating with the parent via Unix domain socket RPC.

```bash
# The agent can write scripts like:
from hermes_tools import web_search, web_extract
results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... filter and process ...
print(summary)
```

**Available tools in sandbox:** `web_search`, `web_extract`, `read_file`, `write_file`, `search`, `patch`, `terminal` (foreground only).

**When the agent uses this:** 3+ tool calls with processing logic between them, bulk data filtering, conditional branching, loops. The intermediate tool results never enter the context window -- only the final `print()` output comes back.

Configure via `~/.hermes/config.yaml`:
```yaml
code_execution:
  timeout: 300       # Max seconds per script (default: 300)
  max_tool_calls: 50 # Max tool calls per execution (default: 50)
```

### 🔀 Subagents (Task Delegation)

The `delegate_task` tool spawns child AIAgent instances with isolated context, restricted toolsets, and their own terminal sessions. Each child gets a fresh conversation and works independently -- only its final summary enters the parent's context.

**Single task:**
```
delegate_task(goal="Debug why tests fail", context="Error: assertion in test_foo.py line 42", toolsets=["terminal", "file"])
```

**Parallel batch (up to 3 concurrent):**
```
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

**Key properties:**
- Each subagent gets its own terminal session (separate from the parent)
- Depth limit of 2 (no grandchildren)
- Subagents cannot call: `delegate_task`, `clarify`, `memory`, `send_message`, `execute_code`
- Interrupt propagation: interrupting the parent interrupts all active children

Configure via `~/.hermes/config.yaml`:
```yaml
delegation:
  max_iterations: 25                        # Max turns per child (default: 25)
  default_toolsets: ["terminal", "file", "web"]  # Default toolsets
```

### 🤖 RL Training (Tinker + Atropos)

> **⚠️ In Development** — RL training integration is not yet functional. The tools and environments below are under active development.

Train language models with reinforcement learning using the Tinker API and Atropos framework.

#### Requirements

1. **API Keys:** Add to `~/.hermes/.env`:
```bash
TINKER_API_KEY=your-tinker-key      # Get from https://tinker-console.thinkingmachines.ai/keys
WANDB_API_KEY=your-wandb-key        # Get from https://wandb.ai/authorize
OPENROUTER_API_KEY=your-key         # Optional: for rl_test_inference
```

3. **That's it!** tinker-atropos is included as a submodule — the installer handles it automatically.

#### Using RL Tools

The agent can now use RL training tools:

```
You: Start training on GSM8k with group_size=16

Agent: I'll set up an RL training run on the GSM8k environment...
[Uses rl_list_environments, rl_select_environment, rl_edit_config, rl_start_training]
```

#### Available RL Tools

| Tool | Description |
|------|-------------|
| `rl_list_environments` | List available RL environments |
| `rl_select_environment` | Select an environment for training |
| `rl_get_current_config` | View all configurable options |
| `rl_edit_config` | Change a configuration value |
| `rl_test_inference` | Test environment with OpenRouter (pre-training validation) |
| `rl_start_training` | Start a training run |
| `rl_check_status` | Check training progress |
| `rl_stop_training` | Stop a running training |
| `rl_get_results` | Fetch WandB metrics |
| `rl_list_runs` | List active training runs |

#### Dedicated RL CLI

For extended RL workflows with longer timeouts:

```bash
python rl_cli.py --model "anthropic/claude-sonnet-4-20250514"
```

### 🧪 Atropos RL Environments

Hermes-Agent integrates with the [Atropos](https://github.com/NousResearch/atropos) RL framework through a layered environment system. This allows training models with reinforcement learning on agentic tasks using hermes-agent's tools.

#### Architecture

The integration has three layers:

| Layer | File | Purpose |
|-------|------|---------|
| **Agent Loop** | `environments/agent_loop.py` | Reusable multi-turn tool-calling engine (standard OpenAI spec) |
| **Base Environment** | `environments/hermes_base_env.py` | Abstract Atropos `BaseEnv` subclass with toolset resolution, ToolContext, scoring |
| **Concrete Envs** | `environments/terminal_test_env.py`, `environments/hermes_swe_env.py` | Task-specific environments |

#### Two-Phase Operation

- **Phase 1 (OpenAI server type)**: Works with any OpenAI-compatible endpoint (VLLM, SGLang, OpenRouter, OpenAI API). The server handles tool call parsing natively. Good for **SFT data generation**, **verifier testing**, and **evaluation**.
- **Phase 2 (VLLM server type)**: Uses ManagedServer for exact token IDs + logprobs via `/generate`. Client-side tool call parser registry reconstructs structured `tool_calls` from raw output. Required for **full RL training**.

#### Quick Start

```bash
# 1. Launch VLLM with tool parser
vllm serve YourModel --tool-parser hermes

# 2. Start the Atropos API server
run-api

# 3. Run an environment
python environments/terminal_test_env.py serve \
    --openai.base_url http://localhost:8000/v1 \
    --openai.model_name YourModel \
    --openai.server_type openai
```

#### ToolContext (Reward Functions)

Reward functions receive a `ToolContext` with unrestricted access to all hermes-agent tools, scoped to the rollout's sandbox:

```python
async def compute_reward(self, item, result, ctx: ToolContext) -> float:
    # Run tests in the model's terminal sandbox
    test = ctx.terminal("pytest -v")
    if test["exit_code"] == 0:
        return 1.0
    # Or check a file, search the web, navigate a browser...
    return 0.0
```

#### Creating Custom Environments

Subclass `HermesAgentBaseEnv` and implement 5 methods:

```python
from environments.hermes_base_env import HermesAgentBaseEnv

class MyEnv(HermesAgentBaseEnv):
    name = "my-env"
    async def setup(self): ...            # Load data
    async def get_next_item(self): ...    # Return next item
    def format_prompt(self, item): ...    # Item -> prompt string
    async def compute_reward(self, item, result, ctx): ...  # Score with ToolContext
    async def evaluate(self, *args, **kwargs): ...          # Periodic eval

if __name__ == "__main__":
    MyEnv.cli()
```

#### Toolset Distributions

Configure which tools are available per group, either explicitly or probabilistically:

```bash
# Explicit toolsets
--env.enabled_toolsets '["terminal","file","web"]'

# Probabilistic distribution (sampled per group)
--env.distribution development
```

#### Tool Call Parsers (Phase 2)

For VLLM server type, a parser registry extracts structured `tool_calls` from raw model output. Supported parsers: `hermes`, `mistral`, `llama3_json`, `qwen`, `deepseek_v3`, `deepseek_v3_1`, `kimi_k2`, `longcat`, `glm45`, `glm47`, `qwen3_coder`.

```bash
--env.tool_call_parser hermes  # Match your VLLM --tool-parser flag
```

---

## Manual Installation

If you prefer full control over the installation process (or the quick-install script doesn't suit your environment), follow these steps to set everything up by hand.

### Prerequisites

| Requirement | Minimum Version | Check Command | Notes |
|-------------|----------------|---------------|-------|
| **Git** | Any recent | `git --version` | Required |
| **Node.js** | 18+ | `node --version` | Optional — needed for browser automation tools |
| **ripgrep** | Any | `rg --version` | Optional — faster file search in terminal tool (falls back to grep) |

> **Note:** Python and pip are **not** prerequisites. The installer uses [uv](https://docs.astral.sh/uv/) to provision Python 3.11 automatically (no sudo needed). If you already have Python 3.11+ installed, uv will use it.

<details>
<summary><strong>Installing prerequisites by platform</strong></summary>

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install git
# Optional:
sudo apt install ripgrep nodejs npm
```

**macOS (Homebrew):**
```bash
brew install git
# Optional:
brew install ripgrep node
```

**Windows (WSL recommended):**
Use the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the Ubuntu instructions above. Alternatively, use the PowerShell quick-install script at the top of this README.

</details>

---

### Step 1: Clone the Repository

Clone with `--recurse-submodules` to pull the required submodules ([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) for the terminal tool backend and [tinker-atropos](https://github.com/nousresearch/tinker-atropos) for RL training):

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

If you already cloned without `--recurse-submodules`, initialize them manually:
```bash
git submodule update --init --recursive
```

---

### Step 2: Install uv & Create Virtual Environment

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that can also provision Python itself. Install it and create the venv in one go:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with Python 3.11 (uv downloads it if not present — no sudo needed)
uv venv venv --python 3.11
```

> **Tip:** You do **not** need to activate the venv to use `hermes`. The entry point has a hardcoded shebang pointing to the venv Python, so it works globally once symlinked (see Step 8). For installing packages, uv can target the venv directly via `VIRTUAL_ENV`.

---

### Step 3: Install Python Dependencies

Install the main package in editable mode with all optional extras (messaging, cron, CLI menus, modal):

```bash
# Tell uv which venv to install into
export VIRTUAL_ENV="$(pwd)/venv"

# Install with all extras
uv pip install -e ".[all]"
```

If you only want the core agent (no Telegram/Discord/cron support):
```bash
uv pip install -e "."
```

<details>
<summary><strong>Optional extras breakdown</strong></summary>

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `all` | Everything below | `uv pip install -e ".[all]"` |
| `messaging` | Telegram & Discord gateway | `uv pip install -e ".[messaging]"` |
| `cron` | Cron expression parsing for scheduled tasks | `uv pip install -e ".[cron]"` |
| `cli` | Terminal menu UI for setup wizard | `uv pip install -e ".[cli]"` |
| `modal` | Modal cloud execution backend (swe-rex + modal + boto3) | `uv pip install -e ".[modal]"` |
| `dev` | pytest & test utilities | `uv pip install -e ".[dev]"` |

You can combine extras: `uv pip install -e ".[messaging,cron]"`

</details>

---

### Step 4: Install Submodule Packages

These are local packages checked out as Git submodules. Install them in editable mode:

```bash
# Terminal tool backend (required for the terminal/command-execution tool)
uv pip install -e "./mini-swe-agent"

# RL training backend
uv pip install -e "./tinker-atropos"
```

Both are optional — if you skip them, the corresponding toolsets simply won't be available.

---

### Step 5: Install Node.js Dependencies (Optional)

Only needed if you plan to use the **browser automation** toolset (Browserbase-powered):

```bash
npm install
```

This installs the `agent-browser` package defined in `package.json`. Skip this step if you don't need browser tools.

---

### Step 6: Create the Configuration Directory

Hermes stores all user configuration in `~/.hermes/`:

```bash
# Create the directory structure
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}

# Copy the example config file
cp cli-config.yaml.example ~/.hermes/config.yaml

# Create an empty .env file for API keys
touch ~/.hermes/.env
```

Your `~/.hermes/` directory should now look like:
```
~/.hermes/
├── config.yaml     # Agent settings (model, terminal, toolsets, compression, etc.)
├── .env            # API keys and secrets (one per line: KEY=value)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (auto-created on first use)
├── cron/           # Scheduled job data
├── sessions/       # Messaging gateway sessions
└── logs/           # Conversation logs
```

---

### Step 7: Add Your API Keys

Open `~/.hermes/.env` in your editor and add at minimum an LLM provider key:

```bash
# Required — at least one LLM provider:
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional — enable additional tools:
FIRECRAWL_API_KEY=fc-your-key          # Web search & scraping
BROWSERBASE_API_KEY=bb-your-key        # Browser automation
BROWSERBASE_PROJECT_ID=your-project-id # Browser automation
FAL_KEY=your-fal-key                   # Image generation (FLUX)
TINKER_API_KEY=your-tinker-key         # RL training
WANDB_API_KEY=your-wandb-key           # RL training metrics

# Optional — messaging gateway:
TELEGRAM_BOT_TOKEN=123456:ABC-DEF      # From @BotFather
TELEGRAM_ALLOWED_USERS=your-user-id    # Comma-separated
DISCORD_BOT_TOKEN=MTIz...              # From Developer Portal
DISCORD_ALLOWED_USERS=your-user-id     # Comma-separated
```

Or set them one at a time via the CLI:
```bash
hermes config set OPENROUTER_API_KEY sk-or-v1-your-key-here
```

---

### Step 8: Add `hermes` to Your PATH

The `hermes` entry point at `venv/bin/hermes` has a hardcoded shebang pointing to the venv's Python, so it works **without activating the venv**. The recommended approach is a symlink into `~/.local/bin` (most distributions already have this on PATH):

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
```

If `~/.local/bin` isn't on your PATH yet, add it:

**Bash** (`~/.bashrc`):
```bash
echo '' >> ~/.bashrc
echo '# Hermes Agent' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Zsh** (`~/.zshrc`):
```bash
echo '' >> ~/.zshrc
echo '# Hermes Agent' >> ~/.zshrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Fish** (`~/.config/fish/config.fish`):
```fish
fish_add_path $HOME/.local/bin
```

---

### Step 9: Run the Setup Wizard (Optional)

The interactive setup wizard walks you through configuring your API keys and preferences:

```bash
hermes setup
```

This is optional if you already configured `~/.hermes/.env` and `~/.hermes/config.yaml` manually in the steps above.

---

### Step 10: Verify the Installation

```bash
# Check that the command is available
hermes version

# Run diagnostics to verify everything is working
hermes doctor

# Check your configuration
hermes status

# Test with a quick query
hermes chat -q "Hello! What tools do you have available?"
```

If `hermes doctor` reports issues, it will tell you exactly what's missing and how to fix it.

---

### Quick-Reference: Manual Install (Condensed)

For those who just want the commands without the explanations:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & enter
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11 (uv downloads it if needed)
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# Install everything
uv pip install -e ".[all]"
uv pip install -e "./mini-swe-agent"
uv pip install -e "./tinker-atropos"
npm install  # optional, for browser tools

# Configure
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' >> ~/.hermes/.env

# Make hermes available globally (no venv activation needed)
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# Verify
hermes doctor
hermes
```

---

## Batch Processing

Process multiple prompts in parallel with automatic checkpointing:

```bash
python batch_runner.py \
  --dataset_file=prompts.jsonl \
  --batch_size=20 \
  --run_name=my_run \
  --num_workers=4 \
  --distribution=default
```

**Key Options:**
| Flag | Description |
|------|-------------|
| `--dataset_file` | JSONL file with prompts |
| `--batch_size` | Prompts per batch |
| `--run_name` | Name for output/checkpoints |
| `--num_workers` | Parallel workers (default: 4) |
| `--distribution` | Toolset distribution |
| `--resume` | Resume from checkpoint |
| `--ephemeral_system_prompt` | Guide behavior without saving to trajectories |
| `--list_distributions` | Show available distributions |

**Output:** `data/<run_name>/trajectories.jsonl`

### Trajectory Compression

Compress trajectories to fit token budgets for training:

```bash
# Compress a directory
python trajectory_compressor.py --input=data/my_run

# Compress with sampling
python trajectory_compressor.py --input=data/my_run --sample_percent=15

# Custom token target
python trajectory_compressor.py --input=data/my_run --target_max_tokens=16000
```

Features:
- Protects first/last turns
- Summarizes middle turns via LLM
- Configurable via `configs/trajectory_compression.yaml`

---

## Python API

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    enabled_toolsets=["web", "terminal"]
)

result = agent.run_conversation("Search for the latest Python news")
print(result["final_response"])
```

---

## Environment Variables Reference

All variables go in `~/.hermes/.env`. Run `hermes config set VAR value` to set them.

**LLM Providers:**
| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (recommended for flexibility) |
| `ANTHROPIC_API_KEY` | Direct Anthropic access |
| `OPENAI_API_KEY` | Direct OpenAI access |

**Provider Auth (OAuth):**
| Variable | Description |
|----------|-------------|
| `HERMES_INFERENCE_PROVIDER` | Override provider selection: `auto`, `openrouter`, `nous` (default: `auto`) |
| `HERMES_PORTAL_BASE_URL` | Override Nous Portal URL (for development/testing) |
| `NOUS_INFERENCE_BASE_URL` | Override Nous inference API URL |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | Min agent key TTL before re-mint (default: 1800 = 30min) |
| `HERMES_DUMP_REQUESTS` | Dump API request payloads to log files for debugging (`true`/`false`) |

**Tool APIs:**
| Variable | Description |
|----------|-------------|
| `FIRECRAWL_API_KEY` | Web scraping (firecrawl.dev) |
| `BROWSERBASE_API_KEY` | Browser automation |
| `BROWSERBASE_PROJECT_ID` | Browserbase project |
| `FAL_KEY` | Image generation (fal.ai) |

**Terminal Backend:**
| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | Backend: `local`, `docker`, `ssh`, `singularity`, `modal` |
| `TERMINAL_DOCKER_IMAGE` | Docker image (default: `python:3.11-slim`) |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity image or `.sif` path |
| `TERMINAL_TIMEOUT` | Command timeout in seconds |
| `TERMINAL_CWD` | Working directory |
| `SUDO_PASSWORD` | Enable sudo (stored plaintext - be careful!) |

**SSH Backend:**
| Variable | Description |
|----------|-------------|
| `TERMINAL_SSH_HOST` | Remote server hostname |
| `TERMINAL_SSH_USER` | SSH username |
| `TERMINAL_SSH_PORT` | SSH port (default: 22) |
| `TERMINAL_SSH_KEY` | Path to private key |

**Messaging:**
| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (@BotFather) |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated user IDs allowed to use bot |
| `TELEGRAM_HOME_CHANNEL` | Default channel for cron delivery |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_ALLOWED_USERS` | Comma-separated user IDs allowed to use bot |
| `DISCORD_HOME_CHANNEL` | Default channel for cron delivery |
| `MESSAGING_CWD` | Working directory for terminal in messaging (default: ~) |

**Agent Behavior:**
| Variable | Description |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | Max tool-calling iterations per conversation (default: 60) |
| `HERMES_TOOL_PROGRESS` | Send progress messages when using tools (`true`/`false`) |
| `HERMES_TOOL_PROGRESS_MODE` | `new` (only when tool changes) or `all` (every call) |

**Context Compression:**
| Variable | Description |
|----------|-------------|
| `CONTEXT_COMPRESSION_ENABLED` | Enable auto-compression (default: true) |
| `CONTEXT_COMPRESSION_THRESHOLD` | Trigger at this % of limit (default: 0.85) |
| `CONTEXT_COMPRESSION_MODEL` | Model for summaries |

---

## File Structure

| Path | Description |
|------|-------------|
| `~/.hermes/config.yaml` | Your settings |
| `~/.hermes/.env` | API keys and secrets |
| `~/.hermes/auth.json` | OAuth provider credentials (managed by `hermes login`) |
| `~/.hermes/cron/` | Scheduled jobs data |
| `~/.hermes/sessions/` | Gateway session data |
| `~/.hermes-agent/` | Installation directory |
| `~/.hermes-agent/logs/` | Session logs |
| `hermes_cli/` | CLI implementation |
| `hermes_cli/auth.py` | Multi-provider auth system |
| `tools/` | Tool implementations |
| `skills/` | Bundled skill sources (copied to `~/.hermes/skills/` on install) |
| `~/.hermes/skills/` | All active skills (bundled + hub-installed + agent-created) |
| `gateway/` | Messaging platform adapters |
| `cron/` | Scheduler implementation |

---

## Troubleshooting

```bash
hermes doctor    # Run diagnostics
hermes status    # Check configuration
hermes config    # View current settings
```

Common issues:
- **"API key not set"**: Run `hermes setup` or `hermes config set OPENROUTER_API_KEY your_key`
- **"hermes: command not found"**: Reload your shell (`source ~/.bashrc`) or check PATH
- **"Run `hermes login` to re-authenticate"**: Your Nous Portal session expired. Run `hermes login` to refresh.
- **"No active paid subscription"**: Your Nous Portal account needs an active subscription for inference.
- **Gateway won't start**: Check `hermes gateway status` and logs
- **Missing config after update**: Run `hermes config check` to see what's new, then `hermes config migrate` to add missing options
- **Provider auto-detection wrong**: Force a provider with `hermes chat --provider openrouter` or set `HERMES_INFERENCE_PROVIDER` in `.env`

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.
