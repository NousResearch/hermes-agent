# Hermes Agent 🦋

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
- Clone to `~/.hermes-agent`
- Create a virtual environment
- Install all dependencies
- Run the interactive setup wizard
- Add `hermes` to your PATH

After installation, reload your shell and run:
```bash
hermes setup    # Configure API keys (if you skipped during install)
hermes          # Start chatting!
```

---

## Configuration

All your settings are stored in `~/.hermes/` for easy access:

```
~/.hermes/
├── config.yaml     # Settings (model, terminal, compression, etc.)
├── .env            # API keys and secrets
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs
```

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

### Required API Keys

You need at least one LLM provider:

| Provider | Get Key | Env Variable |
|----------|---------|--------------|
| **OpenRouter** (recommended) | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | `ANTHROPIC_API_KEY` |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |

### Optional API Keys

| Feature | Provider | Env Variable |
|---------|----------|--------------|
| Web scraping | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY` |
| Browser automation | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| Image generation | [FAL](https://fal.ai/) | `FAL_KEY` |
| Messaging | Telegram, Discord | `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN` |

---

## Commands

```bash
hermes                    # Interactive chat (default)
hermes chat -q "Hello"    # Single query mode
hermes setup              # Configure API keys and settings
hermes config             # View/edit configuration
hermes config check       # Check for missing config (useful after updates)
hermes config migrate     # Interactively add missing options
hermes status             # Show configuration status
hermes doctor             # Diagnose issues
hermes update             # Update to latest version (prompts for new config)
hermes uninstall          # Uninstall (can keep configs for later reinstall)
hermes gateway            # Start messaging gateway
hermes cron list          # View scheduled jobs
hermes version            # Show version info
```

### CLI Commands (inside chat)

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List available tools |
| `/model [name]` | Show or change model |
| `/personality [name]` | Set personality (kawaii, pirate, etc.) |
| `/clear` | Clear screen and reset |
| `/cron` | Manage scheduled tasks |
| `/config` | Show current configuration |
| `/quit` | Exit |

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

**Available toolsets:** `web`, `terminal`, `browser`, `vision`, `creative`, `reasoning`, `skills`, `cronjob`, and more.

### 🖥️ Terminal Backend

The terminal tool can execute commands in different environments:

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
pip install modal boto3
modal setup  # Authenticate
hermes config set terminal.backend modal
```

**Sudo Support:** If a command needs sudo, you'll be prompted for your password (cached for the session). Or set `SUDO_PASSWORD` in `~/.hermes/.env`.

### 📱 Messaging Gateway

Chat with Hermes from Telegram, Discord, or WhatsApp.

#### Telegram Setup

1. **Create a bot:** Message [@BotFather](https://t.me/BotFather) on Telegram, use `/newbot`
2. **Get your user ID:** Message [@userinfobot](https://t.me/userinfobot) - it replies with your numeric ID
3. **Configure:**

```bash
# Add to ~/.hermes/.env:
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_USERS=YOUR_USER_ID    # Comma-separated for multiple users
```

4. **Start the gateway:**

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as systemd service (Linux)
hermes gateway start        # Start the service
```

#### Discord Setup

1. **Create a bot:** Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. **Get your user ID:** Enable Developer Mode in Discord settings, right-click your name → Copy ID
3. **Configure:**

```bash
# Add to ~/.hermes/.env:
DISCORD_BOT_TOKEN=MTIz...
DISCORD_ALLOWED_USERS=YOUR_USER_ID
```

#### Security (Important!)

**Without an allowlist, anyone who finds your bot can use it!**

```bash
# Restrict to specific users (recommended):
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678

# Or allow all users in a specific platform:
# (Leave the variable unset - NOT recommended for bots with terminal access)
```

#### Gateway Commands

| Command | Description |
|---------|-------------|
| `/new` or `/reset` | Start fresh conversation |
| `/status` | Show session info |

See [docs/messaging.md](docs/messaging.md) for WhatsApp and advanced setup.

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

### 🗜️ Context Compression

Long conversations are automatically summarized when approaching context limits:

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.85    # Compress at 85% of limit
```

### 📝 Session Logging

Every conversation is logged to `~/.hermes-agent/logs/` for debugging:

```
logs/
├── session_20260201_143052_a1b2c3.json
└── ...
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

Skills are on-demand knowledge documents the agent can load when needed. They follow a **progressive disclosure** pattern to minimize token usage.

**Using Skills:**
```bash
hermes --toolsets skills -q "What skills do you have?"
hermes --toolsets skills -q "Show me the axolotl skill"
```

**Creating Skills:**

Create `skills/category/skill-name/SKILL.md`:
```markdown
---
name: my-skill
description: Brief description shown in skills_list
tags: [python, automation]
version: 1.0.0
---

# Skill Content

Instructions, examples, and guidelines here...
```

**Skill Structure:**
```
skills/
├── mlops/
│   ├── axolotl/
│   │   ├── SKILL.md          # Main instructions (required)
│   │   ├── references/       # Additional docs
│   │   └── templates/        # Output formats
│   └── vllm/
│       └── SKILL.md
```

---

## Manual Installation

If you prefer not to use the installer:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Run setup script
./setup-hermes.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"
hermes setup
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
| `OPENROUTER_API_KEY` | OpenRouter API key (recommended) |
| `ANTHROPIC_API_KEY` | Direct Anthropic access |
| `OPENAI_API_KEY` | Direct OpenAI access |

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
| `TELEGRAM_HOME_CHANNEL` | Default channel for cron delivery |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_HOME_CHANNEL` | Default channel for cron delivery |

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
| `~/.hermes/cron/` | Scheduled jobs data |
| `~/.hermes/sessions/` | Gateway session data |
| `~/.hermes-agent/` | Installation directory |
| `~/.hermes-agent/logs/` | Session logs |
| `hermes_cli/` | CLI implementation |
| `tools/` | Tool implementations |
| `skills/` | Knowledge documents |
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
- **Gateway won't start**: Check `hermes gateway status` and logs
- **Missing config after update**: Run `hermes config check` to see what's new, then `hermes config migrate` to add missing options

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.
