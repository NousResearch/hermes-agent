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


### Optional API Keys

| Feature | Provider | Env Variable |
|---------|----------|--------------|
| Custom OpenAI Endpoint (OAI or VLLM/SGLANG) | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Web scraping | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY` |
| Browser automation | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| Image generation | [FAL](https://fal.ai/) | `FAL_KEY` |
| RL Training | [Tinker](https://tinker-console.thinkingmachines.ai/) + [WandB](https://wandb.ai/) | `TINKER_API_KEY`, `WANDB_API_KEY` |
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
uv pip install "swe-rex[modal]"   # Installs swe-rex + modal + boto3
modal setup                    # Authenticate with Modal
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

#### Working Directory

- **CLI (`hermes`)**: Uses current directory where you run the command
- **Messaging**: Uses `MESSAGING_CWD` (default: home directory `~`)

```bash
# Set custom messaging working directory in ~/.hermes/.env
MESSAGING_CWD=/home/myuser/projects
```

#### Tool Progress Notifications

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

### 🤖 RL Training (Tinker + Atropos)

Train language models with reinforcement learning using the Tinker API and Atropos framework.

> **Note:** RL training tools require **Python 3.11+** (the upstream `tinker` package has this requirement). On Python 3.10, the RL toolset will be automatically disabled — all other features work fine.

#### Requirements

1. **Python 3.11+** (check with `python3 --version`)
2. **API Keys:** Add to `~/.hermes/.env`:
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
mkdir -p ~/.hermes/{cron,sessions,logs}

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
mkdir -p ~/.hermes/{cron,sessions,logs}
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

### Updating a Manual Installation

To update an existing manual install to the latest version:

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

### Uninstalling a Manual Installation

```bash
# Remove the hermes symlink
rm -f ~/.local/bin/hermes

# Remove the cloned repository
rm -rf /path/to/hermes-agent

# Remove user configuration (optional — keep if you plan to reinstall)
rm -rf ~/.hermes
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

## RL Training with Tinker

Hermes-Agent includes an RL training integration with [Tinker](https://thinkingmachines.ai/tinker/) (Thinking Machines) and [Atropos](https://github.com/NousResearch/atropos) for training language models with reinforcement learning from agent trajectories.

### Prerequisites

1. **Install with Atropos extras** (includes Tinker SDK, atroposlib, torch, wandb):
```bash
pip install -e ".[atropos]"
```

2. **Initialize the tinker-atropos submodule**:
```bash
git submodule update --init
pip install -e ./tinker-atropos
```

3. **Get API keys**:
   - `TINKER_API_KEY` from [Tinker Console](https://tinker-console.thinkingmachines.ai/keys) (requires billing setup)
   - `WANDB_API_KEY` from [Weights & Biases](https://wandb.ai/settings) (for metrics tracking)

4. **Add keys to your `.env` file**:
```bash
# Add to .env or ~/.hermes/.env
TINKER_API_KEY=your_tinker_key
WANDB_API_KEY=your_wandb_key
```

### Architecture

The RL training pipeline uses three processes that communicate over HTTP:

```
┌──────────────────────┐   ┌─────────────────────┐   ┌────────────────────────┐
│ Atropos Rollout API  │   │ Tinker Trainer       │   │ Environment            │
│ (port 8000)          │◄──│ (port 8001)          │◄──│ (worker)               │
│                      │   │                      │   │                        │
│ • Collects batches   │   │ • LoRA training      │   │ • Generates prompts    │
│ • Coordinates env    │   │ • Inference server   │   │ • Calls inference API  │
│   and trainer        │   │ • Weight updates     │   │ • Scores responses     │
│                      │   │ • WandB logging      │   │ • Sends scored batches │
└──────────────────────┘   └─────────────────────┘   └────────────────────────┘
```

### Quick Start: GSM8k Agent Training

This example trains a model on math problems using a Python REPL tool — the model learns to write and execute Python code to solve math:

```bash
# Terminal 1: Start Atropos Rollout API
cd tinker-atropos
source ../.venv/bin/activate
set -a && source ../.env && set +a
run-api

# Terminal 2: Start Tinker Trainer + Inference Server
cd tinker-atropos
source ../.venv/bin/activate
set -a && source ../.env && set +a
python launch_training.py --config configs/gsm8k_agent.yaml

# Terminal 3: Start GSM8k Agent Environment
cd tinker-atropos
source ../.venv/bin/activate
set -a && source ../.env && set +a
python tinker_atropos/environments/gsm8k_agent.py serve --config configs/gsm8k_agent.yaml
```

### Available Environments

| Environment | File | Description |
|------------|------|-------------|
| `gsm8k` | `gsm8k_tinker.py` | Standard GSM8k math (no tools) |
| `gsm8k_agent` | `gsm8k_agent.py` | GSM8k with Python REPL tool calling |

### Configuration

Configs are YAML files in `tinker-atropos/configs/` with three sections:

```yaml
env:                              # Atropos environment settings
  group_size: 4                   # Parallel rollouts per problem
  batch_size: 16                  # Training batch size
  tokenizer_name: "Qwen/Qwen3-4B-Instruct-2507"
  max_token_length: 2048          # Max generation length
  total_steps: 20                 # Training steps

openai:                           # Inference server (served by Tinker trainer)
  - model_name: "Qwen/Qwen3-4B-Instruct-2507"
    base_url: "http://localhost:8001/v1"

tinker:                           # Tinker training parameters
  lora_rank: 16                   # LoRA rank (lower = faster, less capacity)
  learning_rate: 0.00005          # Learning rate
  max_token_trainer_length: 4096  # Max tokens for training
  wandb_project: "hermes-agent-rl"
```

### RL CLI (Agent-Driven Training)

For interactive training management via the Hermes agent:

```bash
# Interactive mode - let the agent manage training
python rl_cli.py --interactive

# List available environments
python rl_cli.py --list-environments

# Direct task
python rl_cli.py "Train a model on GSM8k with tool use"
```

### Sandbox Backends for Agent Environments

For agent environments that need isolated tool execution (e.g., SWE tasks), Hermes-Agent supports multiple sandbox backends:

| Backend | Use Case | Command |
|---------|----------|---------|
| **Nomad + Docker** | Default, local development | `--env.tool_pool_mode nomad` |
| **Nomad + Singularity** | HPC clusters without Docker | `--env.tool_pool_mode nomad --env.driver singularity` |
| **Modal** | Cloud-based, auto-scaling | `--env.tool_pool_mode modal` |

See [docs/MODAL_BACKEND.md](docs/MODAL_BACKEND.md) for Modal backend details.

### Cost

Check the [Tinker Rate Card](https://tinker-console.thinkingmachines.ai/rate-card) for available models and pricing.

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
