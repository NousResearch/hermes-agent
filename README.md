# Hermes Agent

An AI agent with advanced tool-calling capabilities, featuring a flexible toolsets system for organizing and managing tools.

## Features

- **Interactive CLI**: Beautiful terminal interface with animated feedback, personalities, and session management
- **Web Tools**: Search, extract content, and crawl websites
- **Terminal Tools**: Execute commands via local, Docker, Singularity, Modal, or SSH backends
- **Browser Tools**: Automate web browsers to navigate, click, type, and extract content
- **Vision Tools**: Analyze images from URLs
- **Reasoning Tools**: Advanced multi-model reasoning (Mixture of Agents)
- **Creative Tools**: Generate images from text prompts
- **Skills Tools**: On-demand knowledge documents with progressive disclosure
- **Toolsets System**: Organize tools into logical groups for different scenarios
- **Batch Processing**: Process datasets in parallel with checkpointing and statistics tracking
- **Ephemeral System Prompts**: Guide model behavior without polluting training datasets

## Quick Start (CLI)

```bash
# After setup (see below), just run:
./hermes

# Or with options:
./hermes --model "anthropic/claude-sonnet-4" --toolsets "web,terminal"
```

The CLI provides:
- Animated spinners during thinking and tool execution
- Kawaii-style feedback messages
- `/commands` for configuration, history, and session management
- Customizable personalities (`/personality kawaii`, `/personality pirate`, etc.)
- Persistent configuration via `cli-config.yaml`

## Setup

### 1. Clone the Repository
```bash
# Clone with submodules (recommended)
git clone --recurse-submodules https://github.com/NousResearch/Hermes-Agent.git
cd Hermes-Agent

# Or if already cloned without submodules:
git submodule update --init --recursive
```

### 2. Install Dependencies
```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install mini-swe-agent for terminal tools
pip install -e ./mini-swe-agent

# Install Node.js dependencies for browser tools (requires Node.js)
npm install
```

### 3. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
- `OPENROUTER_API_KEY` - LLM access via OpenRouter (get at: https://openrouter.ai/keys)
- `FIRECRAWL_API_KEY` - Web tools (get at: https://firecrawl.dev/)
- `NOUS_API_KEY` - Vision & reasoning tools (get at: https://inference-api.nousresearch.com/)
- `FAL_KEY` - Image generation (get at: https://fal.ai/)

**Optional API Keys (for specific features):**
- `BROWSERBASE_API_KEY` - Browser automation (get at: https://browserbase.com/)
- `BROWSERBASE_PROJECT_ID` - From Browserbase dashboard
- `MORPH_API_KEY` - For legacy Hecate terminal backend (get at: https://morph.so/)

### 4. Configure Terminal Backend

The terminal tool uses **mini-swe-agent** environments. Configure in `.env` or `cli-config.yaml`:

```bash
# Backend: "local", "docker", "singularity", "modal", or "ssh"
TERMINAL_ENV=local          # Default: runs on host machine (no isolation)
TERMINAL_ENV=ssh            # Remote execution via SSH (agent code stays local)
TERMINAL_ENV=singularity    # Recommended for HPC: Apptainer/Singularity containers
TERMINAL_ENV=docker         # Isolated Docker containers
TERMINAL_ENV=modal          # Cloud execution via Modal

# Container image (for docker/singularity/modal backends)
TERMINAL_DOCKER_IMAGE=python:3.11-slim
TERMINAL_SINGULARITY_IMAGE=docker://python:3.11-slim
TERMINAL_TIMEOUT=60

# SSH backend (for ssh)
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa  # Optional, uses ssh-agent if not set
```

**Backend Requirements:**
- **local**: No extra setup (runs directly on your machine, no isolation)
- **ssh**: SSH access to remote machine (great for sandboxing - agent can't touch its own code)
- **singularity**: Requires Apptainer or Singularity installed (common on HPC clusters, no root needed)
- **docker**: Requires Docker installed and user in `docker` group
- **modal**: Requires Modal account (see setup below)

### Singularity/Apptainer Setup (Recommended for HPC)

Singularity/Apptainer provides rootless container execution, ideal for HPC clusters:

```bash
# 1. Verify Apptainer is installed
apptainer --version  # or: singularity --version

# 2. Set up cache directories (important for parallel workers)
# Use /scratch if available (HPC), otherwise /tmp
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer
export APPTAINER_TMPDIR=/scratch/$USER/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# 3. Pre-build SIF image (recommended for parallel batch processing)
# This avoids race conditions when multiple workers start simultaneously
apptainer build $APPTAINER_CACHEDIR/python-nodejs.sif docker://nikolaik/python-nodejs:python3.11-nodejs20

# 4. Configure .env to use the local SIF
TERMINAL_ENV=singularity
TERMINAL_SINGULARITY_IMAGE=/scratch/$USER/.apptainer/python-nodejs.sif
```

**Tip:** The batch scripts in `configs/` automatically handle SIF pre-building if `/scratch` is available.

### Modal Cloud Backend Setup

[Modal](https://modal.com) provides serverless cloud compute for running sandboxed environments at scale.

```bash
# 1. Install Modal and dependencies
pip install modal boto3

# 2. Authenticate with Modal (opens browser)
modal setup

# 3. Set terminal backend to modal in .env
TERMINAL_ENV=modal
```

Modal uses CLI-based authentication (stored in `~/.modal/`), so no API key is needed in `.env`. After running `modal setup`, commands will automatically execute in Modal's cloud sandboxes.

### Browser Tools Setup

Browser tools enable the agent to navigate websites, fill forms, click buttons, and extract content. They use [agent-browser](https://github.com/vercel-labs/agent-browser) CLI with [Browserbase](https://browserbase.com) cloud execution.

```bash
# 1. Install Node.js (if not already installed)
# Use nvm (recommended) or your package manager

# 2. Install agent-browser CLI (choose one option):
npm install -g agent-browser     # Option A: Global install (recommended)
npm install                      # Option B: Local install (uses npx fallback)

# 3. Get Browserbase credentials
# Sign up at https://browserbase.com/ and get your:
# - API Key (from Settings → API Keys)
# - Project ID (from your project dashboard)

# 4. Add to your .env file:
BROWSERBASE_API_KEY=your_api_key_here
BROWSERBASE_PROJECT_ID=your_project_id_here
```

**Available Browser Tools:**

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to a URL |
| `browser_snapshot` | Get text-based page snapshot with element refs |
| `browser_click` | Click an element by ref (e.g., `@e5`) |
| `browser_type` | Type text into an input field |
| `browser_scroll` | Scroll up or down |
| `browser_back` | Go back in browser history |
| `browser_press` | Press a keyboard key (Enter, Tab, etc.) |
| `browser_close` | Close the browser session |
| `browser_get_images` | Get list of images on the page |

**Example Usage:**
```bash
# Use browser tools with web search and vision
python run_agent.py \
  --query "Go to amazon.com and find the price of the latest Kindle" \
  --enabled_toolsets=browser,web,vision

# Use browser-focused distribution
python batch_runner.py \
  --dataset_file=browser_tasks.jsonl \
  --distribution=browser_use \
  --run_name=browser_run
```

See `.env.example` for all available configuration options including debug settings.

### Skills Tools

Skills are on-demand knowledge documents the agent can load when needed. They follow a **progressive disclosure** pattern to minimize token usage:

```
skills/
├── mlops/                    # Category folder
│   ├── axolotl/             # Skill folder
│   │   ├── SKILL.md         # Main instructions (required)
│   │   ├── references/      # Additional docs, API specs
│   │   └── templates/       # Output formats, configs
│   └── vllm/
│       └── SKILL.md
```

**Available Skills Tools:**

| Tool | Description |
|------|-------------|
| `skills_categories` | List available skill categories (~50 tokens) |
| `skills_list` | List skills with name + description (~3k tokens for 40 skills) |
| `skill_view` | Load full skill content, tags, and linked files |

**Example Usage:**
```bash
# Use skills tools
python run_agent.py \
  --query "What skills do you have for fine-tuning? Show me the axolotl skill." \
  --enabled_toolsets=skills
```

**Creating Skills:**

Skills use YAML frontmatter for metadata:
```yaml
---
name: my-skill
description: Brief description shown in skills_list
tags: [tag1, tag2]
related_skills: [other-skill]
version: 1.0.0
---
# Skill Content

Instructions, examples, and guidelines here...
```

Skills can include:
- `references/` - Additional documentation, API specs, examples
- `templates/` - Output formats, config files, boilerplate code
- `scripts/` - Executable helpers (Python, shell scripts)

## Interactive CLI

The CLI provides a rich interactive experience for working with the agent.

### Running the CLI

```bash
# Basic usage
./hermes

# With specific model
./hermes --model "anthropic/claude-sonnet-4"

# With specific toolsets
./hermes --toolsets "web,terminal,skills"
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List available tools by toolset |
| `/toolsets` | List available toolsets |
| `/model [name]` | Show or change the current model |
| `/prompt [text]` | View/set custom system prompt |
| `/personality [name]` | Set a predefined personality |
| `/clear` | Clear screen and reset conversation |
| `/reset` | Reset conversation only |
| `/history` | Show conversation history |
| `/save` | Save current conversation to file |
| `/config` | Show current configuration |
| `/quit` | Exit the CLI |

### Configuration

Copy `cli-config.yaml.example` to `cli-config.yaml` and customize:

```yaml
# Model settings
model:
  default: "anthropic/claude-sonnet-4"

# Terminal backend (local, docker, singularity, modal, or ssh)
terminal:
  env_type: "local"
  cwd: "."  # Use current directory

# Or use SSH for remote execution (keeps agent code isolated)
# terminal:
#   env_type: "ssh"
#   ssh_host: "my-server.example.com"
#   ssh_user: "myuser"
#   ssh_key: "~/.ssh/id_rsa"
#   cwd: "/home/myuser/project"

# Enable specific toolsets
toolsets:
  - all  # or: web, terminal, browser, vision, etc.

# Custom personalities (use with /personality command)
agent:
  personalities:
    helpful: "You are a helpful assistant."
    kawaii: "You are a kawaii assistant! Use cute expressions..."
```

### Personalities

Built-in personalities available via `/personality`:
- `helpful`, `concise`, `technical`, `creative`, `teacher`
- `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`
- `noir`, `uwu`, `philosopher`, `hype`

## Toolsets System

The agent uses a toolsets system for organizing and managing tools. All tools must be part of a toolset to be accessible - individual tool selection is not supported. This ensures consistent and logical grouping of capabilities.

### Key Concepts

- **Toolsets**: Logical groups of tools for specific use cases (e.g., "research", "development", "debugging")
- **Composition**: Toolsets can include other toolsets for powerful combinations
- **Custom Toolsets**: Create your own toolsets at runtime or by editing `toolsets.py`
- **Toolset-Only Access**: Tools are only accessible through toolsets, not individually

### Available Toolsets

See `toolsets.py` for the complete list of predefined toolsets including:
- Basic toolsets (web, terminal, vision, creative, reasoning)
- Composite toolsets (research, development, analysis, etc.)
- Scenario-specific toolsets (debugging, documentation, API testing, etc.)
- Special toolsets (safe mode without terminal, minimal, offline)

### Using Toolsets

```bash
# Use a predefined toolset
python run_agent.py --enabled_toolsets=research --query "Find latest AI papers"

# Combine multiple toolsets
python run_agent.py --enabled_toolsets=web,vision --query "Analyze this website"

# Enable all toolsets explicitly (same as omitting the flag)
python run_agent.py --enabled_toolsets=all --query "Do web research and run commands if helpful"

# Safe mode (no terminal access)
python run_agent.py --enabled_toolsets=safe --query "Help without running commands"

# List all available toolsets and tools
python run_agent.py --list_tools
```

See `toolsets.py` for the complete list of available toolsets and how to create custom ones.

## Basic Usage

### Default (all tools enabled)
```bash
# Uses OpenRouter by default - just set OPENROUTER_API_KEY in .env
python run_agent.py \
  --query "search up the latest docs on jit in python 3.13 and write me basic example that's not in their docs. profile its perf" \
  --max_turns 20 \
  --model anthropic/claude-sonnet-4-20250514
```

### With specific toolset
```bash
python run_agent.py \
  --query "Debug this Python error" \
  --enabled_toolsets=debugging \
  --model anthropic/claude-sonnet-4-20250514
```

### Python API
```python
from run_agent import AIAgent

# Uses OpenRouter by default (reads OPENROUTER_API_KEY from .env)
agent = AIAgent(
    model="anthropic/claude-sonnet-4-20250514",
    enabled_toolsets=["research"]
)
response = agent.chat("Find information about quantum computing")

# Create custom toolset at runtime
from toolsets import create_custom_toolset

create_custom_toolset(
    name="my_tools",
    description="My custom toolkit",
    tools=["web_search"],
    includes=["terminal", "vision"]
)

agent = AIAgent(enabled_toolsets=["my_tools"])
```

## Batch Processing

Process multiple prompts from a dataset in parallel with automatic checkpointing and statistics tracking:

```bash
# Basic batch processing
python batch_runner.py \
  --dataset_file=prompts.jsonl \
  --batch_size=20 \
  --run_name=my_run

# With specific distribution
python batch_runner.py \
  --dataset_file=prompts.jsonl \
  --batch_size=20 \
  --run_name=image_run \
  --distribution=image_gen \
  --num_workers=4
```

**Key Features:**
- Parallel processing with configurable workers
- Toolset distributions for varied data generation
- Automatic checkpointing and resume capability
- Combined output in `data/<run_name>/trajectories.jsonl`
- Tool usage statistics and success rates

Use `--list_distributions` to see available toolset distributions for varied data generation.

### Trajectory Compression

Post-process trajectories to fit within token budgets for training:

```bash
# Compress a directory of JSONL files
python trajectory_compressor.py --input=data/my_run

# Compress a single JSONL file
python trajectory_compressor.py --input=data/trajectories.jsonl

# Compress a 15% sample (useful for creating smaller training sets)
python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=15

# Custom output and token target
python trajectory_compressor.py \
  --input=data/trajectories.jsonl \
  --output=data/compressed.jsonl \
  --target_max_tokens=16000
```

**Features:**
- Protects first turns (system, human, first GPT response, first tool call)
- Protects last N turns (configurable)
- Summarizes middle turns using LLM to fit target token budget
- Supports both directory and single file input
- Optional random sampling with `--sample_percent`
- Configurable via `configs/trajectory_compression.yaml`

### Ephemeral System Prompts

The ephemeral system prompt feature allows you to guide the model's behavior during batch processing **without** saving that prompt to the training dataset trajectories. This is useful for:

- Guiding model behavior during data collection
- Adding task-specific instructions 
- Keeping saved trajectories clean and focused on tool-calling format

**Example:**
```bash
python batch_runner.py \
  --dataset_file=prompts.jsonl \
  --batch_size=10 \
  --run_name=my_run \
  --ephemeral_system_prompt="You are a helpful assistant focused on image generation."
```

The ephemeral prompt will influence the model's behavior during execution, but **only the standard tool-calling system prompt** will be saved in the trajectory files.

The ephemeral prompt influences model behavior during execution, but **only the standard tool-calling system prompt** is saved in trajectory files.

## Command Line Arguments

**Single Agent (`run_agent.py`):**
- `--query`: The question or task for the agent
- `--model`: Model to use (default: claude-opus-4-20250514)
- `--api_key`: API key for authentication
- `--base_url`: API endpoint URL
- `--max_turns`: Maximum number of tool-calling iterations
- `--enabled_toolsets`: Comma-separated list of toolsets to enable. Use `all` (or `*`) to enable everything. If omitted, all toolsets are enabled by default.
- `--disabled_toolsets`: Comma-separated list of toolsets to disable
- `--list_tools`: List all available toolsets and tools
- `--save_trajectories`: Save conversation trajectories to JSONL files

**Batch Processing (`batch_runner.py`):**
- `--dataset_file`: Path to JSONL file with prompts
- `--batch_size`: Number of prompts per batch
- `--run_name`: Name for this run (for output/checkpointing)
- `--distribution`: Toolset distribution to use (default: "default")
- `--num_workers`: Number of parallel workers (default: 4)
- `--resume`: Resume from checkpoint if interrupted
- `--ephemeral_system_prompt`: System prompt used during execution but NOT saved to trajectories
- `--list_distributions`: List available toolset distributions

## Environment Variables

All environment variables can be configured in the `.env` file (copy from `.env.example`).

**LLM Provider (OpenRouter):**
- `OPENROUTER_API_KEY`: Primary LLM access via OpenRouter (supports Claude, GPT-4, Gemini, etc.)
- `LLM_MODEL`: Default model (e.g., `anthropic/claude-sonnet-4`, `openai/gpt-4o`)

**Tool API Keys:**
- `FIRECRAWL_API_KEY`: Web tools (search, extract, crawl)
- `NOUS_API_KEY`: Vision and reasoning tools
- `FAL_KEY`: Image generation tools

**Terminal Tool Configuration (mini-swe-agent backend):**
- `TERMINAL_ENV`: Backend type - `local`, `docker`, `singularity`, `modal`, or `ssh` (default: `local`)
- `TERMINAL_DOCKER_IMAGE`: Docker image for docker backend (default: `python:3.11-slim`)
- `TERMINAL_SINGULARITY_IMAGE`: Singularity/Apptainer image (can be `docker://...` URL or local `.sif` path)
- `TERMINAL_TIMEOUT`: Command timeout in seconds (default: `60`)
- `TERMINAL_LIFETIME_SECONDS`: Cleanup inactive environments after this time (default: `300`)
- `TERMINAL_CWD`: Working directory inside containers (default: `/tmp`)
- `TERMINAL_SCRATCH_DIR`: Custom scratch directory for sandbox storage (optional, auto-detects `/scratch`)
- `SUDO_PASSWORD`: Enable sudo commands by piping password via `sudo -S` (works with all backends)

**SSH Backend Configuration (for remote execution):**
- `TERMINAL_SSH_HOST`: Remote server hostname or IP
- `TERMINAL_SSH_USER`: SSH username
- `TERMINAL_SSH_PORT`: SSH port (default: `22`)
- `TERMINAL_SSH_KEY`: Path to SSH private key (optional, uses ssh-agent if not set)

**Browser Tool Configuration (agent-browser + Browserbase):**
- `BROWSERBASE_API_KEY`: Browserbase API key for cloud browser execution
- `BROWSERBASE_PROJECT_ID`: Browserbase project ID
- `BROWSER_SESSION_TIMEOUT`: Session timeout in seconds (default: `300`)

**Legacy Hecate Terminal Backend (optional):**
- `MORPH_API_KEY`: For Hecate/MorphCloud terminal backend
- `HECATE_VM_LIFETIME_SECONDS`: VM lifetime (default: 300)
- `HECATE_DEFAULT_SNAPSHOT_ID`: Default snapshot (default: snapshot_p5294qxt)

**Debug Options:**
- `WEB_TOOLS_DEBUG`, `VISION_TOOLS_DEBUG`, `MOA_TOOLS_DEBUG`, `IMAGE_TOOLS_DEBUG`: Enable debug logging

## Key Files

| File | Purpose |
|------|---------|
| `hermes` | CLI launcher script (run with `./hermes`) |
| `cli.py` | Interactive CLI implementation |
| `cli-config.yaml` | CLI configuration (copy from `.example`) |
| `run_agent.py` | Main agent runner - single query execution |
| `batch_runner.py` | Parallel batch processing with checkpointing |
| `model_tools.py` | Core tool definitions and handlers |
| `toolsets.py` | Toolset definitions and composition |
| `toolset_distributions.py` | Probability distributions for data generation |
| `trajectory_compressor.py` | Post-process trajectories for training |
| `tools/` | Individual tool implementations |
| `tools/skills_tool.py` | Skills system with progressive disclosure |
| `skills/` | On-demand knowledge documents |
| `docs/` | Documentation |
| `configs/` | Example batch run scripts |
