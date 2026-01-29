# Hermes Agent

An AI agent with advanced tool-calling capabilities, featuring a flexible toolsets system for organizing and managing tools.

## Features

- **Web Tools**: Search, extract content, and crawl websites
- **Terminal Tools**: Execute commands via mini-swe-agent (local, Docker, or Modal backends)
- **Browser Tools**: Automate web browsers to navigate, click, type, and extract content
- **Vision Tools**: Analyze images from URLs
- **Reasoning Tools**: Advanced multi-model reasoning (Mixture of Agents)
- **Creative Tools**: Generate images from text prompts
- **Toolsets System**: Organize tools into logical groups for different scenarios
- **Batch Processing**: Process datasets in parallel with checkpointing and statistics tracking
- **Ephemeral System Prompts**: Guide model behavior without polluting training datasets

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

# Install required packages
pip install -r requirements.txt

# Install mini-swe-agent for terminal tools
pip install -e ./mini-swe-agent
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

The terminal tool uses **mini-swe-agent** environments. Configure in `.env`:

```bash
# Backend: "local", "docker", "singularity", or "modal"
TERMINAL_ENV=local          # Default: runs on host machine (no isolation)
TERMINAL_ENV=singularity    # Recommended for HPC: Apptainer/Singularity containers
TERMINAL_ENV=docker         # Isolated Docker containers
TERMINAL_ENV=modal          # Cloud execution via Modal

# Container image (for docker/singularity/modal backends)
TERMINAL_DOCKER_IMAGE=python:3.11-slim
TERMINAL_SINGULARITY_IMAGE=docker://python:3.11-slim
TERMINAL_TIMEOUT=60
```

**Backend Requirements:**
- **local**: No extra setup (runs directly on your machine, no isolation)
- **singularity**: Requires Apptainer or Singularity installed (common on HPC clusters, no root needed)
- **docker**: Requires Docker installed and user in `docker` group
- **modal**: Requires Modal account (see setup below)

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

# 2. Install agent-browser CLI globally
npm install -g agent-browser

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

For detailed documentation on toolsets, see `TOOLSETS_README.md`.

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

**Quick Start:** See [QUICKSTART_BATCH.md](QUICKSTART_BATCH.md) for a 5-minute getting started guide.  
**Full Documentation:** See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for comprehensive documentation.

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

**Documentation:** See [docs/ephemeral_system_prompt.md](docs/ephemeral_system_prompt.md) for complete details.

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
- `TERMINAL_ENV`: Backend type - `local`, `docker`, or `modal` (default: `local`)
- `TERMINAL_DOCKER_IMAGE`: Docker image to use (default: `python:3.11-slim`)
- `TERMINAL_TIMEOUT`: Command timeout in seconds (default: `60`)
- `TERMINAL_LIFETIME_SECONDS`: Cleanup inactive environments after this time (default: `300`)
- `TERMINAL_CWD`: Working directory inside containers (default: `/tmp`)

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

## Documentation

**Single Agent Usage:**
- `TOOLSETS_README.md`: Comprehensive guide to the toolsets system
- `toolsets.py`: View and modify available toolsets
- `model_tools.py`: Core tool definitions and handlers

**Batch Processing:**
- `QUICKSTART_BATCH.md`: 5-minute quick start guide
- `BATCH_PROCESSING.md`: Complete batch processing documentation
- `toolset_distributions.py`: Toolset distributions for data generation

## Examples

See `TOOLSETS_README.md` for extensive examples of using different toolsets for various scenarios.
