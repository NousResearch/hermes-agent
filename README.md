# Hermes Agent

An AI agent with advanced tool-calling capabilities, featuring a flexible toolsets system for organizing and managing tools.

## Features

- **Web Tools**: Search, extract content, and crawl websites
- **Terminal Tools**: Execute commands with interactive session support
- **Vision Tools**: Analyze images from URLs
- **Reasoning Tools**: Advanced multi-model reasoning (Mixture of Agents)
- **Creative Tools**: Generate images from text prompts
- **Toolsets System**: Organize tools into logical groups for different scenarios
- **Batch Processing**: Process datasets in parallel with checkpointing and statistics tracking

## Setup

### 1. Install Dependencies
```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install Hecate for terminal tools
git clone git@github.com:NousResearch/hecate.git
cd hecate
pip install -e .
cd ..
```

### 2. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
- `ANTHROPIC_API_KEY` - Main agent model (get at: https://console.anthropic.com/)
- `FIRECRAWL_API_KEY` - Web tools (get at: https://firecrawl.dev/)
- `NOUS_API_KEY` - Vision & reasoning tools (get at: https://inference-api.nousresearch.com/)
- `MORPH_API_KEY` - Terminal tools (get at: https://morph.so/)
- `FAL_KEY` - Image generation (get at: https://fal.ai/)
- `OPENAI_API_KEY` - Optional, for some Hecate features

See `.env.example` for all available configuration options including debug settings and terminal tool configuration.

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
python run_agent.py \
  --query "search up the latest docs on jit in python 3.13 and write me basic example that's not in their docs. profile its perf" \
  --max_turns 20 \
  --model claude-sonnet-4-20250514 \
  --base_url https://api.anthropic.com/v1/ \
  --api_key $ANTHROPIC_API_KEY
```

### With specific toolset
```bash
python run_agent.py \
  --query "Debug this Python error" \
  --enabled_toolsets=debugging \
  --model claude-sonnet-4-20250514 \
  --api_key $ANTHROPIC_API_KEY
```

### Python API
```python
from run_agent import AIAgent

# Use a specific toolset
agent = AIAgent(
    model="claude-opus-4-20250514",
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

## Command Line Arguments

- `--query`: The question or task for the agent
- `--model`: Model to use (default: claude-opus-4-20250514)
- `--api_key`: API key for authentication
- `--base_url`: API endpoint URL
- `--max_turns`: Maximum number of tool-calling iterations
- `--enabled_toolsets`: Comma-separated list of toolsets to enable. Use `all` (or `*`) to enable everything. If omitted, all toolsets are enabled by default.
- `--disabled_toolsets`: Comma-separated list of toolsets to disable
- `--list_tools`: List all available toolsets and tools
- `--save_trajectories`: Save conversation trajectories to JSONL files

## Environment Variables

All environment variables can be configured in the `.env` file (copy from `.env.example`).

**Core API Keys:**
- `ANTHROPIC_API_KEY`: Main agent model
- `FIRECRAWL_API_KEY`: Web tools (search, extract, crawl)
- `NOUS_API_KEY`: Vision and reasoning tools
- `MORPH_API_KEY`: Terminal tools
- `FAL_KEY`: Image generation tools
- `OPENAI_API_KEY`: Optional, for some Hecate features

**Configuration Options:**
- `HECATE_VM_LIFETIME_SECONDS`: VM lifetime (default: 300)
- `HECATE_DEFAULT_SNAPSHOT_ID`: Default snapshot (default: snapshot_p5294qxt)
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
