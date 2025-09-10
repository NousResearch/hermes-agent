# Hermes Agent

An AI agent with advanced tool-calling capabilities, featuring a flexible toolsets system for organizing and managing tools.

## Features

- **Web Tools**: Search, extract content, and crawl websites
- **Terminal Tools**: Execute commands with interactive session support
- **Vision Tools**: Analyze images from URLs
- **Reasoning Tools**: Advanced multi-model reasoning (Mixture of Agents)
- **Creative Tools**: Generate images from text prompts
- **Toolsets System**: Organize tools into logical groups for different scenarios

## Setup
```bash
pip install -r requirements.txt
git clone git@github.com:NousResearch/hecate.git
cd hecate
pip install -e .
```

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

## Command Line Arguments

- `--query`: The question or task for the agent
- `--model`: Model to use (default: claude-opus-4-20250514)
- `--api_key`: API key for authentication
- `--base_url`: API endpoint URL
- `--max_turns`: Maximum number of tool-calling iterations
- `--enabled_toolsets`: Comma-separated list of toolsets to enable
- `--disabled_toolsets`: Comma-separated list of toolsets to disable
- `--list_tools`: List all available toolsets and tools
- `--save_trajectories`: Save conversation trajectories to JSONL files

## Environment Variables

Set these environment variables to enable different tools:

- `FIRECRAWL_API_KEY`: For web tools (search, extract, crawl)
- `MORPH_API_KEY`: For terminal tools
- `NOUS_API_KEY`: For vision and reasoning tools
- `FAL_KEY`: For image generation tools
- `ANTHROPIC_API_KEY`: For the main agent model

## Documentation

- `TOOLSETS_README.md`: Comprehensive guide to the toolsets system
- `toolsets.py`: View and modify available toolsets
- `model_tools.py`: Core tool definitions and handlers

## Examples

See `TOOLSETS_README.md` for extensive examples of using different toolsets for various scenarios.
