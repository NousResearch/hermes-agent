# Hermes Agent

AI Agent with advanced tool calling capabilities, real-time logging, and extensible toolsets.

## Features

- 🤖 **Multi-model Support**: Works with Claude, GPT-4, and other OpenAI-compatible models
- 🔧 **Rich Tool Library**: Web search, content extraction, vision analysis, terminal execution, and more
- 📊 **Real-time Logging**: WebSocket-based logging system for monitoring agent execution
- 🖥️ **Desktop UI**: Modern PySide6 frontend with real-time event streaming
- 🎯 **Flexible Toolsets**: Predefined toolset combinations for different use cases
- 💾 **Trajectory Saving**: Save conversation flows for training and analysis
- 🔄 **Auto-retry**: Built-in error handling and retry logic

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python run_agent.py \
  --enabled_toolsets web \
  --query "Search for the latest AI news"
```

### With Real-time Logging

```bash
# Terminal 1: Start API endpoint server
python api_endpoint/logging_server.py

# Terminal 2: Run agent
python run_agent.py \
  --enabled_toolsets web \
  --enable_websocket_logging \
  --query "Your question here"
```

### With Desktop UI (Recommended)

The easiest way to use Hermes Agent is through the desktop UI:

```bash
# One-command launch (starts server + UI)
cd ui && ./start_hermes_ui.sh

# Or manually:
# Terminal 1: Start server
python api_endpoint/logging_server.py

# Terminal 2: Start UI
python ui/hermes_ui.py
```

The UI provides:
- 🖱️ Point-and-click query submission
- 🎛️ Easy model and tool selection
- 📊 Real-time event visualization
- 🔄 Automatic WebSocket connection
- 📝 Session history

## Project Structure

```
Hermes-Agent/
├── run_agent.py              # Main agent runner
├── model_tools.py            # Tool definitions and handling
├── toolsets.py               # Predefined toolset combinations
├── requirements.txt          # Python dependencies
│
├── ui/                      # Desktop UI ⭐ NEW
│   ├── hermes_ui.py         # PySide6 desktop application
│   ├── start_hermes_ui.sh   # UI launcher script
│   └── test_ui_flow.py      # UI integration tests
│
├── tools/                    # Tool implementations
│   ├── web_tools.py         # Web search, extract, crawl
│   ├── vision_tools.py      # Image analysis
│   ├── terminal_tool.py     # Command execution
│   ├── image_generation_tool.py
│   └── ...
│
├── api_endpoint/            # FastAPI + WebSocket logging endpoint
│   ├── logging_server.py    # WebSocket server + Agent API ⭐ ENHANCED
│   ├── websocket_logger.py  # Client library
│   ├── README.md           # API endpoint docs
│   └── ...
│
├── logs/                    # Log files
│   └── realtime/           # WebSocket session logs
│
└── tests/                   # Test files
```

## Available Toolsets

### Basic Toolsets
- **web**: Web search, extract, and crawl
- **terminal**: Command execution
- **vision**: Image analysis
- **creative**: Image generation
- **reasoning**: Mixture of agents

### Composite Toolsets
- **research**: Web + vision tools
- **development**: Web + terminal + vision
- **analysis**: Web + vision + reasoning
- **full_stack**: All tools enabled

### Usage Examples

```bash
# Research with web and vision
python run_agent.py --enabled_toolsets research --query "..."

# Development with terminal access
python run_agent.py --enabled_toolsets development --query "..."

# Combine multiple toolsets
python run_agent.py --enabled_toolsets web,vision --query "..."
```

## Real-time Logging System

Monitor your agent's execution in real-time with the FastAPI WebSocket endpoint using a **persistent connection pool** architecture.

### Architecture

The logging system uses a **singleton WebSocket connection** that persists across multiple agent runs:
- ✅ **No timeouts** - connection stays alive indefinitely
- ✅ **No reconnection overhead** - connect once, reuse forever
- ✅ **Parallel execution** - multiple agents share one connection
- ✅ **Production-ready** - graceful shutdown with signal handlers

See [`api_endpoint/PERSISTENT_CONNECTION_GUIDE.md`](api_endpoint/PERSISTENT_CONNECTION_GUIDE.md) for technical details.

### Features
- Track all API calls and responses
- **Persistent connection** - one WebSocket for all sessions
- Monitor tool executions with parameters and timing
- Capture errors and completion status
- REST API for querying sessions
- Real-time WebSocket broadcasting

### Documentation
See [`api_endpoint/README.md`](api_endpoint/README.md) for complete documentation.

### Quick Start
```bash
# Start API endpoint server
python api_endpoint/logging_server.py

# Run agent with logging
python run_agent.py --enable_websocket_logging --query "..."

# View logs
curl http://localhost:8000/sessions
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
ANTHROPIC_API_KEY=your_key_here
FIRECRAWL_API_KEY=your_key_here
NOUS_API_KEY=your_key_here
FAL_KEY=your_key_here

# Optional
WEB_TOOLS_DEBUG=true  # Enable web tools debug logging
```

### Command-Line Options

```bash
python run_agent.py --help
```

Key options:
- `--query`: Your question/task
- `--model`: Model to use (default: claude-sonnet-4-5-20250929)
- `--enabled_toolsets`: Toolsets to enable
- `--max_turns`: Maximum conversation turns
- `--enable_websocket_logging`: Enable real-time logging
- `--verbose`: Verbose debug output
- `--save_trajectories`: Save conversation trajectories

## Parallel Execution

The persistent connection pool enables true parallel agent execution. Multiple agents can run simultaneously, all sharing the same WebSocket connection for logging.

### Test Parallel Execution

```bash
python test_parallel_execution.py
```

This script runs three tests:
1. **Sequential** - baseline (3 queries one after another)
2. **Parallel** - 3 queries simultaneously  
3. **High Concurrency** - 10 queries simultaneously

**Expected Results:**
- ⚡ ~3x speedup with parallel execution
- ✅ All queries logged to same connection
- ✅ No connection timeouts or errors

### Custom Parallel Code

```python
import asyncio
from run_agent import AIAgent

async def main():
    agent1 = AIAgent(enable_websocket_logging=True)
    agent2 = AIAgent(enable_websocket_logging=True)
    
    # Run in parallel - both use shared connection!
    results = await asyncio.gather(
        agent1.run_conversation("Query 1"),
        agent2.run_conversation("Query 2")
    )

asyncio.run(main())
```

## Examples

### Investment Research
```bash
python run_agent.py \
  --enabled_toolsets web \
  --query "Find publicly traded companies in renewable energy"
```

### Code Analysis
```bash
python run_agent.py \
  --enabled_toolsets development \
  --query "Analyze the codebase and suggest improvements"
```

### Image Analysis
```bash
python run_agent.py \
  --enabled_toolsets vision \
  --query "Analyze this chart and explain the trends"
```

## Development

### Adding New Tools

1. Create tool in `tools/` directory
2. Register in `model_tools.py`
3. Add to appropriate toolset in `toolsets.py`

### Running Tests

```bash
# Test web tools
python tests/test_web_tools.py

# Test API endpoint / logging
cd api_endpoint
./test_websocket_logging.sh
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For questions or issues:
1. Check documentation in `api_endpoint/`
2. Review example usage in this README
3. Open a GitHub issue

---

Built with ❤️ for advanced AI agent workflows
