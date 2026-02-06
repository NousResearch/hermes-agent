# System Patterns: Hermes-Agent

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI (cli.py)                          │
│  - Rich welcome banner with caduceus                            │
│  - prompt_toolkit for input with history                        │
│  - Kawaii-style feedback and personalities                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AIAgent (run_agent.py)                      │
│  - Conversation loop with tool calling                          │
│  - KawaiiSpinner for animated feedback                          │
│  - Retry logic with exponential backoff                         │
│  - Session logging to logs/ directory                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Tool Routing (model_tools.py)                 │
│  - get_tool_definitions() - returns tools for API calls         │
│  - handle_function_call() - dispatches to tool handlers         │
│  - Toolset filtering (enabled/disabled)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │ Web Tools │     │ Terminal  │     │ Browser   │
    │ (Firecrawl)│    │ (mini-swe)│     │(agent-brw)│
    └───────────┘     └───────────┘     └───────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             ▼
                    ┌───────────────┐
                    │  Toolsets     │
                    │  (toolsets.py)│
                    │  Composition  │
                    └───────────────┘
```

## Key Design Patterns

### 1. Toolset Composition Pattern
Toolsets can include other toolsets, allowing flexible composition:

```python
TOOLSETS = {
    "web": {"tools": ["web_search", "web_extract"], "includes": []},
    "debugging": {"tools": ["terminal"], "includes": ["web"]},
    "full_stack": {"tools": [], "includes": ["web", "terminal", "vision", "browser"]}
}
```

Resolution is recursive with cycle detection.

### 2. Graceful Degradation Pattern
Each tool module has a `check_*_requirements()` function:
- Tools are only loaded if requirements are met
- Missing API keys disable tools, not crash the system
- Import errors are caught and tools marked unavailable

```python
try:
    from tools.web_tools import web_search_tool, check_firecrawl_api_key
except ModuleNotFoundError:
    web_search_tool = None
    def check_firecrawl_api_key(): return False
```

### 3. Session Isolation Pattern (task_id)
Stateful tools (terminal, browser) use `task_id` to isolate concurrent sessions:
- Each batch worker gets unique task_id
- VMs and browser sessions are tracked per task_id
- Cleanup functions release resources: `cleanup_vm(task_id)`, `cleanup_browser(task_id)`

### 4. Trajectory Format Pattern
Conversations are saved in ShareGPT format for training:

```json
{"from": "system", "value": "System prompt with <tools>...</tools>"}
{"from": "human", "value": "User message"}
{"from": "gpt", "value": "<think>reasoning</think>\n<tool_call>{...}</tool_call>"}
{"from": "tool", "value": "<tool_response>{...}</tool_response>"}
{"from": "gpt", "value": "Final response"}
```

### 5. Ephemeral System Prompt Pattern
Guide model behavior during data collection without saving to trajectories:
- `ephemeral_system_prompt` influences execution
- Only standard tool-calling system prompt saved to trajectories
- Keeps training data clean

### 6. Retry with Validation Pattern
The agent validates responses before accepting:
- Check tool names against `valid_tool_names` set
- Validate JSON arguments can be parsed
- Check for content after `<think>` blocks
- Roll back to last valid state on persistent failures

## Component Relationships

### AIAgent Class
- Central orchestrator for conversations
- Manages conversation history
- Calls OpenAI-compatible API
- Routes tool calls to handlers
- Provides animated feedback (KawaiiSpinner)

### Tool Modules (tools/*.py)
- Self-contained tool implementations
- Export: handler function + check function + schema
- Return JSON strings (never raw dicts)
- Accept optional `task_id` for stateful tools

### Toolsets System (toolsets.py)
- Defines logical groupings of tools
- Supports composition via `includes`
- `resolve_toolset()` recursively resolves all tools
- `validate_toolset()` checks if name is valid

### Model Tools (model_tools.py)
- Aggregates all tool definitions
- Routes function calls to correct handlers
- Filters tools based on enabled/disabled toolsets
- Bridge between agent and tool implementations

## Critical Implementation Paths

### Tool Execution Flow
1. AIAgent receives tool_calls from API response
2. Validates tool names against `valid_tool_names`
3. Validates JSON arguments can be parsed
4. Calls `handle_function_call()` with tool name, args, task_id
5. `handle_function_call()` routes to appropriate handler
6. Tool executes, returns JSON string
7. Result added to conversation as tool message
8. Loop continues until natural language response

### Configuration Loading Flow
1. `cli.py` calls `load_cli_config()`
2. Loads `cli-config.yaml`, merges with defaults
3. Sets environment variables for terminal config
4. `AIAgent` reads env vars when initializing terminal tool
5. Terminal tool creates appropriate backend based on `TERMINAL_ENV`
