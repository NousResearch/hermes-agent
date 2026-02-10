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

## RL Training Architecture (Consolidated)

### Environment System (`environments/`)

The canonical way to build agentic RL environments in Hermes-Agent:

```
environments/
├── agent_loop.py              ← HermesAgentLoop: OpenAI-spec tool calling
├── hermes_base_env.py         ← HermesAgentBaseEnv: base class for all envs
├── tool_context.py            ← ToolContext: reward function tool access
├── tool_call_parsers/         ← 11+ model parsers (hermes, qwen, deepseek, etc.)
├── terminal_test_env.py       ← Example: file creation tasks
├── hermes_swe_env.py          ← SWE environment
└── gsm8k_agent_env.py         ← GSM8k with Python REPL (TODO)
```

### Two-Phase Operation
- **Phase 1 (OpenAI server)**: Native tool_calls from VLLM/SGLang/OpenRouter
  - Good for: SFT data gen, testing, evaluation
- **Phase 2 (ManagedServer)**: Client-side tool call parser + logprob tracking
  - Required for: RL training
  - Parser registry selects per-model parser (hermes, qwen, llama, etc.)

### Key Design: Proper Tool Calling (NOT ICL)
```python
# CORRECT: pass tools= to chat_completion()
response = await server.chat_completion(
    messages=messages,
    tools=tool_schemas,  # ← tokenizer.apply_chat_template(tools=...) formats these
    temperature=1.0,
)
# Response has response.choices[0].message.tool_calls (structured objects)

# WRONG (old approach): embed tools in system prompt as XML
system_prompt = f"<tools>{json.dumps(tools)}</tools>"  # ← ICL, not proper training format
```

### Sandbox Backends (`atropos/backends/`)

Infrastructure for scaled sandbox execution (separate from the env system):

```
ToolBackend (Protocol)
    ├── NomadToolBackend → SlotPool → NomadClient + SandboxExecutor (HTTP)
    │   ├── Docker driver (default)
    │   └── Singularity driver (HPC)
    └── ModalToolBackend → _ModalSandboxPool → modal.Sandbox.exec() (direct)
        └── _ModalMultiProfileManager (multi-profile support)
```

Accessed via `HermesAgentBaseEnv.terminal_backend` config option:
- `local` - Direct execution (default, development)
- `docker` - Docker containers
- `modal` - Modal cloud sandboxes (production RL)
- `singularity` - HPC clusters
- `ssh` - Remote server

### Training Pipeline (Tinker + Atropos)
```
Terminal 1: run-api (port 8000)              ← Atropos Rollout API
Terminal 2: launch_training.py (port 8001)   ← Tinker Trainer + inference
Terminal 3: environment.py serve             ← Environment (rollouts)
```
