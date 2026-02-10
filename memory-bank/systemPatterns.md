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
  - Server handles tool call parsing via `/v1/chat/completions`
- **Phase 2 (ManagedServer)**: Client-side tool call parser + logprob tracking
  - Required for: RL training (exact token IDs + logprobs for GRPO/PPO)
  - Uses `/generate` endpoint for raw token output
  - Parser registry selects per-model parser (hermes, qwen, llama, etc.)
  - **Verified working** with RunPod SGLang endpoint (Feb 10, 2026)

### Phase 2 Call Chain (Verified)
```
collect_trajectory()
  → ServerManager.managed_server(tokenizer, tool_call_parser)
    → ManagedServer(server=VLLMServer)
      → ManagedServer.chat_completion(messages, tools, n, max_tokens, temp)
        → _convert_messages_to_prompt(messages, tools=tools)  [apply_chat_template]
        → _compute_input_ids(prompt, extending_node)
        → VLLMServer.tokens_and_logprobs_completion(**kwargs)  [public method]
          → _tokens_and_logprobs_comp(stat_dict, **kwargs)     [retry decorator, semaphore]
            → _tokens_and_logprobs_completion_wrapper(**kwargs) [patched for SGLang]
              → aiohttp POST to /generate
              → Returns (prompt_tokens, [output_tokens], [output_logprobs], [finish_reasons])
        → _create_sequence_node(...)  [stores in current_nodes]
        → tool_call_parser.parse(completion_text)  [if parser configured]
        → Returns ChatCompletion with tool_calls
```

### SGLang Compatibility Patch (`environments/patches.py`)
VLLMServer's `_tokens_and_logprobs_completion_wrapper` is monkey-patched to handle SGLang's
different request/response format. Applied automatically at import time via `apply_patches()`.

```
SGLang request:  {"input_ids": [...], "sampling_params": {...}, "return_logprob": true}
SGLang response: {"meta_info": {"output_token_logprobs": [[logprob, token_id, text], ...]}}

VLLM request:   {"prompt": {"prompt_token_ids": [...]}, "logprobs": 0}
VLLM response:  {"logprobs": [[{token_id: logprob}]], "finish_reasons": [...]}
```

Also handles RunPod serverless double-JSON wrapping (response body wrapped in quotes).

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

Infrastructure for scaled sandbox execution, integrated into HermesAgentBaseEnv:

```
ToolBackend (Protocol)
    ├── NomadToolBackend → SlotPool → NomadClient + SandboxExecutor (HTTP)
    │   ├── Docker driver (default)
    │   └── Singularity driver (HPC)
    └── ModalToolBackend → _ModalSandboxPool → modal.Sandbox.exec() (direct)
        └── _ModalMultiProfileManager (multi-profile support)
```

Two execution modes in HermesAgentBaseEnv (controlled by `tool_pool_mode` config):
- `default` - Local tool execution via handle_function_call() + ToolContext
- `modal` / `nomad` - Sandbox routing: slot acquire → setup workspace → agent loop → verify → release

Sandbox routing architecture:
```
collect_trajectory()
    ├── tool_pool_mode="default" → _collect_trajectory_local()
    │   └── _run_agent_loop(tool_handler=None) → compute_reward(ctx)
    │
    └── tool_pool_mode="modal"/"nomad" → _collect_trajectory_sandbox()
        ├── backend.acquire(task_id) → Slot
        ├── exec_tool = backend.execute_batch wrapper → ExecutionResult
        ├── setup_trajectory_workspace(item, exec_tool) [subclass hook]
        ├── _run_agent_loop(tool_handler=sandbox_tool_handler)
        │   └── terminal → backend.execute_batch → JSON string
        │   └── other tools → handle_function_call (local)
        ├── verify_and_score_trajectory(item, result, exec_tool) [subclass hook]
        └── backend.release(slot, reset_workspace=True) [finally]
```

Key interfaces:
- `exec_tool(tool_name, args, timeout)` → `ExecutionResult` (for env hooks)
- `tool_handler(tool_name, args, task_id)` → JSON string (for agent loop)

### Training Pipeline (Tinker + Atropos)
```
Terminal 1: run-api (port 8000)              ← Atropos Rollout API
Terminal 2: launch_training.py (port 8001)   ← Tinker Trainer + inference
Terminal 3: environment.py serve             ← Environment (rollouts)
```
