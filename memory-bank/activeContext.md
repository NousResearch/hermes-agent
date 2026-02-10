# Active Context

## Current Focus
Consolidating the two Atropos environment systems and fixing tool calling to use proper OpenAI-spec approach instead of ICL.

## PR Feedback from Lead Dev (Feb 10, 2026)

The PR was rejected because our approach has three fundamental issues:

### Issue 1: ManagedServer doesn't pass `tools={}` to `apply_chat_template()`
- When using Phase 2 (VLLM/SGLang for RL training), `ManagedServer` needs to pass tools to `tokenizer.apply_chat_template(tools=...)` 
- This makes the system prompt include tool definitions the way models were trained to expect
- **Fix**: Atropos PR #366 adds `tool_call_parser` support to ManagedServer (branch: `tool_call_support`)

### Issue 2: ICL prompt vs proper tool calling
- Our code embeds tools as XML in the system prompt (`<tools>...</tools>`)
- Proper approach: pass `tools=` parameter in `chat_completion()` calls and let the tokenizer's chat template handle formatting
- All Hermes datasets train on the proper format, not ICL

### Issue 3: Only Hermes `<tool_call>` parser, no multi-model support
- Our code only handles Hermes-style `<tool_call>` XML parsing
- Proper approach: parser registry supporting 11+ model families (hermes, qwen, deepseek, llama, mistral, etc.)

## Architecture: What Exists Now (Two Parallel Systems)

### `environments/` (Teknium's proper approach) ✅ CORRECT
```
environments/
├── agent_loop.py              ← Uses tools= in chat_completion() (OpenAI spec)
├── hermes_base_env.py         ← Phase 1 (OpenAI) + Phase 2 (ManagedServer + parser)
├── tool_context.py            ← ToolContext for reward functions
├── tool_call_parsers/         ← 11 model parsers (hermes, qwen, deepseek, llama, etc.)
│   ├── __init__.py            ← Registry with get_parser(), register_parser()
│   ├── hermes_parser.py
│   ├── qwen_parser.py
│   ├── deepseek_v3_parser.py
│   ├── llama_parser.py
│   ├── mistral_parser.py
│   └── ... (11 total)
├── terminal_test_env.py       ← Working example: file creation tasks
├── hermes_swe_env.py          ← SWE environment
└── patches.py                 ← Async-safe monkey patches
```

**How it works correctly:**
1. `HermesAgentLoop.run()` passes `tools=self.tool_schemas` to `chat_completion()`
2. ManagedServer passes tools to `tokenizer.apply_chat_template(tools=...)`
3. Parser registry reconstructs `tool_calls` from raw model output
4. Tool execution uses hermes-agent's `handle_function_call()` from `model_tools.py`

### `atropos/` (Our sandbox-optimized code) - PARTIALLY REDUNDANT
```
atropos/
├── agent/atropos_agent.py     ← ICL-based agent (REDUNDANT with agent_loop.py)
├── envs/agent_env.py          ← Environment with sandbox backends (PARTIALLY REDUNDANT)
├── envs/swe_smith_oracle_env.py ← SWE env using sandbox (KEEP - port to new base)
├── backends/                  ← Sandbox backends (KEEP - valuable infrastructure)
│   ├── modal_backend.py       ← Modal sandbox pool
│   ├── nomad_backend.py       ← Nomad/Docker/Singularity
│   └── base.py                ← ToolBackend protocol
├── slots/                     ← Slot multiplexing (KEEP)
├── nomad/                     ← Nomad client (KEEP)
├── tools/                     ← Sandbox tool registry (PARTIALLY REDUNDANT)
└── sandbox_server.py          ← HTTP server in containers (KEEP)
```

## Plan: Consolidate into `environments/`

### What to KEEP from `atropos/`:
- `backends/` - Modal, Nomad, Singularity backends (valuable infrastructure for scale)
- `slots/` - Slot multiplexing
- `nomad/` - Nomad client
- `sandbox_server.py` - Container HTTP server
- `Dockerfile` - Sandbox container image

### What to REMOVE/REPLACE:
- `atropos/agent/atropos_agent.py` → replaced by `environments/agent_loop.py`
- `atropos/envs/agent_env.py` → functionality merged into `environments/hermes_base_env.py`
- `atropos/tools/` → replaced by `model_tools.py` + `tools/` (hermes-agent's standard tools)

### What to CREATE:
- `environments/gsm8k_agent_env.py` → GSM8k with tool calling, subclasses `HermesAgentBaseEnv`
- Update `environments/hermes_base_env.py` to optionally use sandbox backends (Nomad/Modal) for terminal isolation when needed for scale

### Steps:
1. Install atropos `tool_call_support` branch (PR #366)
2. Create `environments/gsm8k_agent_env.py` using `HermesAgentBaseEnv`
3. Port `swe_smith_oracle_env.py` to use `HermesAgentBaseEnv`
4. Make sandbox backends accessible from `HermesAgentBaseEnv` (terminal_backend config)
5. Remove redundant `atropos/agent/` and `atropos/envs/agent_env.py`
6. Clean up `atropos/tools/` (keep only sandbox-specific tools)
7. Update tinker-atropos gsm8k env to use proper base class
8. Test everything end-to-end

## Previous Completed Work
- Modal backend integration (Feb 8) - KEEP backends, update integration point
- Main branch merge (Feb 9) - completed
- Singularity/Apptainer (Feb 6) - KEEP
- Memory Bank initialized (Feb 5)
