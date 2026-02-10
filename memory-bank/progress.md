# Progress

## Current Sprint: Phase 2 ManagedServer + SGLang Working (Feb 10, 2026)

### ✅ Phase 2 End-to-End Pipeline VERIFIED
Full pipeline working: GSM8k env → collect_trajectory → ManagedServer → VLLMServer (SGLang patched) → tokens + logprobs + masks.

Test results:
- 212 tokens with logprobs and masks from single trajectory
- Reward: 1.0 (correct answer)
- ScoredDataItem has all required fields: tokens, masks, scores, advantages, ref_logprobs, messages
- RunPod SGLang endpoint (b9zmuyn1carwya) with Llama-3-8B-Instruct

### Consolidation Checklist
- [x] Install atropos `tool_call_support` branch (PR #366)
- [x] Create `environments/gsm8k_agent_env.py` using `HermesAgentBaseEnv`
- [x] Create `environments/agent_loop.py` with proper OpenAI-spec tool calling
- [x] Create `environments/tool_call_parsers/` with 13 parsers
- [x] Create `environments/patches.py` for SGLang compatibility
- [x] Add sandbox pool support to `HermesAgentBaseEnv`
- [x] Test Phase 1 (OpenAI server type) with Nous API — WORKS
- [x] Test Phase 2 (ManagedServer) with RunPod SGLang — WORKS
- [x] Port SWE env to `HermesAgentBaseEnv` with multiplexed sandboxing
- [ ] End-to-end test with Modal sandbox (needs live Modal)
- [ ] Remove redundant `atropos/agent/` and `atropos/envs/agent_env.py`
- [ ] Clean up redundant `atropos/tools/`
- [ ] Test end-to-end with Tinker trainer (blocked on billing)
- [ ] Test with actual tool calls (model producing tool_calls, not just text)

## Completed Features

### ✅ Phase 2 ManagedServer + SGLang (Feb 10, 2026)
- SGLang patch in `environments/patches.py` monkey-patches VLLMServer
- Handles SGLang's different request/response format vs VLLM
- Handles RunPod's double-JSON wrapping
- Full chain verified: ManagedServer → VLLMServer → _tokens_and_logprobs_comp (retry) → patched wrapper → /generate endpoint
- SequenceNode tracking: tokens, logprobs, masked_tokens all populated
- **Key discovery**: The AttributeError from earlier was NOT in our current code — likely from a prior code state

### ✅ Phase 1 OpenAI Server Mode (Feb 9-10, 2026)
- GSM8k env works with Nous API (OpenRouter-style endpoint)
- Terminal tool calls properly dispatched
- Tool call parsing handled natively by server (VLLM/SGLang /v1/chat/completions)
- Reward computation verified (math_verify for robust LaTeX comparison)

### ✅ Sandbox Pool Integration (Feb 10, 2026)
- Config fields added to `HermesAgentEnvConfig` for Nomad and Modal
- `_start_sandbox_backend()` / `_stop_sandbox_backend()` lifecycle methods
- Optional hooks: `setup_trajectory_workspace()`, `verify_and_score_trajectory()`
- Integrated into `env_manager()` and `process_manager()` cleanup

### ✅ Tool Call Parsers (Feb 9-10, 2026)
- 13 parsers: hermes, llama3_json, llama4_json, qwen, qwen3_coder, deepseek_v3, deepseek_v31, glm45, glm47, mistral, kimi_k2, longcat
- Registry pattern: `get_parser("hermes")` returns parser instance
- Each parser: `.parse(text) → (content, tool_calls)` 
- Used by ManagedServer in Phase 2 to extract structured tool_calls from raw completion

### ✅ Modal Backend Integration (Feb 8, 2026)
- `ModalToolBackend` with slot-based multiplexing
- Multi-profile support (CPU, GPU, high-memory)
- Auto-scaling sandbox pool via Modal Sandboxes

### ✅ Main Branch Merge (Feb 9, 2026)
- Merged 22,560 lines, 79 files, 5 conflicts resolved
- New: hermes_cli/, file_operations, RL training tools, gateway, cron

### ✅ Tinker RL Training Setup (Feb 9, 2026)
- tinker 0.12.0 + tinker-atropos installed
- GSM8k agent config created
- Pipeline verified: Tinker API connection works, all imports pass
- **Blocked on billing** (Tinker 402 error)

### ✅ Singularity/Apptainer Sandbox (Feb 6, 2026)
- Nomad raw_exec driver for HPC clusters
- All sandbox operations tested and working

### ✅ Memory Bank (Feb 5, 2026)
- Project documentation structure initialized

## What to KEEP vs REMOVE

### KEEP (valuable infrastructure):
| Component | Location | Purpose |
|-----------|----------|---------|
| Modal backend | `atropos/backends/modal_backend.py` | Cloud sandbox pool |
| Nomad backend | `atropos/backends/nomad_backend.py` | Docker/Singularity sandboxes |
| Slot pool | `atropos/slots/` | Container multiplexing |
| Nomad client | `atropos/nomad/` | Nomad API |
| Sandbox server | `atropos/sandbox_server.py` | HTTP server in containers |
| Dockerfile | `atropos/Dockerfile` | Container image |
| Agent loop | `environments/agent_loop.py` | Proper OpenAI-spec tool calling |
| Base env | `environments/hermes_base_env.py` | Phase 1/2 with parsers |
| Tool parsers | `environments/tool_call_parsers/` | 13 model parsers |
| SGLang patch | `environments/patches.py` | SGLang compatibility |

### REMOVE (redundant with environments/):
| Component | Location | Replaced By |
|-----------|----------|-------------|
| ICL agent | `atropos/agent/atropos_agent.py` | `environments/agent_loop.py` |
| AgentEnv | `atropos/envs/agent_env.py` | `environments/hermes_base_env.py` |
| Tool registry | `atropos/tools/` | `model_tools.py` + `tools/` |
| GSM8k ICL env | `tinker-atropos/.../gsm8k_agent.py` | `environments/gsm8k_agent_env.py` |

## Known Issues
- Tinker billing (402 error) - user's payment didn't process
- `bwrap_available: false` in Singularity containers
- Llama-3-8B-Instruct doesn't reliably produce tool calls via Phase 2 (needs Hermes-format model)
- Model answered GSM8k correctly but didn't actually USE the terminal tool (computed mentally)

## Evolution of Decisions

### Agent Architecture
- **v1 (our branch)**: ICL-based agent with `<tool_call>` XML tags in system prompt
- **v2 (Teknium's)**: Proper OpenAI-spec tool calling with `tools=` parameter
- **Decision**: Adopt v2, consolidate into `environments/`, keep sandbox backends from v1

### Environment Organization
- **Before**: Two parallel systems (`atropos/envs/` and `environments/`)
- **After**: Single system in `environments/`, using `HermesAgentBaseEnv` as base class
- Sandbox backends remain in `atropos/backends/` but integrate via terminal backend config

### Phase 2 SGLang Support
- **Problem**: VLLMServer hardcoded for VLLM's /generate format, SGLang is different
- **Solution**: Monkey-patch `_tokens_and_logprobs_completion_wrapper` in `environments/patches.py`
- **Applied**: Automatically at import time via `apply_patches()` in `hermes_base_env.py`
- **Handles**: SGLang format differences AND RunPod's double-JSON wrapping
