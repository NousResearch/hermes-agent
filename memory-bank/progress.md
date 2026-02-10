# Progress

## Current Sprint: Consolidate Environment Systems (Feb 10, 2026)

PR feedback from lead dev identified three fundamental issues with our approach:
1. Tool calling uses ICL (in-context learning) instead of proper `tools=` parameter
2. ManagedServer doesn't pass tools to `apply_chat_template()`
3. Only Hermes parser, no multi-model support

Teknium already built the correct approach in `environments/` directory. Our task is to consolidate.

### Status
- [ ] Install atropos `tool_call_support` branch (PR #366)
- [ ] Create `environments/gsm8k_agent_env.py` using `HermesAgentBaseEnv`
- [ ] Port SWE env to `HermesAgentBaseEnv`
- [ ] Make sandbox backends accessible from `HermesAgentBaseEnv`
- [ ] Remove redundant `atropos/agent/` and `atropos/envs/agent_env.py`
- [ ] Clean up redundant `atropos/tools/`
- [ ] Test end-to-end with Tinker

## Completed Features

### ✅ Modal Backend Integration (Feb 8, 2026)
- `ModalToolBackend` with slot-based multiplexing
- Multi-profile support (CPU, GPU, high-memory)
- Auto-scaling sandbox pool via Modal Sandboxes
- **Status: KEEP backends, but change integration point from atropos/envs/ to environments/**

### ✅ Main Branch Merge (Feb 9, 2026)
- Merged 22,560 lines, 79 files, 5 conflicts resolved
- New: hermes_cli/, file_operations, RL training tools, gateway, cron

### ✅ Tinker RL Training Setup (Feb 9, 2026)
- tinker 0.12.0 + tinker-atropos installed
- GSM8k agent env created (needs rewrite to use proper base class)
- Config for Qwen3-4B created
- Pipeline verified: Tinker API connection works, all imports pass
- **Blocked on billing** (Tinker 402 error - regional payment issue)

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
| Tool parsers | `environments/tool_call_parsers/` | 11+ model parsers |

### REMOVE (redundant with environments/):
| Component | Location | Replaced By |
|-----------|----------|-------------|
| ICL agent | `atropos/agent/atropos_agent.py` | `environments/agent_loop.py` |
| AgentEnv | `atropos/envs/agent_env.py` | `environments/hermes_base_env.py` |
| Tool registry | `atropos/tools/` | `model_tools.py` + `tools/` |
| GSM8k ICL env | `tinker-atropos/.../gsm8k_agent.py` | New proper version |

## Known Issues
- Tinker billing (402 error) - user's payment didn't process
- `bwrap_available: false` in Singularity containers
- atropos `tool_call_support` branch not yet installed (PR #366)

## Evolution of Decisions

### Agent Architecture
- **v1 (our branch)**: ICL-based agent with `<tool_call>` XML tags in system prompt
- **v2 (Teknium's)**: Proper OpenAI-spec tool calling with `tools=` parameter
- **Decision**: Adopt v2, consolidate into `environments/`, keep sandbox backends from v1

### Environment Organization
- **Before**: Two parallel systems (`atropos/envs/` and `environments/`)
- **After**: Single system in `environments/`, using `HermesAgentBaseEnv` as base class
- Sandbox backends remain in `atropos/backends/` but integrate via terminal backend config
