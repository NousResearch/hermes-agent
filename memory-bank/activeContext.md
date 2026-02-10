# Active Context

## Current Focus
Adding sandbox pool support directly to `HermesAgentBaseEnv` so that `tool_pool_mode=modal/nomad` works alongside the default terminal-tool approach.

## Implementation Plan (Feb 10, 2026)

### Goal
The command should work:
```bash
python environments/swe_smith_oracle_env.py process \
    --env.tool_pool_mode modal \
    --env.modal_image python:3.11
```

### Changes to `environments/hermes_base_env.py`:

**1. Add config fields to `HermesAgentEnvConfig`:**
- `tool_pool_mode: str = "default"` — "default" (terminal tool), "nomad", or "modal"
- Nomad fields: `nomad_address`, `sandbox_job_id`, `sandbox_image`, `slots_per_container`, etc.
- Modal fields: `modal_app_name`, `modal_image`, `modal_gpu`, `modal_slots_per_sandbox`, etc.
- Shared: `allow_network`, `require_sandbox`, `purge_job_on_start`, `purge_job_on_shutdown`

**2. Add methods to `HermesAgentBaseEnv`:**
- `_start_sandbox_backend()` / `_stop_sandbox_backend()` — lifecycle management
- `setup_trajectory_workspace(item, exec_tool, trajectory_id)` → optional hook (no-op default)
- `verify_and_score_trajectory(item, result, exec_tool)` → optional hook (calls compute_reward by default)

**3. Modify `collect_trajectory()`:**
- When `tool_pool_mode == "default"`: existing behavior (terminal tool handles isolation)
- When `tool_pool_mode in ("nomad", "modal")`: acquire slot → run agent with sandbox-backed tools → verify → release

**4. Port SWE env to `environments/`:**
- Move/rewrite `swe_smith_oracle_env.py` to subclass `HermesAgentBaseEnv`
- Override `setup_trajectory_workspace()` (git clone/worktree)
- Override `verify_and_score_trajectory()` (pytest verification)

### Key Imports
```python
from atropos.backends import create_tool_backend  # Nomad/Modal backends
from atropos.backends.base import ToolBackend
from atropos.slots.executor import ExecutionResult
```

### What's Already Working
- ✅ atroposlib with tool_call_support (ManagedServer has tool_call_parser)
- ✅ GSM8k agent env with HermesAgentBaseEnv (Phase 1 tested, process mode)
- ✅ mini-swe-agent installed (terminal tool available)
- ✅ Modal backend (tested, working with sandboxes)
- ✅ Nomad/Singularity backends (tested, working)
- ✅ Tool call parsers (11+ models)

### What Blocks
- Tinker billing (402 error) — can't test Phase 2 training yet
- No VLLM on this machine — can't test ManagedServer locally
