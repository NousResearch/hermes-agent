# Active Context

## Current Task: SWE Smith Oracle Env with Modal Backend

### Goal
Run this command:
```bash
python environments/swe_smith_oracle_env.py process \
    --env.use_wandb false \
    --env.total_steps 2 \
    --env.group_size 1 \
    --env.max_items 2 \
    --env.tool_pool_mode modal \
    --env.modal_image python:3.11 \
    --env.modal_slots_per_sandbox 10 \
    --env.modal_min_sandboxes 1
```

### What's Done
1. ✅ **agent_loop.py** - Added `tool_handler` parameter
   - New param: `tool_handler=None` in `__init__`
   - When `self.tool_handler` is set, it's called INSTEAD of `handle_function_call()`
   - Signature: `async tool_handler(tool_name, args, task_id) -> str`
   - Shows `[sandbox]` instead of backend name in terminal preview

2. ✅ **Phase 2 ManagedServer + SGLang** - Fully working (previous session)

3. ✅ **hermes_base_env.py** - Sandbox routing in collect_trajectory() (THIS SESSION)
   - Refactored `collect_trajectory()` into:
     - `_use_sandbox_backend()` - checks if sandbox should be used
     - `_collect_trajectory_local()` - existing path (ToolContext + handle_function_call)
     - `_collect_trajectory_sandbox()` - NEW sandbox path with slot lifecycle
     - `_run_agent_loop()` - shared agent loop for Phase 1/2, accepts tool_handler
     - `_build_scored_item()` - shared scored item construction
   - Sandbox path:
     1. `backend.acquire(task_id)` → Slot
     2. `exec_tool` callable wrapping `backend.execute_batch([(slot, tool_name, args)])`
     3. `setup_trajectory_workspace(item, exec_tool=exec_tool)` → workspace_meta
     4. `sandbox_tool_handler` routes terminal→sandbox, other→local
     5. `_run_agent_loop(tool_handler=sandbox_tool_handler)`
     6. `verify_and_score_trajectory(item, result, exec_tool=exec_tool)`
     7. `backend.release(slot, reset_workspace=True)` in finally
   - Added `handle_function_call` import for non-terminal tool fallback

4. ✅ **swe_smith_oracle_env.py** - Sandbox hooks (THIS SESSION)
   - `setup_trajectory_workspace()` - bare repo cache + git worktree (ported from atropos/envs/swe_smith_oracle_env.py)
   - `verify_and_score_trajectory()` - install deps + run pytest in sandbox
   - `compute_reward()` retained for local (non-sandbox) path
   - Uses `exec_tool("bash", {"command": cmd}, timeout=600)` → `ExecutionResult`

5. ✅ **All tests pass**:
   - Syntax checks (ast.parse) on both files
   - Import checks (both modules import cleanly)
   - Method existence checks (all new methods present)
   - Signature checks (exec_tool, trajectory_id, workspace_meta params)
   - Backend integration (ModalSandboxConfig.from_agent_env_config, create_tool_backend)
   - `_use_sandbox_backend()` logic (True when modal+backend set, False otherwise)

6. ✅ **End-to-end test with Qwen 3 8B + Modal sandbox** (THIS SESSION)
   - RunPod endpoint: `0tx0ruuuo4f10c` (Qwen/Qwen3-8B via SGLang)
   - 5 terminal tool calls executed IN sandbox: `ls`, `git status`, `git log`, `cat parse.py`, `cat tests/`
   - In-sandbox verification: install deps + pytest → score=0.0 (model inspected but didn't fix)
   - Full token tracking with logprobs via Phase 2 ManagedServer
   - Key finding: Llama-3-8B template silently drops `tools=` param, Qwen 3 has full Hermes format support

### Current Task: Integrate Slot Pool Backend into tools/terminal_tool.py

#### Step 1: Add `_SlotPoolEnvironment` to `tools/terminal_tool.py`
- New class alongside existing `_LocalEnvironment`, `_DockerEnvironment`, etc.
- Routes through `atropos/backends/` (ModalToolBackend or NomadToolBackend)
- N:M slot multiplexing: 5-10 sandboxes × 10 slots each = 50-100 concurrent
- Singleton `_SlotPoolManager` (like `_ModalPoolManager`) manages backend lifecycle
- `execute()` acquires slot → `backend.execute_batch([(slot, "bash", ...)])` → returns `{"output": ..., "returncode": ...}`
- `cleanup()` releases slot back to pool

#### Step 2: Wire into `_create_environment()`
- `TERMINAL_ENV=slot_pool` → `_SlotPoolEnvironment(...)`
- Sub-config: `TERMINAL_SLOT_BACKEND=modal` or `TERMINAL_SLOT_BACKEND=nomad`
- Reuse existing `TERMINAL_MODAL_*` and Nomad env vars for configuration

#### Step 3: Remove redundant `atropos/tools/` files
- DELETE: `hermes_external_tools.py`, `build_registry.py`, `sandbox_stubs.py`, `toolset_resolver.py`
- KEEP: `base.py` (ToolCall/ToolResult types), `tool_executor.py` (batched queue), `terminal_stateful_tool.py`, `tmux_tool.py`

#### Step 4: Clean up `atropos/envs/` and `atropos/agent/` (defer)
- Remove `atropos/envs/agent_env.py` → replaced by `environments/hermes_base_env.py`
- Remove `atropos/agent/atropos_agent.py` → replaced by `environments/agent_loop.py`

#### Later
- Test with Tinker trainer (blocked on billing)
- Add more environments (endless-terminals, terminalbench 2)

### Key Architecture Insight
Two separate sandbox integration points:
1. **`tools/terminal_tool.py` with `TERMINAL_ENV=slot_pool`** — for hermes CLI, batch_runner, any code using `handle_function_call("terminal", ...)`. Uses `_SlotPoolEnvironment` which wraps `atropos/backends/`.
2. **`environments/hermes_base_env.py` with `tool_pool_mode=modal/nomad`** — for RL environments. Uses `_collect_trajectory_sandbox()` which directly acquires slots and creates `sandbox_tool_handler`.

Both use the same underlying `atropos/backends/` (ModalToolBackend, NomadToolBackend) with the same slot pool.

### Architecture Summary

```
environments/hermes_base_env.py  (HermesAgentBaseEnv)
    │
    ├── tool_pool_mode="default" (existing path)
    │   └── collect_trajectory() → HermesAgentLoop(tool_handler=None) 
    │       → handle_function_call() → hermes terminal tool (local)
    │
    └── tool_pool_mode="modal" or "nomad" (new path)
        └── collect_trajectory():
            1. slot = backend.acquire(task_id)
            2. exec_tool = lambda routing through backend.execute_batch
            3. setup_trajectory_workspace(item, exec_tool=exec_tool)  [subclass hook]
            4. HermesAgentLoop(tool_handler=sandbox_tool_handler)
               → terminal calls → backend.execute_batch(slot, "bash", ...)
            5. verify_and_score_trajectory(item, result, exec_tool=exec_tool) [subclass hook]
            6. backend.release(slot, reset_workspace=True)

atropos/backends/modal_backend.py  (ModalToolBackend)
    └── acquire(trajectory_id) → Slot
    └── execute_batch([(slot, "bash", {"command": "..."})])  → [ExecutionResult]
    └── release(slot, reset_workspace=True)
```

### Key Files to Modify
1. `environments/hermes_base_env.py` - Add sandbox path in `collect_trajectory()`
2. `environments/swe_smith_oracle_env.py` - Override `setup_trajectory_workspace()` and `verify_and_score_trajectory()` to use exec_tool

### Important Notes
- `exec_tool` returns `ExecutionResult` (from `atropos/slots/executor.py`) with `.success`, `.output`, `.error`, `.metadata`
- `tool_handler` returns JSON string (for agent loop message format)
- These are DIFFERENT interfaces for different purposes:
  - `exec_tool`: used by env hooks (setup/verify) - returns structured result
  - `tool_handler`: used by agent loop - returns JSON string like hermes tools do
- The ModalToolBackend.execute_batch calls _ModalSandboxWithSlots.execute which runs `sandbox.exec("bash", "-c", command)` on Modal
- For the SWE env, the worktree setup pattern from `atropos/envs/swe_smith_oracle_env.py` should be reused (bare repo cache + worktree add)
