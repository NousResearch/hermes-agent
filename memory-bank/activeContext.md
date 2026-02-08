# Active Context

## Current Focus
Modal backend integration has been **MERGED AND UPDATED** from the `modal-integration` branch.

## Recently Completed (Feb 8, 2026)

### Modal Backend Integration - MERGED & WORKING
Merged the `modal-integration` branch into `atropos-integrations` and fixed integration issues.

**What was merged (from another dev's branch):**
1. `atropos/backends/modal_backend.py` - Complete Modal backend with:
   - `ModalSandboxConfig` - Unified config with YAML profiles, env vars, and AgentEnv config loading
   - `_ModalSandboxWithSlots` - Modal Sandbox wrapper with slot-based multiplexing
   - `_ModalSandboxPool` - Auto-scaling pool of Modal sandboxes
   - `_ModalMultiProfileManager` - Multi-profile support (CPU, GPU, high-memory)
   - `ModalToolBackend` - Full ToolBackend implementation
2. `atropos/backends/__init__.py` - Updated `create_tool_backend()` to support `modal` mode
3. `tools/terminal_tool.py` - Native Modal Sandbox integration with:
   - `ModalProfile` config + YAML loading
   - `_ModalSandboxPool` (sync, thread-based for CLI use)
   - `_ModalPoolManager` (singleton, multi-profile)
   - `_ModalSandboxEnvironment` replacing old `_ModalEnvironment`
4. `docs/MODAL_BACKEND.md` - Comprehensive documentation
5. `modal_profiles.yaml.example` - Example profiles config
6. `tests/test_modal_integration.py` - Integration tests
7. `tests/test_modal_stress.py` - Stress tests
8. `tests/test_modal_terminal.py` - Terminal tool tests

**What I fixed after merge:**
1. `atropos/envs/agent_env.py` - Replaced old stub Modal fields with proper config fields matching `ModalSandboxConfig.from_agent_env_config()`:
   - `modal_image`, `modal_gpu`, `modal_cpu`, `modal_memory`
   - `modal_slots_per_sandbox`, `modal_min_sandboxes`, `modal_max_sandboxes`
   - `modal_idle_timeout`, `modal_max_lifetime`
   - `modal_acquire_timeout`, `modal_execution_timeout`
   - `modal_secrets`, `modal_env_vars`, `modal_workspace_base`
2. `atropos/backends/modal_backend.py` - Guarded `yaml` import with try/except

**Key Architecture Decisions:**
- Uses **Modal Sandboxes** (not Functions) - long-lived containers that stay hot
- Uses `sandbox.exec()` directly instead of HTTP/sandbox_server.py - simpler approach
- Slot-based multiplexing matching Nomad's pattern
- Multi-profile support for heterogeneous workloads (CPU vs GPU)
- Named sandbox recovery for resilience
- Modal SDK v1.3.2 compatible

## Previous Work (Feb 6, 2026)
### Singularity/Apptainer Sandbox Integration - FULLY WORKING
See progress.md for details.

## Usage

### Modal Backend (Atropos):
```bash
python -m atropos.envs.swe_smith_oracle_env process \
    --env.tool_pool_mode modal \
    --env.modal_image python:3.11 \
    --env.modal_slots_per_sandbox 10 \
    --env.modal_max_sandboxes 5
```

### Modal Terminal Tool (CLI):
```bash
export TERMINAL_ENV=modal
export TERMINAL_MODAL_IMAGE=python:3.11
./hermes
```

### With GPU Profile:
```bash
# In modal_profiles.yaml
profiles:
  pytorch-gpu:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    gpu: T4
    memory: 16384
```

## Next Steps
- Live test Modal backend with actual Modal credentials
- Test multi-profile GPU workflows
- Test sandbox recovery after restart
- Integrate with SWE-smith-oracle env for full GRPO training loop
