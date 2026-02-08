# Progress

## Completed Features

### ✅ Modal Backend Integration (Feb 8, 2026 - MERGED & TESTED)
Merged the `modal-integration` branch and fixed integration issues.

**What Works:**
- `ModalToolBackend` implements full `ToolBackend` interface (start, stop, acquire, release, execute_batch)
- Modal Sandboxes used for long-lived containers (not Functions)
- `sandbox.exec()` for direct command execution (no HTTP server needed)
- Slot-based multiplexing matching Nomad pattern
- Multi-profile support (`ModalSandboxConfig`, `_ModalMultiProfileManager`)
- YAML profile loading (`modal_profiles.yaml`)
- `AgentEnvConfig` fields for all Modal settings (`--env.modal_*`)
- `create_tool_backend()` supports `tool_pool_mode="modal"`
- Terminal tool (`tools/terminal_tool.py`) native Modal integration with pool management
- Named sandbox recovery via `Sandbox.from_name()`
- Auto-scaling sandbox pool per profile
- Artifact helpers (read, list, archive)

**CLI Usage:**
```bash
# Atropos backend
python -m atropos.envs.swe_smith_oracle_env process \
    --env.tool_pool_mode modal \
    --env.modal_image python:3.11

# Terminal tool
TERMINAL_ENV=modal ./hermes
```

**Files Modified/Created:**
- `atropos/backends/modal_backend.py` - Full implementation (~1200 lines)
- `atropos/backends/__init__.py` - `create_tool_backend()` updated
- `atropos/envs/agent_env.py` - 15 Modal config fields added
- `tools/terminal_tool.py` - Native Modal sandbox pool
- `docs/MODAL_BACKEND.md` - Documentation
- `modal_profiles.yaml.example` - Example profiles
- `tests/test_modal_integration.py` - Integration tests
- `tests/test_modal_stress.py` - Stress tests
- `tests/test_modal_terminal.py` - Terminal tool tests

### ✅ Singularity/Apptainer Sandbox Integration (Feb 6, 2026 - FULLY TESTED)
Adapted the Atropos sandbox environment from Docker to Singularity/Apptainer for HPC clusters.

**What Works:**
- `create_sandbox_job()` supports both `driver="docker"` and `driver="singularity"`
- SlotPoolConfig and NomadBackendConfig propagate driver settings
- Singularity container runs sandbox_server.py via Nomad's raw_exec driver
- All sandbox operations work: bash execution, file read/write
- **CLI arguments** `--env.driver` and `--env.singularity_image` for AgentEnvConfig
- **Static port binding** for Singularity (ReservedPorts vs DynamicPorts)

### ✅ Memory Bank Initialized (Feb 5, 2026)
Set up project documentation structure for context persistence.

## In Progress
None currently.

## Known Issues
- Modal backend not yet live-tested with actual Modal cloud credentials
- `bwrap_available: false` in Singularity containers
- Health check timing - may need longer wait for container startup on slower systems

## What's Left to Build

### Modal Backend
- [ ] Live test with Modal credentials on actual cloud
- [ ] Test multi-profile GPU workflows
- [ ] Test sandbox recovery after restart
- [ ] Integrate with SWE-smith-oracle env for GRPO training loop
- [ ] Performance benchmarking vs Nomad backend

### HPC Deployment
- [ ] Test on actual HPC cluster with Slurm/PBS integration
- [ ] Document cluster-specific deployment procedures

### Documentation
- [ ] Add Singularity deployment to README
- [ ] Create HPC deployment skill in skills/mlops/

## Evolution of Decisions

### Container Runtime Selection
- **Initial**: Docker-only via Nomad docker driver
- **Problem**: HPC clusters don't allow Docker without sudo
- **Solution**: Added Singularity/Apptainer support via raw_exec driver
- **Result**: Both runtimes now supported with same API

### Modal Backend Architecture
- **Initial**: Stub placeholder raising RuntimeError
- **Investigation**: Modal Sandboxes vs Functions - chose Sandboxes for long-lived containers
- **Design**: Direct `sandbox.exec()` instead of HTTP/sandbox_server.py (simpler, no networking needed)
- **Implementation**: Merged from `modal-integration` branch, fixed agent_env.py config fields
- **Result**: Three backends now supported: Nomad/Docker, Nomad/Singularity, Modal
