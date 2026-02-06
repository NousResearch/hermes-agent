# Active Context

## Current Focus
Singularity/Apptainer integration for HPC environments has been **COMPLETED AND TESTED**.

## Recently Completed (Feb 6, 2026)

### Singularity/Apptainer Sandbox Integration - FULLY WORKING
Successfully adapted the Atropos implementation from Docker to Singularity/Apptainer for HPC clusters where Docker cannot run without sudo permissions.

**Files Modified:**
1. `atropos/nomad/client.py` - Added `driver` and `singularity_image` parameters to `create_sandbox_job()`; Fixed port detection to check both `DynamicPorts` and `ReservedPorts` in `get_job_allocations()`
2. `atropos/slots/pool.py` - Added `driver` and `singularity_image` to `SlotPoolConfig`
3. `atropos/backends/nomad_backend.py` - Added driver options to `NomadBackendConfig`
4. `atropos/envs/agent_env.py` - Added CLI arguments `--env.driver` and `--env.singularity_image` to `AgentEnvConfig`

**Files Created:**
1. `nomad-singularity.hcl` - Nomad config with raw_exec driver enabled
2. `atropos/atropos-sandbox.sif` - Singularity image (80MB) built from Docker image
3. `test_singularity_job.py` - Test script for Singularity integration

**Key Implementation Details:**
- Uses Nomad's `raw_exec` driver to run `apptainer` commands
- Shell wrapper (`/bin/sh -c`) ensures Nomad environment variables expand correctly
- Binds Nomad allocation directory to `/data` for workspace persistence
- Uses **static ports** (`ReservedPorts`) instead of dynamic ports since raw_exec runs directly on host
- `get_job_allocations()` now checks both `DynamicPorts` (Docker) and `ReservedPorts` (Singularity)

**Test Results (All Passing):**
- Health check: ✅ Server responding with 5 slots
- Bash execution: ✅ Commands execute inside Singularity container
- Write file: ✅ File written to slot workspace
- Read file: ✅ File read back successfully

## Usage

### For Docker (default):
```python
config = SlotPoolConfig(
    driver="docker",
    image="atropos-sandbox:local",
)
```

### For Singularity/Apptainer:
```python
config = SlotPoolConfig(
    driver="singularity",
    singularity_image="/path/to/atropos-sandbox.sif",
)
```

### Nomad Configuration:
```bash
# Start Nomad with Singularity support
nomad agent -dev -config=nomad-singularity.hcl
```

## Next Steps
- Deploy to HPC cluster for production testing
- Consider adding bubblewrap (bwrap) support inside Singularity for additional sandboxing
- Document HPC-specific deployment procedures in skills/mlops/
