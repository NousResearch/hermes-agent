# Progress

## Completed Features

### ✅ Singularity/Apptainer Sandbox Integration (Feb 6, 2026 - FULLY TESTED)
Adapted the Atropos sandbox environment from Docker to Singularity/Apptainer for HPC clusters.

**What Works:**
- `create_sandbox_job()` supports both `driver="docker"` and `driver="singularity"`
- SlotPoolConfig and NomadBackendConfig propagate driver settings
- Singularity container runs sandbox_server.py via Nomad's raw_exec driver
- All sandbox operations work: bash execution, file read/write
- Nomad environment variables properly expanded via shell wrapper
- **CLI arguments** `--env.driver` and `--env.singularity_image` for AgentEnvConfig
- **Static port binding** for Singularity (ReservedPorts vs DynamicPorts)
- **Port detection** works for both Docker and Singularity allocations

**CLI Usage:**
```bash
python -m atropos.envs.swe_smith_oracle_env process \
    --env.driver singularity \
    --env.singularity_image /path/to/atropos-sandbox.sif
```

**Created Files:**
- `nomad-singularity.hcl` - Nomad config with raw_exec enabled
- `atropos/atropos-sandbox.sif` - 80MB Singularity image
- `test_singularity_job.py` - Integration test script

**Modified Files:**
- `atropos/nomad/client.py` - driver support + ReservedPorts detection
- `atropos/slots/pool.py` - driver config fields
- `atropos/backends/nomad_backend.py` - driver config fields
- `atropos/envs/agent_env.py` - CLI arguments for driver selection

### ✅ Memory Bank Initialized (Feb 5, 2026)
Set up project documentation structure for context persistence.

## In Progress
None currently.

## Known Issues
- `bwrap_available: false` in Singularity containers - bubblewrap sandboxing not available inside the container (kernel namespaces already in use)
- Health check timing - may need longer wait for container startup on slower systems

## What's Left to Build

### HPC Deployment
- [ ] Test on actual HPC cluster with Slurm/PBS integration
- [ ] Document cluster-specific deployment procedures
- [ ] Add support for shared filesystem workspace binding

### Enhanced Sandboxing
- [ ] Investigate alternative sandboxing inside Singularity (seccomp, etc.)
- [ ] Add network isolation options for Singularity

### Documentation
- [ ] Add Singularity deployment to README
- [ ] Create HPC deployment skill in skills/mlops/

## Evolution of Decisions

### Container Runtime Selection
- **Initial**: Docker-only via Nomad docker driver
- **Problem**: HPC clusters don't allow Docker without sudo
- **Solution**: Added Singularity/Apptainer support via raw_exec driver
- **Result**: Both runtimes now supported with same API
