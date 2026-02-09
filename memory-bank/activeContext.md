# Active Context

## Current Focus
Tinker RL training integration - pipeline fully wired up, waiting on Tinker billing to test.

## Recently Completed (Feb 9, 2026)

### Tinker RL Training Integration
Created a complete agent training pipeline using Tinker (Thinking Machines) + Atropos:

**New Files Created:**
1. `tinker-atropos/tinker_atropos/environments/gsm8k_agent.py` - Agent GSM8k environment with:
   - Python REPL tool calling (Hermes-style `<tool_call>` format)
   - Multi-step agent loop within `collect_trajectories()`
   - Math answer verification via `math_verify`
   - Subprocess-based Python execution
   - WandB metrics (percent_correct, tool_use_rate)
2. `tinker-atropos/configs/gsm8k_agent.yaml` - Config for Qwen3-4B-Instruct training

**Dependencies Updated:**
- `pyproject.toml` `[atropos]` extra now includes: tinker SDK, torch, wandb, math-verify
- Installed: tinker 0.12.0, tinker-atropos 0.1.0, torch (CPU)

**README Updated:**
- Added comprehensive "RL Training with Tinker" section with architecture diagram, quick start, config docs
- Added TINKER_API_KEY and WANDB_API_KEY to optional keys table

**Verified Working:**
- Tinker SDK connection ✅
- All imports (tinker, tinker_atropos, trainer, environment) ✅
- Python REPL execution + tool call parsing ✅
- Math verification ✅
- Atropos run-api (port 8000) ✅
- Tinker trainer starts, loads config, creates inference server (port 8001) ✅

**Blocked:** Tinker billing (402 error) - user's payment didn't process (possibly regional card issue)

### Main Branch Merge (Feb 9, 2026)
Merged `origin/main` into `atropos-integrations` - 22,560 lines, 79 files, 5 conflicts resolved.

### Modal Backend (Feb 8, 2026)
Merged modal-integration branch, working with Modal Sandboxes.

### Singularity/Apptainer (Feb 6, 2026)
Completed and tested.

## Architecture: Training Pipeline

```
Terminal 1: run-api (port 8000) - Atropos Rollout API
Terminal 2: launch_training.py (port 8001) - Tinker Trainer + FastAPI inference
Terminal 3: gsm8k_agent.py serve - Environment (generates trajectories)
```

The agent env gets math problems → model calls Python REPL tool → scores answer → sends to Atropos → Tinker does LoRA training → updates sampling weights → repeat.

## Next Steps
- [ ] Resolve Tinker billing to test full training loop
- [ ] Run GSM8k agent training for ~20 steps (proof of concept)
- [ ] Monitor WandB for reward improvement
- [ ] Graduate to more complex agent envs (SWE tasks with Modal backend)
