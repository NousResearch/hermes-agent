# Local Delta

This branch carries local Hermes customizations that are intentionally kept on top of upstream `origin/main`.

## Purpose
- Preserve the local meta-router + MR-ALS runtime integration.
- Preserve the OMX coding-harness integration used for routed `code` tasks.
- Make future upstream updates repeatable without losing local custom work.

## Primary custom files
- `run_agent.py`
- `gateway/meta_router_runtime.py`
- `gateway/omx_executor.py`
- `tests/cli/test_run_agent_omx_handoff.py`
- `tests/gateway/test_meta_router_memory_plan.py`
- `tests/gateway/test_omx_executor.py`

## What the local delta does
### 1. Meta-router / MR-ALS runtime wiring
- Restores the pre-classify + Phase 1 setup block in `run_agent.py`.
- Restores the MR-ALS post-turn Phase 2 / outcome-only block in `run_agent.py`.
- Keeps the shared runtime bridge in `gateway/meta_router_runtime.py`.

### 2. OMX handoff path
- Keeps the `run_agent.py` OMX shortcut for routed `code` tasks.
- Keeps `gateway/omx_executor.py` as the Hermes -> OMX execution bridge.
- Forces the explicit OMX binary path through Hermes config and command resolution.
- Adds explicit `codex exec -C <workdir>` so the launched Codex process uses the intended working root.
- Preserves timeout-salvage behavior when OMX already wrote execution artifacts.
- Uses `--ignore-user-config` on noninteractive OMX exec so unrelated user config/hooks do not destabilize the coding lane.

### 3. Verification coverage
- `tests/cli/test_run_agent_omx_handoff.py`
- `tests/gateway/test_meta_router_memory_plan.py`
- `tests/gateway/test_omx_executor.py`

## Update rules
- Never update Hermes from a dirty worktree.
- Always create a backup branch before rebasing.
- Use a temporary worktree for the rebase rehearsal.
- Keep upstream-only web/UI changes on upstream unless they are required by local runtime work.
- Prefer additive seam files over broad edits in upstream-owned hot files.

## Expected conflict hotspots
- `run_agent.py`
- occasionally `gateway/status.py`
- occasionally tests that assert exact routing / handoff semantics

## Required post-update verification
Run at minimum:
```bash
cd /home/samade10/.hermes/hermes-agent
/home/samade10/.hermes/venv/bin/python -m pytest -q -o addopts=''   tests/cli/test_run_agent_omx_handoff.py   tests/gateway/test_omx_executor.py   tests/gateway/test_meta_router_execution_result.py   tests/gateway/test_meta_router_memory_plan.py
```

Then verify runtime:
```bash
/home/samade10/.npm-global/bin/omx doctor
cd /home/samade10/.openclaw/workspace
python3 skills/maintainer/meta-router/scripts/model_learner.py
python3 skills/maintainer/meta-router/scripts/skill_learner.py
python3 skills/maintainer/meta-router/scripts/plugin_sync.py --report
```
