# CPA-002 Control Plane Alignment Rollback (proposal only)

## Scope
This document is a rollback plan only. It is not an execution plan for runtime change.

## Observed candidate context (do not mutate)
- Candidate branch: `runtime/hermes-orch-control-plane-alignment`
- Candidate parent/working head before rollback packaging: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
- Original live base anchor: `e9b8ae6be137abead6d19ed8a67c523f8c527096`
- No new runtime gateway/API process changes in this lane.

## Observed gateway/runtime state snapshot (as provided)
- default launcher PID: `11896`
- default python PID: `19416`
- builder-grok PID observed in prior lane: `23900` (absent); gateway_state was stale/draining
- Import routing for execution context points to `C:/Users/fallo/AppData/Local/hermes/hermes-agent` via `PYTHONPATH` / editable mapping
- `kanban.auto_decompose=false` for default and builder-grok scopes

## Rollback boundary and target
- Rollback boundary: return from candidate HEAD `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79` back to live base `e9b8ae6be137abead6d19ed8a67c523f8c527096`.
- Do not activate any candidate branch without revalidation.

## Supported stop/start outline (proposal)
1. Re-check whether a separate `builder-grok` gateway process is running.
   - If not running, do not run a stop step for it.
2. Stop the default gateway path first (`11896`) only if a live activation was made from this branch and only under authorized procedure.
3. Stop any known builder-grok gateway only if it is currently running and confirmed to be this branch’s process.
4. Perform rollback checkout:
   - `git checkout runtime/hermes-orch-control-plane-alignment`
   - `git reset --hard e9b8ae6be137abead6d19ed8a67c523f8c527096`
5. Confirm import and command wiring before any restart:
   - validate `PYTHONPATH` still resolves `hermes_cli` and `tools` from intended path
   - dry-run/inspection only (no runtime side effects)
6. Restart required services only after approval and only if rollback was authorized.

## Import verification (proposal)
- Validate import path before/after rollback with a non-invasive interpreter command.
- Confirm modules import and return expected version/commit context.

## Routing verification (proposal)
- Verify that routing still points to `C:/Users/fallo/AppData/Local/hermes/hermes-agent` path mapping for CLI/tests.
- Confirm no stale gateway_state references to `builder-grok` remain as active pointers.
- Confirm `kanban.auto_decompose` remains `false`.

## Abort conditions
- Abort if process ownership/PID checks do not map cleanly to expected launcher/python roles.
- Abort if builder-grok process is found active with unknown branch and active drain state indicates mixed routing.
- Abort if import routing verification resolves an unexpected code path.
- Abort if any rollback stop/start step reports active task execution.

## Do not execute
- Do not execute rollback in this task.
- Do not activate/stop/restart gateway in this task.
- Do not run full suite tests in this task.
