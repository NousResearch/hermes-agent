# CPA-002 Control Plane Alignment Rollback (proposal only)

## Scope and identity snapshot
- Candidate branch: `runtime/hermes-orch-control-plane-alignment`
- live base anchor: `e9b8ae6be137abead6d19ed8a67c523f8c527096`
- accepted source-stack head: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
- pre-repair evidence commit: `80304e0aaae1d8d45d0bc27897d17d07132affeb`
- final repair commit: discoverable by `git log -1 --format=%H -- docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`
- Checker status currently: `FAIL_REPAIRABLE` (evidence repair lane only, not runtime approval).

## Purpose of rollback boundary
Rollback means returning observable execution to the already tracked live base state with no destructive branch rewrite and no claim of source behavior mutation by this repair.

Rollback boundary is **not** `git reset --hard` of the candidate branch.

## Mandatory pre-rollback checkpoint requirements (must complete before any action)
1. Re-check live tracked/untracked scope:
   - verify tracked repo clean and untracked set is constrained to evidence artifacts only, including:
     - `docs/evidence/HERMES-ORCH-001-REPORT.md`
     - `docs/evidence/HERMES-ORCH-FINISH-001-HOF-000-RECOVERY.md`
     - `docs/evidence/HERMES-ORCH-FINISH-001-RUNTIME-INVENTORY.md`
     - `docs/evidence/HERMES-ORCH-FINISH-001/`
   - if collisions exist, abort and resolve explicitly.
2. Create/verify a local immutable rollback reference at `e9b8ae6be137abead6d19ed8a67c523f8c527096` (for example a reflog-safe tag or branch ref) before any restore.
3. Snapshot live branch/HEAD/status before mutation:
   - `git rev-parse --abbrev-ref HEAD`
   - `git rev-parse HEAD`
   - `git status --porcelain=v1`
4. Capture process/runtime evidence for verification trail (read-only):
   - PIDs and argv/cwd/environment snapshot for processes `11896`, `19416`, and optional builder-grok
   - import-routing proof: `PYTHONPATH` and editable mapping
   - `gateway_state` files under `gateway_state.json`, plus lock/state markers
5. Backup exact config files and active Kanban DB(s) for restoration path (hash + ACL metadata):
   - candidate routing/config files
   - `kanban.db*` and related DB artifacts
   - compute/readability verification on each backup after write.
6. Confirm `PID 23900` remains absent before planning builder-grok handling.

## Supported rollback manager behavior (proposal only)
- Only use inspected manager commands after `hermes gateway --help` and ownership check confirms target scope:
  - `hermes gateway status --profile <profile>`
  - optional profile-qualified variants for stop/start
- Do not assume builder-grok exists; only stop/start if process ownership and profile map confirms it is live and active.
- If default launcher is running, `hermes gateway stop` must target the same ownership context.
- If no managed gateways are running and profile is known-stable, do not run stop.

## Rollback execution outline (if authorized)
1. After checks pass and explicit **Alignment/Restart Authorization Gate** is satisfied, stop managed gateways that are confirmed owned by this run context.
2. Verify `candidate-first PYTHONPATH` is not still active for live routing before restore.
3. Restore live branch/HEAD via **non-destructive** operation selected from state inspection (e.g. `git checkout <live-branch>` then equivalent merge/branch switch), not a forced reset to candidate state.
4. Re-apply only required tracked config and DB backups if they were changed during activation window.
5. Restore routing to live base and restart only the exact previously running gateway set after all backups restored.
6. Verify imports and commit context return to live base:
   - import command under live `PYTHONPATH` should resolve `hermes_cli`/`tools` from live source path
   - quick command-level import/version check only.
7. Reopen and validate `gateway_state` files for non-drifting stale/unknown states.

## Alignment/Restart authorization gate
Rollback proceeds only if all four conditions hold:
- Collision scan is clean.
- Immutable rollback ref exists and is readable.
- Backup creation for config/DB succeeds with checksum + ACL capture.
- `hermes gateway status` confirms ownership and scope for every target process.
- For each condition failure: abort.

## Abort conditions
- Process ownership/PID checks are ambiguous or mixed.
- Builder-grok process appears active but ownership cannot be proven and/or stale draining state cannot be resolved.
- Import routing proves a non-live path during planned rollback.
- Any backup cannot be read back or has missing/incorrect ACL attributes.
- `gateway_state` or process snapshots indicate unknown active task execution.
- Candidate test or proof gates unresolved (`FAIL_REPAIRABLE` state remains).

## Candidate-vs-live routing truth retained
- Live routing proof at `2026-07-17T15:37:23Z`:
  - PID `19416` `PYTHONPATH`: `C:/Users/fallo/AppData/Local/hermes/hermes-agent;C:/Users/fallo/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages`
  - cwd `C:/Users/fallo/AppData/Local/hermes`
  - PID `11896` is the launcher parent of `19416`
  - PID `23900` absent
- Candidate test/runtime operations must continue to use candidate-first PYTHONPATH:
  `C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment`.
- `default` and `builder-grok` `kanban.auto_decompose=false`.
- no matching new CPA/HKNIR/HOF-020A/HOF-020-R1/HOF-029 cards in all-board read-only title query.

## Do not execute in this task
- Do not execute rollback now.
- Do not activate/stop/start gateways now.
- Do not edit DB/config/source now.
- Do not run restart until checker clears `FAIL_REPAIRABLE` and explicit authorization is granted.
