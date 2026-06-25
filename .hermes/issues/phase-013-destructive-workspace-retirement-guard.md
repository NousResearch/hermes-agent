# Phase 013: Destructive Workspace Retirement Guard

status: implemented-local-tested · awaiting owner commit + VPS deploy
priority: P0
owner: Hermes Agent
reported_at: 2026-06-07 Asia/Bangkok
resolved_at: 2026-06-21 Asia/Bangkok (via Use AI Relay · Codex coded · Opus reviewed)
project: EmailHunter

## Resolution (2026-06-21)

Implemented a narrow workspace-retirement guard in `tools/approval.py`:

- `detect_workspace_retirement_command()` + `WORKSPACE_RETIREMENT_PATTERNS`
  block whole project-root / worktree-root deletion: `git worktree remove`,
  recursive delete of any `/.worktree/` path, and recursive delete of a
  `/srv/projects/<name>` root (NOT subpaths).
- `_workspace_retirement_block_result()` returns a hardline-style block with
  a message: archive + restore-verify first, never auto-delete from
  "no leftovers" / "100%".
- Wired at BOTH `check_dangerous_command` call sites, right after the existing
  hardline floor (so `--yolo` cannot bypass it).
- Tests: `tests/tools/test_workspace_retirement_guard.py` (19 cases, 8 block /
  11 allow) — all pass; existing `test_command_guards.py` (19) still pass;
  destructive-confirm tests (14) still pass. No false positive on
  `rm -rf /tmp/x`, `node_modules`, project subpaths, `git worktree list/prune`.

Still pending (NOT done autonomously — needs owner):
- commit + merge (owner approval gate)
- deploy to the running VPS Hermes runtime at
  `/home/linux-nat/SynerryTools/hermes-agent/main`

## Incident

AI deleted the local EmailHunter workspace at:

`/Users/rattanasak/Documents/Viber Project/Tech Tools/EmailHunter`

The deletion happened during a local-to-VPS migration closeout after the owner asked for no localhost leftovers. The AI interpreted the instruction too literally and removed the local workspace after remote/VPS checks, instead of requiring a mandatory archive/restore checkpoint and explicit final deletion confirmation.

## Impact

- Local project folder was absent on 2026-06-07.
- Owner lost trust in the existing delete-prevention system.
- Recovery was possible because the AI had pushed a local retirement snapshot branch before deletion.

## Recovery Completed

- Restored the local folder from GitHub branch `codex/local-retirement-snapshot-20260606`.
- Restored local `.env` from VPS shared secret file without printing secret values.
- Added sanitized `vps-gitlab` remote URL with no embedded credential.
- VPS production remained healthy during recovery.

## Root Cause

Current Hermes closeout and safety rules say not to delete unrelated user changes, but they do not create a hard stop for whole-workspace deletion when the owner uses broad language like "no leftovers" or "100%". The system lacks a required destructive-operation gate for:

- `rm -rf` on a project root
- deleting a workspace/worktree directory
- deleting ignored local files such as `.env`
- deleting local dependency/runtime state without a restore plan

## Required Fix

Implement a mandatory destructive workspace guard:

1. Detect commands that delete project roots or worktree roots.
2. Require an archive path or cloneable restore source before deletion.
3. Require a restore drill command before deletion is allowed.
4. Require explicit owner confirmation that names the exact path to delete.
5. Record the deletion decision in `.hermes/issues` or handoff.
6. Never convert "no leftovers" into project-root deletion automatically.

## Verification Required

- Add tests or policy checks that fail if an AI plan includes project-root deletion without the guard fields.
- Add closeout wording that distinguishes "nonessential leftovers" from "required production assets".
- Add a recovery checklist to the local-to-VPS migration workflow.

