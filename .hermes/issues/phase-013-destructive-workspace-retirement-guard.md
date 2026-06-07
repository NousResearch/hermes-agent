# Phase 013: Destructive Workspace Retirement Guard

status: open
priority: P0
owner: Hermes Agent
reported_at: 2026-06-07 Asia/Bangkok
project: EmailHunter

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

