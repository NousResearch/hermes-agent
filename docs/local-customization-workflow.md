# Local Customization Workflow

This repo now treats upstream tracking and Henry-specific customizations as separate concerns.

## Branch roles

- `main` — local mirror of `origin/main`; do not treat this as a scratch branch.
- `henry/patches` — durable local patch stack for Henry-specific behavior that is not upstream yet.
- `sync/<timestamp>` — disposable integration branch created during update/replay runs.
- `rescue/<timestamp>` — emergency snapshot branch for dirty work before sync.

## Golden rules

- Never develop directly on a dirty `main`.
- Keep local customizations as small, named commits.
- Upstream first when possible. Local patch stacks should stay small.
- Use `scripts/sync-local-mods.sh`, not raw `git pull`, when you want an upstream refresh that preserves local mods.
- If a change matters, it must exist as a real commit on `henry/patches` or as an upstream PR.

## Normal update flow

1. Make sure your current work is committed on `henry/patches`.
2. Fetch upstream and refresh the clean base via `scripts/sync-local-mods.sh`.
3. Let the script create a disposable sync worktree.
4. Review any replay conflicts in that disposable worktree.
5. Run smoke tests before promoting the refreshed patch stack.

## Add a new local customization

1. Start from `henry/patches` or a worktree based on it.
2. Make one focused change.
3. Run the narrowest useful tests.
4. Commit with a message that explains why the change remains local.
5. If the change belongs upstream, open a PR and plan to drop it from the local patch stack later.

## Override / escape hatches

- Direct pushes to `main` are blocked by `.githooks/pre-push` unless you intentionally set `HERMES_ALLOW_MAIN_PUSH=1`.
- Dirty trees are expected to be rejected or snapshotted before sync.
- `scripts/sync-local-mods.sh --help` documents the available flags.

## Recovery flow if sync fails

1. Inspect the sync worktree printed by the script.
2. Resolve replay conflicts there, not on `main`.
3. If needed, inspect the most recent `rescue/<timestamp>` branch for preserved pre-sync work.
4. If the replay attempt is junk, delete the disposable sync worktree/branch and rerun after cleaning up the patch stack.

## Practical intent

The goal is simple: upstream updates should be boring, and local behavior should be explicit enough that future-you can understand what happened without digging through a swamp of uncommitted drift.
