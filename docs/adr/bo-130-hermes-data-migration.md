# BO-130 ADR — Heavy Hermes path migration to `/mnt/hermes-data`

## Status

Accepted as an implementation plan, not applied.

This ADR does **not** move data. It defines the migration order, safety gates, rollback plan, and path-class decisions for moving heavy Hermes/Kanban storage pressure away from the small root disk.

## Context

Current host facts from the BO-129/BO-130 read-only checks:

- `/` is critically pressured: about 48 GiB total, about 47 GiB used, about 1.2 GiB free, 98% used.
- `/mnt/hermes-data` is available and mounted from `/dev/sdb` as `ext4`, about 49 GiB total, about 44 GiB free.
- `/etc/fstab` contains a persistent mount entry for `/mnt/hermes-data` with `nofail`.
- `/mnt/hermes-data` ownership is `ubuntu:ubuntu`, mode `755`.
- Current largest recurring pressure classes are:
  - `~/.hermes/sessions`
  - `~/.hermes/hermes-agent/.worktrees`
  - `~/.hermes/kanban/workspaces`
  - `/tmp`
  - `~/.npm`
  - `~/.cache/uv`
- BO-126 through BO-129 already established a report-first cleanup lifecycle for Kanban workspaces. Current large Kanban workspaces are `blocked-active` or `approval-required`; cleanup candidates are currently zero.

The goal is to stop treating root disk pressure as a repeated emergency cleanup problem. Heavy, recurring, non-boot-critical Hermes storage should live on the larger data disk where safe, while root keeps only runtime-critical code/config and small state.

## Decision summary

| Path class | Decision | Reason |
| --- | --- | --- |
| `~/.hermes/kanban/workspaces` | Migrate first, after gateway/runtime quiesce | Large, recurring, task-owned workspaces; good fit for data disk; must preserve task workspace paths via symlink or config-backed path. |
| `~/.hermes/hermes-agent/.worktrees` | Migrate or recreate under data disk after active/dirty audit | Large implementation residue; not generic cache; requires git worktree registry and dirty-state checks. |
| `~/.npm`, `~/.cache/uv`, other package caches | Prefer cache env relocation after cleanup | Rebuild/download cost only; safe to redirect once tool env is configured consistently. |
| `~/.hermes/sessions` | Defer direct migration; design retention/archive first | It is durable conversation/tool history, not cache. Moving blindly creates a second-truth and recovery risk. |
| `/tmp` | Do not migrate; keep TTL janitor/reporting | `/tmp` semantics should stay OS-local. Use allowlisted inactive cleanup, not symlink migration. |
| Docker/containerd | Separate approval-gated lane | Requires container/runtime-specific proof; do not mix with Hermes path migration. |
| `~/.hermes/config.yaml`, memories, skills, credentials-adjacent state | Do not migrate in this ADR | Small but authority-critical. Keep on root unless a full Hermes-home migration is designed separately. |

Recommended migration strategy: **selective directory migration with rollbackable symlinks**, not whole-`~/.hermes` migration.

## Why not move all of `~/.hermes`?

Moving the entire Hermes home would combine unrelated risks:

- runtime config and memory authority,
- gateway and cron state,
- Kanban DB and workspaces,
- sessions and large tool outputs,
- repo checkout and worktrees,
- skills and operational scripts.

That is too much blast radius for a disk-pressure fix. The safer design is to migrate only the heavy variable-cost paths whose lifecycle and rollback can be checked independently.

## Proposed target layout

```text
/mnt/hermes-data/hermes/
  kanban-workspaces/       # target for ~/.hermes/kanban/workspaces
  hermes-agent-worktrees/  # target for ~/.hermes/hermes-agent/.worktrees
  caches/
    npm/
    uv/
  archives/
    sessions/             # optional future archive tier, not primary live sessions by default
```

Root-side compatibility paths should remain stable for existing code:

```text
~/.hermes/kanban/workspaces -> /mnt/hermes-data/hermes/kanban-workspaces
~/.hermes/hermes-agent/.worktrees -> /mnt/hermes-data/hermes/hermes-agent-worktrees
```

Package caches should prefer environment/config redirection over symlinks when practical:

```text
NPM_CONFIG_CACHE=/mnt/hermes-data/hermes/caches/npm
UV_CACHE_DIR=/mnt/hermes-data/hermes/caches/uv
```

Those env changes must be applied through the relevant Hermes/runtime service configuration, not ad-hoc shell state.

## Pre-apply gates

Before any migration apply, all of these must pass:

1. Explicit current-turn approval for the exact path class being moved.
2. Gateway/runtime quiesce plan approved separately if live Hermes processes depend on the path.
3. `df -hT / /mnt/hermes-data` confirms source pressure and target capacity.
4. `/mnt/hermes-data` mount is present and writable by `ubuntu`.
5. Source path exists and is not already a symlink to an unexpected target.
6. Active-reference check shows no process cwd or tmux pane under the source path.
7. For git worktrees, `git worktree list --porcelain` and per-worktree `git status --short` are captured.
8. Dirty/untracked worktrees are excluded unless they have explicit disposition.
9. A timestamped source manifest is written with path, size, inode type, owner/mode, and file count.
10. Rollback command is prepared before the move.

## Apply sketch for Kanban workspaces

This is a plan only. Do not run without separate approval.

```bash
set -euo pipefail
src="$HOME/.hermes/kanban/workspaces"
dst="/mnt/hermes-data/hermes/kanban-workspaces"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

# preflight: mount, capacity, active refs, source type, target absence
mkdir -p /mnt/hermes-data/hermes
rsync -aHAX --numeric-ids "$src/" "$dst/.staging-$stamp/"
mv "$src" "$src.pre-migration-$stamp"
mv "$dst/.staging-$stamp" "$dst"
ln -s "$dst" "$src"

# verify: symlink, task workspace discovery, BO-126/BO-129 report dry-run
python scripts/kanban_disk_pressure_report.py --json
```

Rollback sketch:

```bash
rm "$HOME/.hermes/kanban/workspaces"
mv "$HOME/.hermes/kanban/workspaces.pre-migration-$stamp" "$HOME/.hermes/kanban/workspaces"
rm -rf --one-file-system -- "/mnt/hermes-data/hermes/kanban-workspaces"
```

Do not delete the `.pre-migration-*` source until the gateway/runtime smoke and BO-129 report prove the new path works.

## Apply sketch for Hermes repo worktrees

This path is more dangerous than package caches because git worktree metadata can point to exact paths.

Preferred approach:

1. List worktrees and classify each as active, dirty, clean-removable, or preserve-required.
2. Remove clean stale worktrees through `git worktree remove` where possible.
3. For remaining long-lived worktrees, create new worktrees directly under `/mnt/hermes-data/hermes/hermes-agent-worktrees` from the canonical repo and retire old ones one by one.
4. Only use a directory-level symlink for `.worktrees` after proving no existing worktree registry entry becomes inconsistent.

Avoid bulk `mv ~/.hermes/hermes-agent/.worktrees` unless the git registry state is captured and rollback-tested.

## Apply sketch for package caches

Package caches are cheaper to move than workspaces:

1. Clear or shrink old root cache if approved.
2. Create target cache dirs under `/mnt/hermes-data/hermes/caches`.
3. Add env projection through Hermes/runtime service configuration:
   - `NPM_CONFIG_CACHE=/mnt/hermes-data/hermes/caches/npm`
   - `UV_CACHE_DIR=/mnt/hermes-data/hermes/caches/uv`
4. Verify with a small cache-producing command and check the files land under `/mnt/hermes-data`.

## Post-apply verification

For every applied path class:

- `df -hT / /mnt/hermes-data` shows root relief and target usage increase.
- BO-129 report still runs successfully.
- Kanban workspace classifier can read all existing workspace paths.
- No active process is using the old pre-migration path.
- Gateway/runtime smoke is separately approved and performed if the gateway was restarted or quiesced.
- Rollback source is retained until the next successful daily report.
- Kanban evidence records moved, retained, blocked-active, and approval-required residue separately.

## Residue policy

After migration:

- `*.pre-migration-*` directories are temporary rollback surfaces, not new archives.
- They must be removed or explicitly retained with owner, reason, and remove-by metadata after verification.
- Workspace tar backups are hardblock residue unless moved under `/mnt/hermes-data/hermes/archives` with retention metadata.

## Decision

Adopt selective migration:

1. **First migration candidate:** `~/.hermes/kanban/workspaces` to `/mnt/hermes-data/hermes/kanban-workspaces`, using a compatibility symlink and rollback source retention.
2. **Second candidate:** package caches via env/config redirection.
3. **Third candidate:** Hermes repo `.worktrees`, but only after git worktree registry cleanup and dirty-state disposition.
4. **Deferred:** live `~/.hermes/sessions` migration until a retention/archive design exists.
5. **Rejected for this ADR:** whole-`~/.hermes` move, `/tmp` migration, Docker/containerd migration.

## BO-130 boundary

This ADR is complete when the decision and migration plan are recorded. Actual migration, cleanup apply, gateway restart/reload, env/service mutation, and deletion remain separate approval gates.
