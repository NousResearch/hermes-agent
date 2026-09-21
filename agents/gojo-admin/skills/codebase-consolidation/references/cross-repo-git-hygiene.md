# Cross-Repo Git Hygiene Scan

Fast audit of repo state across the whole system. Produces a clean/dirty report with upstream divergence.

## Trigger

- User asks "show me uncommitted changes everywhere"
- Pre-merge sanity check when updating multiple repos
- Periodic housekeeping for a system with many small repos

## One-liner sweep

```bash
find /home/kensei -name '.git' -type d 2>/dev/null \
  | grep -vE '(/node_modules/|/vendor/|/.hermes/cache|/\.cache/)' \
  | sort
```

## Classification rules

| State | Action |
|-------|--------|
| Clean, up to date | Mark ✅, no action |
| Clean, ahead of upstream | Push immediately |
| Clean, behind upstream | Fast-forward only — safe to fetch+merge if zero local changes |
| Dirty (uncommitted files) | Inspect — merge, discard, or commit depending on content |
| Dirty + ahead/behind | Commit first, then merge/rebase |
| Ahead + behind simultaneously | Divergent branch — needs rebase or merge, not fast-forward |

## Verification after any change

```bash
# For every repo touched
status=$(git -C $repo status --short)
upstream=$(git -C $repo rev-list --left-right --count HEAD...origin/main 2>/dev/null)
# $upstream must be "0\t0" and $status must be empty
```

## Common dirt categories to recognise

- **Mode-only diff** (`old mode 100644 / new mode 100755`) — safe to discard, no content change
- **Untracked screenshots/assets** — usually generated artefacts, add to `.gitignore` or commit if final
- **Plugin cache files** (`.claude/plugins/cache/.in_use/*`) — ephemeral, safe to ignore or add to global gitignore
- **`.bak` timestamped files** — migration leftovers, safe to delete

## Cron path safety after repo retirement

If a repo is retired and replaced with a symlink, resolve the full path for every script referenced by `hermes cron list`:

```bash
readlink -f ~/.hermes/scripts/some_script.sh
```
