# Repo Path Convention

**Rule:** All remote Git clones and working copies must live under `projects/github*` — nowhere else.

## Canonical locations

| Path | Purpose |
|------|---------|
| `projects/github-staging/` | In-progress repos (active work, PRs in flight) |
| `projects/github-staged/` | Completed repos (archived) |

## Detection

When assessing disk usage, identify repos living outside `projects/github*`:

```bash
# Find git repos in /root that should move to projects/github*
for dir in /root/*/; do
  if [ -d "$dir/.git" ]; then
    remote=$(git -C "$dir" remote get-url origin 2>/dev/null)
    if [ -n "$remote" ]; then
      echo "REPO OUTSIDE PROJECTS: $dir -> $remote"
    fi
  fi
done
```

## Resolution

1. Check if a copy already exists in `projects/github-staging/` or `projects/github-staged/`
2. If yes → confirm remotes match, then delete the `/root/` copy
3. If no → `git clone <url> projects/github-staging<reponame> && rm -rf /root/<reponame>`
4. Update any scripts/crons referencing the old path

## Pitfall

Genie's FILESYSTEM.md manifest can be stale — always run `du -sh` to verify actual sizes
before reporting reclaimable space to the user.

## Examples

- `indigo-repo`, `BOOK`, `get-md-work` found in `/root/` → deleted (copies already in `github-staging/`)
- `hermes-agent/`, `.rustup/`, `.linkedin-mcp/` → NOT git remotes (toolchains) — leave alone
