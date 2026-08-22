# Canonical Checkout Sync Pattern

## When to use

After writing a durable artifact (architecture plan, schema, documentation, mission verdict) to a workspace clone that is NOT the canonical checkout. Use to ensure the deliverable survives workspace cleanup or branch deletion.

## The problem

The LAH Stack maintains two separate clones of `openclaw-runtime`:

| Clone | Path | Remote | Role |
|-------|------|--------|------|
| Workspace | `/home/deploy/openclaw-runtime` | `git@github.com:leanframeworklab/openclaw-runtime.git` | Execution workspace `openclaw-runtime-dev` |
| Canonical | `/home/deploy/lah-stack-repos/openclaw-runtime` | `git@github.com-lah-stack:leanframeworklab/openclaw-runtime.git` | Canonical authority per `repo_mappings.json` |

These are **different inode numbers** (separate filesystem entries) — not symlinks, not shared git objects. Writing to one does not affect the other.

## Detection

```bash
# Compare inodes to confirm separate clones
ws_inode=$(stat --format="%i" /home/deploy/openclaw-runtime 2>/dev/null)
ca_inode=$(stat --format="%i" /home/deploy/lah-stack-repos/openclaw-runtime 2>/dev/null)
if [ "$ws_inode" != "$ca_inode" ]; then
  echo "SEPARATE CLONES — sync needed"
fi

# Verify which remote each uses
cd /home/deploy/openclaw-runtime && git remote -v
cd /home/deploy/lah-stack-repos/openclaw-runtime && git remote -v
```

## Sync procedure

### Option A — Commit to workspace, push, pull in canonical

1. Commit on workspace branch
2. `git push origin <branch>`
3. In canonical checkout: `git fetch origin <branch> && git checkout <branch>`
4. Verify: `git log --oneline -1` matches workspace HEAD

### Option B — Copy file directly

```bash
src="/home/deploy/openclaw-runtime/lah-openclaw-mvp/docs/superpowers/plans/2026-07-29-CLOE_X402_DISCOVERY.md"
dst="/home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/docs/superpowers/plans/"
cp "$src" "$dst"
```

Then commit in canonical checkout on the appropriate branch.

## Verification

```bash
# SHA-256 must match
sha256sum /home/deploy/openclaw-runtime/.../file.md
sha256sum /home/deploy/lah-stack-repos/openclaw-runtime/.../file.md
diff <(cd /home/deploy/openclaw-runtime && git rev-parse HEAD) \
     <(cd /home/deploy/lah-stack-repos/openclaw-runtime && git rev-parse HEAD)
```

## Pitfalls

- **Remote mismatch**: The workspace clone uses `github.com` (no `-lah-stack`), the canonical uses `github.com-lah-stack`. They point to the same upstream repo but through different SSH host configurations. Pushing from workspace does NOT automatically sync the canonical checkout.
- **Different branches**: Workspace may be on a feature branch while canonical is on a different branch or detached HEAD. Sync to the same branch or copy directly.
- **Dirty workspace**: If the workspace has pre-existing uncommitted changes, use Option B (direct copy) to avoid committing unrelated changes alongside the deliverable.
- **Worktrees**: Check `git worktree list` — some clones may be worktrees of a common parent, which changes the sync strategy. If they share a `.git` directory, a commit in one is visible in the other.
- **SHA infinite loop in document provenance**: If the document being synced contains a "canonicalization commit: <SHA>" field referencing its own commit, every `git commit --amend` changes the SHA, creating a loop. Avoid embedding the commit SHA in documents that are part of that same commit. Use branch names or parent SHAs instead. If a self-referential SHA is mandatory, script a two-pass: commit → record SHA → patch → commit --amend.
