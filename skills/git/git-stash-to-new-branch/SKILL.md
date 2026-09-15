---
name: git-stash-to-new-branch
description: Move uncommitted work between branches without destructive git operations (branch -D, reset --hard, rebase). Use git stash + new branch + stash pop instead.
---
# Git Stash to New Branch — Safe Cross-Branch Work Migration

When you need to move uncommitted work from one branch to another (rebase without destructive commands):

## The Pattern

```bash
# 1. Stash current work (from old/branch)
git stash

# 2. Check out the TARGET base branch
git checkout target-branch

# 3. Create NEW branch from target
git checkout -b new-branch-name

# 4. Pop stash — auto-merges onto new base
git stash pop
```

## Why This Works

- `git stash` safely saves work as a patch stack
- Creating a fresh branch from the target avoids `git rebase` (can create divergent history)
- No `branch -D`, `reset --hard`, or `rebase` needed
- `stash pop` does a three-way merge, showing real conflicts

## Handling Stash Pop Conflicts

When `stash pop` produces merge conflicts:

1. **Resolve each conflicted file** using `patch` tool (targeted replace)
2. **For Sidebar-type files** (complex imports): manually merge by reading conflict markers
3. **For locale JSON files**: use `git checkout --theirs` for quick take-them-all, then re-patch missing pieces
4. After resolving: `git add <resolved-files>` and `git commit`

## Common Conflict Patterns

### Import conflicts (keep both):
```
<<<<<< Updated upstream
import { useMemoryStore } from '../../stores/memoryStore';
=======
import { PersonaSelector } from '../Persona/PersonaSelector';
>>>>>>> Stashed changes
```
→ Replace with both imports (no `<<<<<<` markers)

### Locale files: if V20+V21 both modified, `git checkout --theirs` + manually add missing V20 blocks

### Block conflicts (keep both features):
```
<<<<<< Updated upstream
{/* V20 feature block */}
=======
{/* V21 feature block */}
>>>>>>> Stashed changes
```
→ Replace with both blocks concatenated (both features coexist)

## When to Use This

- Subagent delivered on wrong branch (like v18-base instead of v20-base)
- Need to rebase work without destructive git operations
- User has denied `branch -D`, `reset --hard`, `rebase` before
- Working with WSL network issues where complex git ops timeout

## Verification After Pop

```bash
git log --oneline -3  # confirm on right base
git status --short     # confirm all files staged
git diff --cached --stat  # confirm change size reasonable
```

## Example: Moving V21 Work from V18-base to V20-base

```bash
git stash                          # save V21 work
git checkout v20-emotion-tracking  # target base (V20)
git checkout -b v21-persona-isolation  # new clean branch
git stash pop                      # auto-merge V21 onto V20
# resolve conflicts → add → commit → push
```
