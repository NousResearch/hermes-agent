---
name: git-merged-branch-detection
description: Detect if a feature branch is already merged into target before attempting merge operations
---

# Git — Detect if Feature Branch is Already Merged

## Problem
When tasked with merging a feature branch (e.g., v39-persona-games), don't assume a complex merge is needed. The branch may already be merged into the target.

## Verification Commands (in order)

### 1. Check if file commits exist in target branch
```bash
git log origin/master -- src/components/Game/GameDialog.tsx
```
If the branch's commit hash appears, the file is already in master.

### 2. Check if branch is fully merged
```bash
git branch --contains origin/v39-persona-games
```
Shows all branches containing the given commit. If `master` appears, it's merged.

### 3. Test-merge without committing
```bash
git fetch origin
git merge --no-commit --no-ff origin/v39-persona-games
```
"Already up to date" = nothing to merge.

### 4. Find common ancestor
```bash
git merge-base origin/master origin/v39-persona-games
```
Compare with `git log origin/master --oneline -1` to see if master is ahead or behind.

## Key Pattern
Branches can exist independently yet their commits appear in master's linear history if:
- Repo was force-pushed/rebased after branch creation
- Branch commits were cherry-picked directly to master
- Branch was reset and recommitted

Always verify before starting a merge workflow.

## When to Use
- Before any branch merge operation
- When a branch exists but the feature seems to already work in target
- When `git diff --name-only origin/master..origin/branch` shows many files but merge seems simple
