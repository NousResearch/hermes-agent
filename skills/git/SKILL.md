---
name: git-divergent-history-manual-merge
description: "When git merge refuses divergent histories: manually merge file-by-file using git show :2: and :3: stages, then git add to resolve"
tags: [git, merge, conflict-resolution]
---

# Git Divergent History Manual Merge

## Problem

`git merge` or `git merge --allow-unrelated-histories` refuses to merge two divergent branches where both modified the same files with structural differences (e.g., local V5 branch 347-line level.py vs origin/master 509-line level.py).

Git shows: `merge conflict` but `git mergetool` may fail, or files have conflicting implementations rather than just content.

## Solution: Manual File-by-File Merge

### Step 1: Check conflict state

```bash
git status
# Look for "both modified:" files
# "ours" = local/HEAD = :2 in git show
# "theirs" = origin/master = :3 in git show
```

### Step 2: Read both versions simultaneously

```bash
git show :2:<file> > /tmp/ours.py    # local/HEAD version
git show :3:<file> > /tmp/theirs.py  # origin/master version
```

Or use execute_code to read them programmatically:
```python
import subprocess
ours = subprocess.check_output(['git', 'show', ':2:<file>'], text=True)
theirs = subprocess.check_output(['git', 'show', ':3:<file>'], text=True)
```

### Step 3: Compare and decide per-section

Use execute_code to diff both versions and identify:
- Which changes from `ours` to keep
- Which changes from `theirs` to keep
- Which new code to add from each

### Step 4: Write merged file

Use write_file (not echo heredoc) to write the final merged content.

### Step 5: Mark conflict resolved

```bash
git add <file>
```

This removes conflict markers from index and clears MERGE_HEAD for that file.

Repeat for each conflicted file.

### Step 6: Verify and commit

```bash
git status  # should be clean (no MERGE_HEAD)
python3 -m py_compile <all_merged_files>
git commit -m "Merge description"
git push
```

## Key Insight

When git refuses to auto-merge structurally different files (different implementations, not just content), manual merge is faster than fighting with mergetool. The `:2:` and `:3:` stages give you both versions simultaneously — compare them in Python and write the merged result.

## When to Use This vs. mergetool

| Situation | Approach |
|----------|---------|
| Simple text conflicts (3-way merge) | mergetool / manual edit |
| Files modified by both branches with different implementations | Manual merge (this skill) |
| One branch added new file, other modified it | Manual merge |
| git merge --allow-unrelated-histories refuses | Manual merge |

## Gotcha

- `git checkout --theirs/--ours` replaces entire file — not useful when both versions have valid changes you need to combine
- After `git add <file>`, the conflict markers are gone from index but the file is staged — commit to finalize
- Always syntax check after merge: `python3 -m py_compile <files>`
