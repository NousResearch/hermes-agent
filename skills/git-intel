---
name: git-intel
description: >
  Six tools for passive git repository intelligence — repo summary, commit
  history, diff statistics, contributor leaderboard, file history with blame,
  and branch comparison. Pure Python stdlib, no API keys, no external
  dependencies. Works on any local git repository.
version: 1.0.0
metadata:
  hermes:
    tags: [git, devops, code-analysis, contributors, diff]
    category: devops
triggers:
  - user asks about a git repository (commits, history, contributors, branches)
  - user wants to analyze a codebase or understand who wrote what
  - user asks "who contributed most to this repo"
  - user wants to compare two branches or see what changed between commits
  - user asks about a specific file's history or who last modified it
  - user mentions "git log", "git blame", "git diff", "git shortlog"
  - user wants to understand the scope of a pull request or release
---

# Git Intelligence Toolset

Six passive analysis tools for any local git repository.
Pure Python stdlib — no pip installs, no API keys, zero external dependencies.

---

## Tools

### `git_repo_summary`
High-level overview of a repository.

**Returns:** current branch, all branches, remotes, latest commit, total
commit count, contributor count, tracked file count, repo size (KB),
tag count, dirty working tree flag.

**Example prompts:**
- "Give me an overview of this repo"
- "What's the status of the project in /home/user/myproject?"
- "How many commits and contributors does this repo have?"

---

### `git_log`
Commit history with filters.

**Parameters:**
- `limit` — max commits (1–500, default 20)
- `author` — filter by name/email substring
- `since` — e.g. `"2024-01-01"` or `"2 weeks ago"`
- `until` — upper date bound
- `path_filter` — only commits touching this file/dir
- `branch` — which ref to log (default HEAD)
- `no_merges` — exclude merge commits (default true)

**Example prompts:**
- "Show me the last 10 commits"
- "What did alice@example.com commit in the last month?"
- "Show commits that touched src/auth.py"

---

### `git_diff_stats`
Line-level diff statistics between any two refs.

**Parameters:**
- `base` — base ref (default `HEAD~1`)
- `target` — target ref (default `HEAD`)
- `path_filter` — limit to file/dir

**Returns:** files changed, total insertions, total deletions, net change,
per-file breakdown sorted by impact.

**Example prompts:**
- "What changed in the last commit?"
- "How many lines changed between v1.0 and v2.0?"
- "What's the diff between main and feature/auth?"

---

### `git_contributors`
Contributor leaderboard with commit counts and line statistics.

**Parameters:**
- `limit` — max contributors (default 20)
- `since` — e.g. `"6 months ago"` for recent activity
- `branch` — which branch to analyze

**Returns:** contributors sorted by commits (desc), with lines added/removed
and contribution percentage for the top 10.

**Example prompts:**
- "Who are the top contributors to this repo?"
- "Who has been most active in the last 3 months?"
- "Show me the contributor leaderboard"

---

### `git_file_history`
Full history and blame summary for a specific file.

**Parameters:**
- `file_path` — path to file (required)
- `limit` — max commits (default 30)
- `show_blame_summary` — include lines-per-author breakdown (default true)

**Returns:** all commits that touched the file, author breakdown,
first and latest commits, blame summary.

**Example prompts:**
- "Show me the history of src/main.py"
- "Who wrote most of the code in utils/auth.js?"
- "When was this file last modified and by whom?"

---

### `git_branch_compare`
Compare two branches or refs.

**Parameters:**
- `base` — base branch (default `main`)
- `target` — branch to compare (default `HEAD`)

**Returns:** ahead/behind counts, unique commit lists for each side,
common ancestor, full diff statistics.

**Example prompts:**
- "How does feature/payments compare to main?"
- "What commits are in my branch that aren't in main yet?"
- "Is this branch up to date with master?"

---

## Zero Dependencies

Uses only Python stdlib:
- `subprocess` — runs git commands
- `os` — path resolution and .git directory detection
- `collections` — aggregation helpers
- `datetime` — timestamp handling

Automatically walks up the directory tree to find the `.git` folder —
no need to be in the repo root.

---

## Pitfalls

- All tools require `git` to be installed and on PATH
- `git_contributors` with `show_blame_summary=true` is slow on large files
  (runs `git blame` which reads every line)
- `git_diff_stats` between distant refs (e.g. v1.0 vs HEAD) can be slow
  on very large repos
- Binary files show as 0 insertions/deletions in numstat output — this is
  expected git behavior
- `git_branch_compare` requires both refs to exist locally; remote-only
  branches must be fetched first

---

## Verification

```bash
# Quick smoke test — run from inside any git repo:
python3 -c "
from tools.git_intel_tool import git_repo_summary, git_log, git_contributors
import json
print(json.dumps(git_repo_summary('.'), indent=2))
"
```
