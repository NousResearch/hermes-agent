---
name: git-workflow
description: "Branch, commit, rebase, and clean history before a PR."
version: 0.2.0
author: Burak Koç (@HeLLGURD), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Git, Workflow, Branching, Rebase, Merge, Commit, Version-Control]
    category: software-development
    requires_toolsets: [terminal, file]
    related_skills: [github-code-review, github-pr-workflow]
---

# Git Workflow Skill

Drive everyday local Git — branching, staging, committing, rebasing, conflict
resolution, stash handling, and history cleanup before a pull request. Works
with any branching strategy (GitFlow, trunk-based, GitHub Flow) and needs no
external service. It stops at the repo boundary: opening, reviewing, or merging
PRs on a hosting provider belongs to the `github-pr-workflow` and
`github-code-review` skills.

## When to Use

- User asks for help with any local git operation
- A branch is behind main and needs syncing or rebasing
- Messy WIP commits need squashing into logical units before review
- A merge or rebase conflict needs resolving
- Something went wrong (bad commit, lost work) and needs undoing safely

Do NOT use this for:

- Opening, reviewing, or merging PRs on GitHub — use `github-pr-workflow` for
  the PR lifecycle or `github-code-review` for diffs and inline comments
- Initializing a brand-new repo — just run `git init` and answer inline

## Prerequisites

- `git` on PATH, with the working directory inside a git repository.
- For remote operations: an SSH key or HTTPS credential already configured.
- No environment variables and no extra dependencies.

## How to Run

Run every git command through the `terminal` tool from the repository root.
When a conflict or commit split needs a file edited, open it with `read_file`
and apply changes with `patch` — do not hand-write a parser. To scan a diff for
debug artifacts, use `search_files` rather than piping to a shell utility. This
is a procedural skill: there is nothing to install and no script to run.

## Quick Reference

| Task | Command |
|---|---|
| New branch | `git checkout -b feat/my-feature` |
| Stage interactively | `git add -p` |
| Commit | `git commit -m "feat: description"` |
| Sync with main | `git fetch origin && git rebase origin/main` |
| Squash before PR | `git rebase -i origin/main` |
| Undo last commit, keep work | `git reset --soft HEAD~1` |
| Stash WIP | `git stash push -m "description"` |
| Cherry-pick | `git cherry-pick <hash>` |
| Force-push safely | `git push --force-with-lease` |
| Recover lost work | `git reflog` |

## Procedure

### 1 — Start a feature branch

Always branch from the latest main:

```bash
git fetch origin
git checkout main
git pull origin main
git checkout -b feat/my-feature
```

Branch naming (suggest based on task type):

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/short-description` | `feat/dark-mode` |
| Bug fix | `fix/short-description` | `fix/login-crash` |
| Hotfix | `hotfix/short-description` | `hotfix/null-pointer` |
| Refactor | `refactor/short-description` | `refactor/auth-module` |
| Docs | `docs/short-description` | `docs/api-reference` |
| Chore | `chore/short-description` | `chore/update-deps` |

### 2 — Stage and commit

Show the diff before staging, then stage interactively so unrelated changes do
not sneak in:

```bash
git status
git diff
git add -p
git commit -m "feat(auth): add OAuth2 PKCE flow support"
```

Commit message rules (Conventional Commits):

- Format: `type(scope): short description` (scope optional)
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`
- Subject ≤ 72 chars, imperative mood ("add" not "added")
- Append `!` for breaking changes: `feat!: drop Python 3.9 support`
- Optional body wraps at 72 chars and explains *why*, not *what*

### 3 — Sync with main (rebase)

Rebase keeps feature-branch history linear:

```bash
git fetch origin
git rebase origin/main
```

If conflicts arise, resolve them one file at a time. Open each conflicted file
with `read_file`, edit it with `patch` to remove every conflict marker, then:

```bash
git diff --diff-filter=U   # list conflicted files
git add <resolved-file>
git rebase --continue      # or: git rebase --abort to bail out
```

Conflict markers look like this — keep one side, or combine both, and delete
all markers so the file is valid afterwards:

```
<<<<<<< HEAD (your branch)
your changes
=======
incoming changes from main
>>>>>>> origin/main
```

### 4 — Clean up commits before a PR

```bash
git log origin/main..HEAD --oneline   # how far ahead you are
git rebase -i origin/main
```

In the interactive list: `pick` keeps the commit, `reword` edits its message,
`squash` merges into the previous commit (combining messages), `fixup` merges
and discards the message, `drop` deletes it. A common squash:

```
pick a1b2c3 feat: initial implementation
fixup d4e5f6 wip
fixup g7h8i9 fix typo
```

The result is one clean commit that keeps the first message.

### 5 — Undo safely

| Situation | Command | Notes |
|---|---|---|
| Undo last commit, keep changes staged | `git reset --soft HEAD~1` | Safest undo |
| Undo last commit, keep changes unstaged | `git reset HEAD~1` | Work stays in the tree |
| Undo last commit, discard changes | `git reset --hard HEAD~1` | Destructive — preview with `git status` / `git diff`, then confirm first |
| Undo a pushed commit | `git revert <hash>` | New commit, safe on shared branches |
| Unstage a file | `git restore --staged <file>` | |
| Discard working-tree changes | `git restore <file>` | Destructive — confirm first |
| Recover a deleted branch | `git reflog` then `git checkout -b <name> <hash>` | |

Before ANY destructive command, preview what would be lost and get an explicit
yes from the user (see Pitfalls):

```bash
git status          # untracked and modified files
git diff HEAD       # everything that would be discarded
git stash list      # make sure WIP is not only in a stash
```

### 6 — Stash management

```bash
git stash push -m "WIP: half-done auth refactor"
git stash list
git stash apply             # apply, keep it in the list
git stash pop               # apply and remove
git stash show -p stash@{0} # inspect a stash
git stash drop stash@{0}    # delete a stash
```

### 7 — Cherry-pick

```bash
git log --oneline other-branch
git cherry-pick <hash>
git cherry-pick a1b2c3..d4e5f6      # a range (exclusive..inclusive)
git cherry-pick --no-commit <hash>  # stage only
```

On conflict: resolve, `git add`, then `git cherry-pick --continue`.

### 8 — Push safely

```bash
git push -u origin feat/my-feature   # first push (set upstream)
git push                             # later pushes
git push --force-with-lease          # after a rebase — NOT --force
```

`--force-with-lease` fails if someone else pushed since your last fetch, so it
cannot silently clobber their work. Never force-push to `main` or `master`.

### 9 — Prepare for a PR

```bash
git fetch origin && git rebase origin/main   # 1. get current
git diff origin/main..HEAD                    # 2. review your own diff
git log origin/main..HEAD --oneline           # 3. check commit count/messages
# 4. run the project's tests (pytest / npm test / cargo test / go test ./...)
# 5. scan changed files for debug artifacts with the `search_files` tool
#    (console.log, debugger, pdb.set_trace, breakpoint(), TODO REMOVE)
git push -u origin feat/my-feature            # 6. push
```

### 10 — Recover from common mistakes

**"My branch has diverged from main"**

```bash
git fetch origin
git log --oneline --graph origin/main HEAD
git rebase origin/main
```

**"I committed to main by accident"** — preserve the work first, then reset
main only after previewing the change and getting an explicit confirmation:

```bash
git branch fix/my-accidental-commit   # 1. save the commit on a new branch
git status                            # 2. preview what leaves main
git log origin/main..HEAD --oneline
git diff origin/main..HEAD
# 3. Only after the user confirms the above is safe to drop from main:
git reset --hard origin/main
git checkout fix/my-accidental-commit # 4. continue on the saved branch
```

**"I lost a stash or dropped commits"**

```bash
git reflog --all
git checkout -b recovery/<name> <hash>
```

## Pitfalls

- **Confirm before every hard reset.** A `git reset --hard` (or any other
  `--hard`/discard) permanently drops uncommitted work. Show `git status` and
  `git diff HEAD` so the user sees exactly what disappears, and wait for an
  explicit yes before running it — this includes `git reset --hard origin/main`
  even when a branch was created first.
- **Never `--force` push to main/master/develop.** Use `--force-with-lease`,
  and confirm the target branch before any force operation.
- **Prefer `revert` over `reset` on shared branches.** `reset` rewrites history
  others may have pulled; `revert` adds a safe, new commit.
- **Check `git stash list` before a hard reset** so work that lives only in a
  stash is not lost.
- **Dry-run rebases mentally.** Read `git log --oneline origin/main..HEAD`
  before `git rebase -i` so you know exactly what you are rewriting.

## Verification

- `git status` is clean (or shows only what you expect) after each step.
- `git log --oneline origin/main..HEAD` lists the intended commits, squashed
  and messaged as planned.
- After a rebase, `git diff --diff-filter=U` returns nothing — no unresolved
  conflicts remain.
- Force-pushes used `--force-with-lease` and succeeded without overwriting a
  teammate's commit.
- No `git reset --hard` ran without a preceding `git status` / `git diff`
  preview and an explicit user confirmation.
