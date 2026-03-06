---
name: git-workflow
description: >
  Git branching, committing, conflict resolution, pull request creation, and repository
  hygiene workflows. Use when the user wants to create branches, write commit messages,
  squash or rebase commits, resolve merge conflicts, open a PR, or clean up a messy
  git history. Do NOT use for Docker, CI/CD pipelines, or GitHub Actions — those have
  separate skills.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [Git, DevOps, Version Control, GitHub, GitLab, Branching, Pull Requests]
    related_skills: []
---

# Git Workflow

A complete procedural guide for common Git tasks: branching strategy, clean commits,
conflict resolution, rebasing, and pull request discipline.

---

## When to Use

Load this skill when the user asks to:
- Create or rename a branch
- Write or fix a commit message
- Squash, amend, or rebase commits
- Resolve a merge conflict
- Open, update, or describe a pull request
- Clean up a local or remote branch list
- Recover from a detached HEAD or accidental commit to `main`

---

## Quick Reference

| Task | Command |
|---|---|
| Create & switch branch | `git checkout -b feat/my-feature` |
| Stage all changes | `git add -p` (interactive) or `git add .` |
| Commit with message | `git commit -m "type(scope): description"` |
| Amend last commit | `git commit --amend --no-edit` |
| Squash last N commits | `git rebase -i HEAD~N` |
| Rebase onto main | `git fetch origin && git rebase origin/main` |
| Push branch | `git push -u origin HEAD` |
| Force-push safely | `git push --force-with-lease` |
| Delete remote branch | `git push origin --delete branch-name` |
| Abort a rebase | `git rebase --abort` |
| Stash changes | `git stash push -m "description"` |
| Pop stash | `git stash pop` |
| Show log (graph) | `git log --oneline --graph --decorate --all` |

---

## Procedure

### 1. Start a new feature branch

Always branch from an up-to-date base:

```bash
git checkout main          # or master / trunk
git pull origin main       # sync with remote
git checkout -b feat/short-description   # create branch
```

**Branch naming convention:**
- `feat/` — new feature
- `fix/` — bug fix
- `docs/` — documentation only
- `refactor/` — code restructuring, no behavior change
- `test/` — adding or fixing tests
- `chore/` — build, CI, dependency updates

Keep branch names lowercase, hyphenated, and ≤ 50 characters.

---

### 2. Write clean commits

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short imperative description>

[optional body — explain WHY, not WHAT]

[optional footer — Closes #123, BREAKING CHANGE: ...]
```

**Rules:**
- Subject line ≤ 72 characters
- Use imperative mood: "add feature" not "added feature"
- One logical change per commit — don't mix fixes with refactors
- Never commit secrets, `.env` files, or build artifacts

**Verify what you're committing before staging:**
```bash
git diff                  # unstaged changes
git diff --cached         # staged changes
git status                # overall state
```

---

### 3. Interactive staging (preferred)

Use `git add -p` to stage only the relevant hunks, not entire files. This catches
unintended debug prints or WIP changes before they enter the history.

```bash
git add -p                # review each hunk: y/n/s(plit)/e(dit)
git commit -m "fix(auth): handle expired token on refresh"
```

---

### 4. Squash and clean up before PR

Before opening a pull request, squash WIP commits into logical units:

```bash
git rebase -i HEAD~N      # N = number of commits to review
```

In the editor, use:
- `pick` — keep as-is
- `squash` (or `s`) — merge into previous commit
- `reword` (or `r`) — keep commit, edit message
- `fixup` (or `f`) — merge into previous, discard message
- `drop` (or `d`) — delete commit entirely

After rewriting history, force-push with lease (safer than `--force`):
```bash
git push --force-with-lease
```

---

### 5. Rebase onto main (keep history linear)

Before merging, always rebase onto the latest main to avoid a messy merge commit:

```bash
git fetch origin
git rebase origin/main
```

If conflicts arise during rebase, see **Conflict Resolution** below, then:
```bash
git rebase --continue     # after resolving each conflict
```

---

### 6. Conflict Resolution

When `git merge` or `git rebase` reports a conflict:

**Step 1 — identify conflicting files:**
```bash
git status                # shows "both modified" files
```

**Step 2 — open the conflicting file.** Conflict markers look like:
```
<<<<<<< HEAD (your changes)
your version of the code
=======
their version of the code
>>>>>>> origin/main (incoming changes)
```

**Step 3 — resolve:** Edit the file to keep the correct version. Remove ALL
conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).

**Step 4 — run the helper script to verify no markers remain:**
```bash
python scripts/check_conflicts.py
```

**Step 5 — stage and continue:**
```bash
git add <resolved-file>
git rebase --continue     # or: git merge --continue
```

**To abort and start over:**
```bash
git rebase --abort        # or: git merge --abort
```

---

### 7. Open a Pull Request

**Before opening the PR checklist:**
- [ ] Branch is rebased onto latest `main`/`master`
- [ ] All WIP commits are squashed
- [ ] Tests pass locally: `pytest` / `npm test` / etc.
- [ ] No unintended files in `git diff origin/main`
- [ ] Branch name follows the convention

**Push and open PR:**
```bash
git push -u origin HEAD
```

Then on GitHub/GitLab, use the **PR description template** in
`references/pr_template.md`.

**PR title format** (same as commit convention):
```
fix(gateway): add reconnect backoff for WhatsApp session failures
```

---

### 8. Common Recovery Scenarios

**Accidentally committed to `main`:**
```bash
git branch feat/my-accidental-work    # save the work to a new branch
git reset --hard origin/main          # reset main back
git checkout feat/my-accidental-work  # continue on the correct branch
```

**Detached HEAD:**
```bash
git checkout -b recovery-branch       # create branch from current state
```

**Undo last commit (keep changes staged):**
```bash
git reset --soft HEAD~1
```

**Discard all local changes (destructive):**
```bash
git checkout -- .                     # unstaged only
git reset --hard HEAD                 # staged + unstaged
```

**Find a lost commit:**
```bash
git reflog                            # shows all recent HEAD positions
git checkout <sha>                    # inspect it
```

---

### 9. Clean up merged branches

```bash
# Delete local branches already merged into main
git branch --merged main | grep -v "^\* \|main\|master" | xargs git branch -d

# Prune remote-tracking refs for deleted remote branches
git fetch --prune

# List remote branches that no longer exist locally
git remote prune origin --dry-run
```

---

## Pitfalls

- **Never `--force` push to `main` or `master`** — always use `--force-with-lease`
  on feature branches only.
- **Don't `git add .` blindly** — always check `git status` and `git diff` first.
  Use `.gitignore` to keep build artifacts, `.env` files, and `__pycache__` out.
- **Rebase rewrites history** — coordinate with teammates before rebasing a branch
  that others have checked out.
- **Merge conflicts in lock files** (`package-lock.json`, `poetry.lock`,
  `requirements.txt`) — don't manually edit them. Regenerate with the package manager
  after resolving the source conflict:
  ```bash
  pip install -r requirements.txt   # Python
  npm install                       # Node
  ```
- **Large binary files** — never commit them directly; use Git LFS.
- **Stale branches** — run the cleanup commands in Step 9 regularly to avoid
  confusion.

---

## Verification

After completing a workflow step, confirm state with:

```bash
git log --oneline -10                         # recent history looks clean
git diff origin/main --stat                   # only expected files changed
git status                                    # working tree is clean
```

For PR readiness:
```bash
python scripts/check_conflicts.py             # no unresolved conflict markers
git log origin/main..HEAD --oneline           # only your commits, well-described
```
