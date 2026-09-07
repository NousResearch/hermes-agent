---
name: kanban-git-verified-completion
description: Commit and push every repo edit before kanban_complete; avoid the git-verified completion gate rejecting your card.
version: 1.0.0
author: Ed Box Mtce (maintenance)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, git, completion, verification]
    category: devops
    requires_toolsets: [kanban]
environments:
  - kanban
---

# Kanban Git-Verified Completion

You are about to call `kanban_complete`. Before you do, make sure every file
you touched inside a git repo is **committed and pushed** — otherwise the
git-verified completion gate in `kanban_complete` will reject your card and it
stays `running` (no state change, no downstream release).

This is the guardrail behind post-mortem card `t_6c39e36f`: a card reached
DONE while its working-tree edits (design docs + a lessons file) were never
committed, and concurrent workers share checkout directories. Never let that
happen again.

## When this applies

This skill applies whenever `HERMES_KANBAN_WORKSPACE` points into a git repo —
i.e. you ran a git task out of a worktree or a shared checkout and edited files.

## Commit + push before completing

1. Look at what you changed:
   ```bash
   git -C "$HERMES_KANBAN_WORKSPACE" status --porcelain
   ```
2. Stage and commit **every** change you made (only your task's files):
   ```bash
   git -C "$HERMES_KANBAN_WORKSPACE" add -A
   git -C "$HERMES_KANBAN_WORKSPACE" commit -m "task <HERMES_KANBAN_TASK>: <what you did>"
   ```
   Do not leave staged/unstaged edits, deletions, or untracked files behind.
   If a file is an *artifact* that will never be committed, move it out of the
   repo or add it to `.gitignore` — the tree must be clean.
3. Push the commit so it is on the remote:
   ```bash
   git -C "$HERMES_KANBAN_WORKSPACE" push
   ```
4. Record the commit SHAs in your completion metadata so the gate can verify
   them against `origin/<branch>`:
   ```
   metadata.commits = ["<sha1>", ...]   # one real SHA per commit you made
   ```
   (Only the SHAs you actually pushed; a SHA that is not on `origin/<branch>`
   will be rejected.)

## The gate checks, in order

`kanban_complete` rejects (card stays in-flight) when, for a workspace inside a
git repo:

1. **Working tree is not clean** — `git status --porcelain` is non-empty. Fix:
   commit, or stash/discard, or `.gitignore` the artifacts.
2. **A declared `metadata.commits` SHA is not on `origin/<branch>`** — Fix:
   `git push origin <branch>`.

A repo with **no `origin` remote** only enforces #1 (a clean tree); there is
nothing to push to, so the pushed-commit check is skipped.

## If you cannot meet the gate

Do NOT fake a SHA or silently drop files to pass. If you genuinely cannot
commit + push (no git in the workspace, no push credential, or a change that
must NOT be committed), call `kanban_block(reason="...")` instead of
`kanban_complete`, and say exactly what is blocking you. The human will
decide. Completing a card with an unpushed or uncommitted repo change is
exactly the failure this guardrail exists to stop.
