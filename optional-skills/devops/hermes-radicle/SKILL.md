---
name: hermes-radicle
description: Work with Radicle repos for peer-to-peer Git collaboration.
version: 1.0.1
author: Joey Stanford (@rinchen)
license: MIT
platforms: [linux, macos]
compatibility: "Radicle installed and `rad auth` complete; Radicle node running (`rad node start`). Optional: `gh` CLI for GitHub mirroring."
metadata:
  hermes:
    tags: [Radicle, Git, P2P, Version Control, Decentralized]
    category: devops
    requires_toolsets: [terminal]
---

# Radicle Skill

Guide Radicle peer-to-peer Git workflows: init/clone, patches, issues, node ops,
sync, and multi-remote mirrors. Does not replace GitHub CI or forge PR UIs —
keep those where they help; use Radicle for distributed provenance and review.

## When to Use

- Initialize or clone a Radicle repository (`rad://` remotes)
- Create or update patches and issues on Radicle
- Start/stop/inspect a local Radicle node
- Sync with peers or inspect remotes
- Detect whether the current session repo is Radicle-enabled
- Configure dual/triple push (e.g. GitHub + Radicle) with correct fail-open fallbacks

## Prerequisites

- Radicle CLI installed and identity configured: `rad auth`
- Local node running when pushing/syncing: `rad node start`
- Optional: `gh` CLI when mirroring to GitHub
- Supported platforms: Linux and macOS (`platforms: [linux, macos]`)

Quick check via `terminal`:

```bash
rad --version && rad node status
```

## How to Run

Prefer `terminal` for all `rad` / `git` invocations. Inspect remote names and
URLs in the `terminal` tool result directly (for example after `git remote -v`).
Use `search_files` only when hunting files on disk, not for filtering remote
listings.

## Quick Reference

| Task | Command |
|------|---------|
| Init repo | `rad init` |
| Clone | `rad clone rad:<rid>` |
| Create patch | `rad patch create --base <default-branch>` |
| List issues | `rad issue list` |
| Create issue | `rad issue create --title "..." --description "..."` |
| Node start/stop | `rad node start` / `rad node stop` |
| Node status | `rad node status` |
| Sync peers | `rad sync` |
| Show remotes | `git remote -v` |
| Push to Radicle remote | `git push rad <branch>` |

## Procedure

### 1. Session repo detection

When the user is inside a Git working tree, run via `terminal`:

```bash
git remote -v
```

If any remote is named `rad` / `rad-push`, or any URL starts with `rad://` /
`rad:`, treat the repo as Radicle-enabled. Read the RID from `rad.rid` (if set)
or from the remote URL, then tell the user the repo is on Radicle.

### 2. Init a new Radicle repo

```bash
rad init
```

Creates a repository ID and configures the `rad` remote. Confirm with
`git remote -v` — you should see a `rad` remote with a `rad://` URL.

### 3. Clone an existing Radicle repo

```bash
rad clone rad:<rid>
cd <project>
git remote -v
```

Expect `rad` (and often `rad-push`) remotes after a successful clone.

### 4. Patches and issues

```bash
git checkout <feature-branch>
rad patch create --base <default-branch>
```

```bash
rad issue list
rad issue create --title "..." --description "..."
```

Use the repo's actual default branch (`main`, `master`, …). A wrong refspec
fails against every configured push URL.

### 5. Node ops and sync

```bash
rad node start
rad node stop
rad node status
rad node info
rad sync
```

### 6. Dual / multi-mirror push (generic)

Prefer **distinct remotes** so each target can be pushed independently:

```bash
git remote add github <github-url>   # if not already present as a fetch remote
# after rad init, `rad` already exists
git push github <branch>
git push rad <branch>
```

If you intentionally stack multiple push URLs on one remote (e.g. `origin`):

```bash
git remote set-url --add --push origin <github-url>
git remote set-url --add --push origin "$(git remote get-url --push rad)"
```

Then a plain `git push` attempts every push URL in order and **fails closed** —
the first failing URL aborts the rest.

**Correct fallback when one mirror is down** (a branch refspec does *not*
select a subset of push URLs):

1. List push URLs: `git remote -v`
2. Temporarily remove the unavailable URL:
   `git remote set-url --delete --push origin <unavailable-url>`
3. Push the healthy targets: `git push origin <branch>`
4. Restore the URL when the node/service is back:
   `git remote set-url --add --push origin <unavailable-url>`
5. Or skip stacked pushurls entirely and push named remotes:
   `git push github <branch>` then later `git push rad <branch>`

Pushing directly to a URL also bypasses other pushurls:

```bash
git push <github-url> HEAD:<branch>
```

### 7. Sharing locations

When sharing a repo, include whatever locations the user actually configured
(placeholders only — never hardcode a personal RID or forge URL in this skill):

- Git forge: `https://github.com/<owner>/<repo>` (or equivalent)
- Radicle: `rad:<rid>`

## Pitfalls

- Stacked `origin` pushurls are fail-closed: one down mirror blocks the others
  until you delete that push URL or push a distinct remote / direct URL.
- `git push origin <branch>` still hits **all** of `origin`'s push URLs; it is
  not a GitHub-only escape hatch.
- Default branch names differ (`main` vs `master`). Always push the branch that
  exists locally.
- A stopped Radicle node (`rad node status`) will make Radicle pushurls fail —
  start the node before retrying Radicle.

## Verification

- `rad --version` succeeds
- `rad node status` shows a running node when sync/push is required
- `git remote -v` shows the expected `rad://` remote after init/clone
- Patch/issue commands return without auth/node errors
- Mirror fallback either uses distinct remotes or delete/restore of the failed
  push URL (never assumes a refspec skips Radicle)
