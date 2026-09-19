---
name: hermes-ngit
description: Work with Nostr Git repos via ngit and gitworkshop.
version: 1.1.0
author: Joey Stanford (@rinchen)
license: CC-BY-SA-4.0
platforms: [linux, macos]
compatibility: "ngit installed (gitworkshop.dev/ngit) with git-remote-nostr on PATH; an identity configured via `git config --global nostr.nsec` or a bunker reference. No daemon required. Optional: `gh` CLI for GitHub mirroring."
metadata:
  hermes:
    tags: [Nostr, Git, ngit, Decentralized, gitworkshop]
    category: devops
    requires_toolsets: [terminal]
---

# ngit Skill

Guide Nostr-based Git hosting via ngit / gitworkshop.dev: announce repos, open
PRs/issues, sync grasp servers, and configure multi-mirror push. Does not replace
GitHub CI — keep forges where they help; use ngit for permissionless provenance.

**Credit / Attribution.** Built on **Dan Conway's** ngit command reference
(`DanConwayDev/ngit-cli`, `skills/ngit/SKILL.md`), incorporated under
**CC-BY-SA-4.0**. Dan supplies the command reference, PR/issues lifecycle, account
management, and flag tables. Joey Stanford added multi-mirror packaging, the
`ngit init` remote-rewrite gotcha, and session detection. See `NOTICE`.

## When to Use

- Announce or clone a Nostr Git repo (`nostr://` remotes)
- Open/update ngit PRs (mandatory `pr/` branch prefix) or issues
- Keep gitworkshop.dev current via `git push` (not `ngit sync` alone)
- Configure dual/triple mirrors (GitHub + Radicle + Nostr) safely
- Recover from grasp-server NIP-34 state-event "purgatory" push rejections
- Detect whether the current session repo is Nostr-enabled

## Prerequisites

- `ngit` installed ([gitworkshop.dev/ngit](https://gitworkshop.dev/ngit) —
  `curl -Ls https://ngit.dev/install.sh | bash` installs `ngit` + `git-remote-nostr`)
- `git-remote-nostr` on PATH
- Identity configured:
  - raw nsec: `git config --global nostr.nsec "<nsec1...>"`, or
  - bunker (preferred): `nostrsec://...` / `--bunker-url bunker://...`
- Optional: `gh` CLI for GitHub mirroring
- Supported platforms: Linux and macOS (`platforms: [linux, macos]`)

**No daemon required.** Unlike Radicle, ngit/relays need no local node.

Quick check via `terminal`:

```bash
ngit --version && ngit account whoami --json --offline
```

## How to Run

Prefer `terminal` for all `ngit` / `git` invocations. Inspect remote names and
URLs in the `terminal` tool result directly (for example after `git remote -v`).
Use `search_files` only when hunting files on disk, not for filtering remote
listings.

## Quick Reference

| Task | Command |
|------|---------|
| Announce repo | `ngit init -d --name <repo>` |
| Repo metadata | `ngit repo --json --offline` |
| Clone | `git clone nostr://<npub>/<relay-hint>/<id>` |
| Open PR branch | `git checkout -b pr/<feature>` then `git push -u origin pr/<feature>` |
| List PRs / issues | `ngit pr list --json` / `ngit issue list --json` |
| Sync grasp refs | `ngit sync` (does **not** update gitworkshop announcement) |
| Whoami | `ngit account whoami --json` |
| Show remotes | `git remote -v` |

## Procedure

### 1. Session repo detection

When the user is inside a Git working tree, run via `terminal`:

```bash
git remote -v
```

If any remote URL contains `nostr://`, treat the repo as Nostr-enabled. Optionally
confirm with `ngit repo --json --offline` (after `git fetch origin` if the cache is
cold — `is_nostr_repo: false` can be a false negative). Derive a gitworkshop link as
`https://gitworkshop.dev/<nostr-url-without-scheme>`.

### 2. How ngit works (mental model)

Nostr publishes signed events to relays. Identity is a keypair; data replicates across
relays. ngit splits Git into:

- **Git state (refs)** — signed events on Nostr (source of truth)
- **Git data (objects)** — ordinary git servers listed in the announcement

**Grasp servers** (e.g. `relay.ngit.dev`) combine a relay + git server. `ngit init`
announces the repo (NIP-34) and sets a `nostr://` origin. `ngit sync` only pushes
**refs** to grasp git servers — it does **not** update the announcement gitworkshop
reads. Keep gitworkshop current by adding the `nostr://` URL as an `origin` **pushurl**
so normal `git push` drives `git-remote-nostr`. `ngit send` opens a PR/proposal — do
not run it on every push.

### 3. Key rules

- **`pr/` prefix is MANDATORY** for PRs (e.g. `pr/my-feature`)
- Prefer `--json` on `ngit` read commands; use `--offline` after the first network call
- Never invent NIP-05 addresses; use `npub1...` unless a NIP-05 was given
- `<ID|nevent>` accepts hex or `nevent1...`; `--json` returns `nevent1…` for ids
- Reference related items in `--body` with `nostr:nevent1…` / `nostr:naddr1…` URIs

### 4. GOTCHA: `ngit init` rewrites `origin`

`ngit init` sets `remote.origin.url` to `nostr://` and **drops existing pushurls**.

1. Capture GitHub (or other) push URLs **before** `ngit init`
2. After init, restore pushurls in this order:
   1. `git remote set-url --push origin <github-url>`
   2. `git remote set-url --add --push origin <nostr://url>`
   3. Optional Radicle: `git remote set-url --add --push origin "$(git remote get-url --push rad)"`

Doing `--add --push` before `--push` leaves only Nostr and drops GitHub.

```bash
# BEFORE ngit init — capture the forge push URL while origin is still GitHub/etc.
GH_PUSH=$(git remote get-url --push origin)

ngit init -d --name <repo-name>

[ -n "$GH_PUSH" ] && git remote set-url --push origin "$GH_PUSH"
git remote set-url --add --push origin "$(git remote get-url origin)"
[ -n "$(git remote get-url --push rad 2>/dev/null)" ] && \
  git remote set-url --add --push origin "$(git remote get-url --push rad)"
```

### 5. Dual / multi-mirror push (generic)

Prefer **distinct remotes** so each target can be pushed independently:

```bash
git remote add github <github-url>   # fetch source for forge PRs
# after ngit init, origin.url is nostr://; keep stacked pushurls on origin OR push named remotes
git push github <branch>
git push origin <branch>             # if origin carries Nostr (+ optional others)
```

**Hardened split (recommended):** forge PRs merge on GitHub, but `ngit init` makes
`git pull` fetch Nostr only — so GitHub merges can be invisible until push rejects.

```bash
git remote add github <github-url>
git config branch.<main>.remote github
git config branch.<main>.merge refs/heads/<main>
git config remote.pushDefault origin
git config push.default current
git config remote.github.fetch '+refs/heads/<main>:refs/remotes/github/<main>'
```

Lifecycle that stays converged:

```bash
git checkout -b feat/x
git push -u github feat/x          # PR branch to forge only — never bare `git push`
# … PR merges on GitHub …
git checkout <main> && git pull    # from github
git push                           # origin triple/dual mirror
git push -d github feat/x && git branch -d feat/x
```

**Fail-closed stacked pushurls:** `git push` aborts at the first failed URL. A branch
refspec does **not** skip a subset of push URLs. Correct fallback:

1. `git remote -v` to list push URLs
2. `git remote set-url --delete --push origin <unavailable-url>`
3. `git push origin <branch>` (remaining healthy URLs)
4. Restore with `git remote set-url --add --push origin <unavailable-url>`
5. Or push a named remote / direct URL: `git push <github-url> HEAD:<branch>`

If Nostr is the flaky URL, drop that pushurl and use `ngit sync` separately until relays
recover — do not assume `git push origin <branch>` skips Nostr while it remains configured.

### 6. Announce, clone, publish

```bash
ngit init --name "My Project" --description "What it does" -d
ngit repo edit --description "New description"
ngit repo --json --offline
```

```bash
git clone nostr://<npub>/<relay-hint>/<identifier>
git clone nostr://<npub>/<identifier>
```

Append Nostr as a pushurl only after restoring any forge pushurl (see gotcha):

```bash
git remote set-url --add --push origin "$(git remote get-url origin)"
```

### 7. Pull requests

Branch name **MUST** start with `pr/`:

```bash
git checkout -b pr/my-feature
# single commit — omit title/description; commit subject/body used automatically
git push -u origin pr/my-feature

# multiple commits — literal \n\n in push options (not ANSI-C quoting)
git push -u origin pr/my-feature \
  -o 'title=My feature title' \
  -o 'description=First paragraph.\n\nSecond paragraph.'
```

Advanced `ngit send` (ANSI-C quoting for real newlines in `--description`):

```bash
ngit send HEAD~2 --subject "My Feature" --description $'First paragraph.\n\nSecond paragraph.'
ngit send --defaults
ngit send HEAD~2 --in-reply-to <PR-event-id>
```

```bash
ngit pr list --json
ngit pr view <ID|nevent> --json --comments
ngit pr comment <ID|nevent> --body "Looks good"
ngit pr checkout <ID|nevent>
ngit merge <ID|nevent>                 # local merge; then publish
git push origin <default-branch>
ngit pr close <ID|nevent> --reason "..."
ngit pr reopen <ID|nevent> --reason "..."
ngit pr ready <ID|nevent> --reason "..."
ngit pr draft <ID|nevent> --reason "..."
ngit pr label <ID|nevent> --label bug
```

### 8. Issues and accounts

```bash
ngit issue create --subject "Bug title" --body "Details" --label bug
ngit issue list --json
ngit issue view <ID|nevent> --json --comments
ngit issue comment <ID|nevent> --body "Reproduced"
ngit issue close <ID|nevent> --reason "wontfix"
ngit issue resolved <ID|nevent> --reason "fixed"
ngit issue reopen <ID|nevent> --reason "regression"
```

```bash
ngit account whoami --json
ngit account login
ngit account login --bunker-url bunker://...
ngit account create --name "Alice"
ngit account export-keys
ngit account logout
ngit --nsec <nsec> <command>           # CI inline; never commit nsec
```

### 9. Sync and tuning

```bash
ngit sync
ngit sync --ref-name main
ngit --customize
git config nostr.repo-relay-only true
git config nostr.http-io-timeout-ms 600000
```

| Flag | Description |
|------|-------------|
| `-d`, `--defaults` | Non-interactive defaults |
| `--offline` | Local cache only |
| `--json` | Structured output (ngit only) |
| `-n`, `--nsec <NSEC>` | Inline key for CI |
| `-f`, `--force` | Bypass safety guards |
| `-v`, `--verbose` | Verbose output |

### 10. Sharing locations

Use placeholders for whatever locations the user configured — never hardcode a personal
npub, RID, or forge path in this skill:

- Git forge: `https://github.com/<owner>/<repo>`
- Radicle: `rad:<rid>`
- Nostr / gitworkshop: `https://gitworkshop.dev/<npub>/<relay>/<repo>`

### 11. Recovery: NIP-34 state-event "purgatory"

**Symptom.** After deleting merged PR branches, `git push` to a grasp server fails with
`ERR authorisation failed: N state events in purgatory` listing branches that no longer
exist locally.

**Cause.** Each `ngit init` publishes a kind 30617 state event. Some grasp servers do not
honor NIP-33 replaceable semantics — old events accumulate. The server may check the
**union** of all purgatory events.

**Probe** on the `nostr://` URL only (so forge/Radicle pushurls are untouched):

```bash
git branch enrich <declared-commit>
git branch -f main <another-declared-commit>
git push --force nostr://<npub>/<grasp-relay>/<repo> main enrich
```

- Accepted → latest-only check → `ngit init --clean -f -d`
- Rejected listing other branches → union check → full reconcile:

```bash
# Phase 1 — recreate every declared branch at its exact commit
git branch <name> <sha>
# …

# Phase 2 — satisfy the union on the nostr URL only
git push --force --all nostr://<npub>/<grasp-relay>/<repo>

# Phase 3 — single combined state event
ngit init --clean -f -d

# Phase 4 — drop stale branches; prune ONLY the nostr URL (never origin)
git branch -f main <current-head-sha>
git branch -D <stale-1> <stale-2>
git push --force --prune nostr://<npub>/<grasp-relay>/<repo>
ngit init --clean -f -d
```

**Hard rule:** never `--prune` via `origin` when `origin` also carries GitHub/Radicle
pushurls — that deletes real forge branches. Always push purge steps to the `nostr://`
URL directly. Treat the primary grasp relay as authoritative; secondary grasp hosts can
lag.

## Pitfalls

- Stacked `origin` pushurls are fail-closed; delete/restore the bad URL or use distinct
  remotes / direct URLs — a refspec does not skip Nostr/Radicle.
- `git pull` after `ngit init` may fetch Nostr only while PRs merge on GitHub — split
  pull (`github`) vs push (`origin`) as above.
- Diagnosing divergence: fetch each remote into throwaway refs and compare with
  `git merge-base --is-ancestor` + `git diff` / tree OIDs before any reset. Prefer
  `git merge --no-ff` of the forge tip; hard-resetting to GitHub can non-fast-forward
  Nostr/Radicle.
- Never store a raw nsec in a repo or repo-local git config — global config or bunker only.
- `ngit sync` alone leaves gitworkshop stale; use Nostr pushurl + `git push`.

## Verification

- `ngit --version` and `ngit account whoami --json` succeed
- `git remote -v` shows `nostr://` after init/clone
- PRs use `pr/` branches; `ngit pr list --json` returns expected entries
- Mirror fallback uses distinct remotes or delete/restore of failed push URLs
- Purgatory recovery prunes only via a direct `nostr://` URL, never stacked `origin`
