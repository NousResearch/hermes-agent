---
name: hermes-config-git-backup
description: Version-control a Hermes Agent setup to a private GitHub repo, secrets-safe.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms:
- linux
- macos
metadata:
  hermes:
    tags:
    - hermes
    - git
    - backup
    - github
    - devops
    - cron
    - security
    category: devops
    related_skills:
    - hermes-agent
    - github-auth
    - script-first-cron-design
    config:
    - key: hermes-config-git-backup.enabled
      description: Enable hermes-config-git-backup skill behavior
      default: true
      prompt: Enable hermes-config-git-backup skill?
---


# Hermes Config → Private GitHub Backup

Version-control everything you've customized on top of the base Hermes image
(HERMES_HOME) to a **private** GitHub repo, without ever leaking credentials,
and keep it synced daily via a silent watchdog cron.

## When to use

- User wants their Hermes config / skills / wiki / cron jobs backed up or
  versioned to GitHub.
- User wants reproducibility, rollback, or an audit trail of config changes.
- Migrating a setup to another machine/profile.

## The hard rule: HERMES_HOME contains secrets

`HERMES_HOME` (e.g. `/opt/data` or `~/.hermes`) mixes customization WITH
credentials and live data. A naive `git add -A && push` leaks all of it —
even to a private repo (private repos can be forked, shared, or exposed via a
leaked token). **Always use a fail-closed .gitignore + a staging safety gate.**

### Never commit (verify absent every time)
- Credentials: `.env*`, `auth.json`, `google/` (token.json, client_secret.json),
  `google_*.json`, `channel_directory.json`, `whatsapp/`, `pairing/`
- Live data: `state.db*`, `kanban.db*`, `sessions/`, `logs/`, `personal-context/`
  (relationship graph + audit log — sensitive even in a private repo)
- Runtime churn: `cron/state/` (rewrites timestamps every tick → noisy diffs),
  `plugins/` state, `disk-cleanup/` logs, all caches, `skills/.hub/` (big index
  cache), upstream source dir `hermes-agent/` (its own git repo)

### Safe to version
`config.yaml` (verify no inline secrets first!), `skills/`, `wiki/`, `scripts/`,
`cron/jobs.json` + prompt templates, `plans/`, `memories/` (MEMORY.md/USER.md —
OK in a PRIVATE repo only), `SOUL.md`, `docker-compose.yml`, `agent-hooks/`,
`*-manifest.json`, setup shell scripts, `shell-hooks-allowlist.json`.

## Setup steps

### 1. Recon (read-only)
```bash
cd "$HERMES_HOME"
git rev-parse --is-inside-work-tree 2>&1     # already a repo?
ls -la                                       # inventory
# Verify config.yaml has no real secrets (empty '' / dummy values only):
grep -nEi "api_key|client_secret|password|token" config.yaml | grep -vE ":\s*''|: *$"
# Find nested git repos (skills sometimes ship their own .git):
find . -maxdepth 3 -name .git -type d
```

### 2. Fail-closed .gitignore
Ignore everything, then allow-list. See `references/gitignore-template.txt`.
Key pattern: `/*` first, then `!/config.yaml`, `!/skills/`, etc., then
re-exclude churn/secrets INSIDE allow-listed dirs.

### 3. Init + stage + AUDIT before any commit
```bash
git init -q && git branch -M main
git config user.name "<github-user>" && git config user.email "<email>"
git add -A
# CRITICAL leak gate — must print nothing:
git diff --cached --name-only | grep -iE \
 '^\.env|auth\.json|google_.*\.json|^google/|state\.db|kanban\.db|channel_directory|^whatsapp/|^pairing/|^sessions/|^logs/|^personal-context/|token\.json|client_secret'
```
If a skill has an embedded `.git`, `git rm --cached -f <dir>`, `rm -rf <dir>/.git`,
then re-add so it commits as plain files (not a broken submodule).

### 4. Auth: gh device flow (no token in chat)
`gh` may not be installed and not on PATH. Install to userspace if no root:
```bash
# arm64 example; pick asset for `uname -m`
VER=$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest | grep -oP '"tag_name":\s*"v\K[^"]+')
curl -fsSL -o /tmp/gh.tgz "https://github.com/cli/cli/releases/download/v${VER}/gh_${VER}_linux_arm64.tar.gz"
tar xzf /tmp/gh.tgz -C /tmp && cp /tmp/gh_${VER}_linux_arm64/bin/gh "$HERMES_HOME/bin/gh"
```
`gh auth login --web` buffers/needs a TTY and often fails headless
(`tcsetattr: Inappropriate ioctl`). **Use the raw OAuth device flow with curl**
instead — reliable and scriptable. See `scripts/gh-device-login.sh`.
IMPORTANT: request BOTH scopes `repo,read:org` — `gh auth login --with-token`
REJECTS a token missing `read:org`. Pipe the final token to
`gh auth login --with-token` so it persists to `hosts.yml` (don't rely on the
`GH_TOKEN` env var — it vanishes when the shell ends). Then `gh auth setup-git`.

### 5. Create private repo + push
```bash
gh repo create <user>/<repo> --private --source=. --remote=origin --push \
  --description "Hermes config (private)"
# Verify on GitHub's actual tree, not just local:
gh api "repos/<user>/<repo>/git/trees/main?recursive=1" -q '.tree[].path' \
  | grep -iE 'token|secret|\.env|auth\.json' || echo "CLEAN"
```

### 6. Sync tooling + daily watchdog cron
Install `scripts/git-sync/hermes-sync.sh` (see template). It:
- stages, runs the leak gate, aborts on sensitive files
- generates a VERBOSE commit (what: areas + file stats; why: intent) — see below
- `--quiet` for cron (silent when nothing changed)

Cron wrapper = **watchdog pattern**: silent on success, alert only on failure.
The cron `script` field needs a path RELATIVE to `$HERMES_HOME/scripts/` and
CANNOT take args — so use a tiny wrapper that hardcodes `--quiet`:
```bash
# scripts/git-sync/hermes-sync-cron.sh
out=$(/abs/path/scripts/git-sync/hermes-sync.sh --quiet 2>&1); rc=$?
[ $rc -ne 0 ] && { echo "⚠️ Hermes backup FAILED ($rc):"; echo "$out"; exit $rc; }
exit 0   # success => silent (no ping)
```
Register with `no_agent: true` (zero LLM cost), schedule e.g. `0 4 * * *`.

## Commit messages: why + learnings, NOT a file list

The git diff already records WHICH files changed — repeating that in the message
is noise. Keep the message about intent. `hermes-sync.sh` produces a short
subject plus a body of:
- **Why:** one line (manual vs scheduled intent).
- **Notes / learnings:** optional — only when you pass it.

Pass intent as args (quote each):
```bash
hermes-sync "feat: add X skill"
hermes-sync "fix: dedupe resolver" "Gateway hashes senders before the agent sees
them, so the raw-number path never fired — matched on the sha256 instead."
HERMES_SYNC_NOTES="why this matters / gotcha" hermes-sync "chore: tune cron"
```
Subject auto-derives to `sync: update <areas> (<ts>)` when omitted (cron path).
Capture real learnings (decisions, gotchas, why-not-the-obvious-thing) — those
are what a future reader can't reconstruct from the diff.

## Cleaning messy history

Build-out often leaves many `auto-sync: <timestamp>` test commits. Squash into
one well-documented commit before sharing:
```bash
git branch backup-pre-squash                  # safety
git reset --soft <root-commit>                # keep all files staged
git commit --amend -F /tmp/initial-msg.txt    # rich what/why message
git push --force-with-lease origin main
git branch -D backup-pre-squash
```

## Pitfalls

- **config.yaml / .env are write-protected** from the agent's patch/write tools
  (security rail). Edit config via `hermes config set KEY VAL`; edit `.env` via
  a script using `read -r VAR < file` (NOT `$(cat …)` — `security.redact_secrets`
  mangles command substitution containing secret-shaped strings, corrupting the
  written file).
- **`cron/state/` churn**: if committed, every cron tick rewrites timestamps and
  the daily backup pushes (and pings) daily for nothing. Exclude it.
  Also exclude `cron_state/` (alternate state dir used by some scripts).
- **Generated deps (node_modules) bloat the repo.** If `scripts/whatsapp-bridge/node_modules/`
  or similar was committed, `git rm --cached` stops tracking but the blobs persist
  in history. Use `git filter-branch` + aggressive gc to reclaim the space, then
  add a `setup.sh` so fresh clones can regenerate deps. See the
  `versioning-hermes-home` skill for the full purge + setup.sh recipe.
- **Script portability.** After versioning, audit scripts for hardcoded
  `/opt/data` and `/opt/hermes` paths — they break on any machine with a
  different HERMES_HOME. See the `versioning-hermes-home` skill's
  "Script portability" section and `references/script-portability-pattern.md`.
- **`cron/jobs.json` also carries `next_run_at`/`last_run_at`** that update on
  tick — expect occasional no-op-looking diffs; that's why the cron is a
  silent-success watchdog, not a "synced ✅" notifier.
- **Duplicate `credential.helper` entries break git push silently:** If `git config --global credential.https://github.com.helper` has been set more than once (e.g. by running `git config --add` twice), git fails with `cannot overwrite multiple values` and falls back to prompting for a username — which hangs headless. Symptom: push fails with `fatal: could not read Username for 'https://github.com': No such device or address`. Fix with `--replace-all`:
  ```bash
  git config --global --replace-all credential.https://github.com.helper "!/path/to/gh auth git-credential"
  git config --global --replace-all credential.https://gist.github.com.helper "!/path/to/gh auth git-credential"
  ```
  Verify: `git config --list --global | grep credential` — should show exactly one entry per host.
- **`gh` CLI path in credential helper must be absolute:** The git credential helper entry must use the **full absolute path** to `gh` (e.g. `!/opt/data/bin/gh auth git-credential`). Cron jobs run with a minimal `PATH` that won't include the directory where `gh` lives, so a bare `gh` command in the helper fails with "No such device or address" even though `gh` works interactively.
- **Don't rely on `GH_TOKEN` env** for persistence — write to `gh` `hosts.yml`.
- **Verify leaks on GitHub's tree** (`gh api .../git/trees/main?recursive=1`),
  not just `git ls-files` — proves what actually left the machine.
- Token shown in chat (if device flow code is read aloud) is one-time and
  consumed on auth; the stored `gho_` token is what matters. Rotate via GitHub
  settings if ever pasted in plaintext.

## Verification checklist

- [ ] `gh api .../git/trees/main?recursive=1` shows ZERO secret/credential paths
- [ ] repo `visibility == PRIVATE`
- [ ] `hermes-sync` clean tree → "Nothing to sync"; `--quiet` clean → silent
- [ ] cron success → silent; cron failure → alerts (test with a bad remote URL)
- [ ] local HEAD == origin/main
- [ ] a real commit shows a verbose what/why body
