# AI Office Mac START HERE

Updated: 2026-05-08 20:13 KST
Source machine: WSL `/home/lidises/hermes-agent`
Branch: `ai-office-stage6-7-cleanup-20260508`
Known WSL HEAD before Mac-doc commit: `604698991671`
Current pushed handoff HEAD: `2943093817cc`

## Purpose

Use the AI Office read-only dashboard on Mac while reusing the WSL/GitHub work as much as possible.

The safe topology is:

```text
GitHub / fork branch
  Shared source code and docs

NAS Hermes ledger
  Shared handoff notes, context, START HERE, status

Mac local machine
  Mac-specific Hermes checkout, Python venv, node_modules, ~/.hermes/.env, ~/.hermes/config.yaml, sessions, logs, auth
```

Do not run Hermes directly from the NAS share. The NAS is the shared ledger, not the runtime location.

## What can be reused on Mac

Reuse these from GitHub or NAS:

- Branch source code: `ai-office-stage6-7-cleanup-20260508`
- AI Office backend: `hermes_cli/office_state.py`, `hermes_cli/office_adapters.py`, `hermes_cli/office_redaction.py`
- AI Office route: `GET /api/office/state`
- AI Office frontend: `web/src/pages/OfficePage.tsx`, `web/src/lib/api.ts`, `web/src/App.tsx`
- Docs under `docs/ai-office/`
- NAS handoff notes under `/mnt/nas/Hermes/_admin/handoffs/` or the Mac-visible equivalent

Recreate these on Mac instead of copying from WSL:

- `.venv/`
- `web/node_modules/`
- `hermes_cli/web_dist/` build output
- `~/.hermes/.env`
- `~/.hermes/config.yaml` unless reviewed line-by-line
- `~/.hermes/auth.json`
- sessions, logs, cron DB, gateway runtime state, local caches

## Mac setup from GitHub

```bash
mkdir -p ~/dev
cd ~/dev

git clone -b ai-office-stage6-7-cleanup-20260508 git@github.com:lidises/hermes-agent.git
cd hermes-agent

git rev-parse --short=12 HEAD
git status --short
```

Expected branch:

```bash
git branch --show-current
# ai-office-stage6-7-cleanup-20260508
```

If the branch already exists locally:

```bash
cd ~/dev/hermes-agent
git fetch fork ai-office-stage6-7-cleanup-20260508 || git fetch origin ai-office-stage6-7-cleanup-20260508
git switch ai-office-stage6-7-cleanup-20260508
git pull --ff-only
```

## Mac Python environment

```bash
cd ~/dev/hermes-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[web]'
python -m pip install -e '.[pty]'
python -m pip install -e '.[acp]'
```

## Mac Node/frontend environment

Use Mac-native Node. Do not copy WSL `node_modules`.

```bash
cd ~/dev/hermes-agent/web
npm ci
npm run build
```

If `npm ci` fails because the local package manager state is stale, use:

```bash
npm install
npm run build
```

## Mac environment variables

Create Mac-local env only:

```bash
mkdir -p ~/.hermes
cp ~/dev/hermes-agent/docs/ai-office/mac-env.template ~/.hermes/.env.ai-office.template
nano ~/.hermes/.env
```

Only put real secrets in Mac `~/.hermes/.env`. Do not put secrets in GitHub, NAS, Obsidian, or this repo.

At minimum, set only the providers/tools actually used on Mac. See `docs/ai-office/mac-env.template` for placeholders.

If the Mac has a local Obsidian vault or NAS mount, set Mac paths, not WSL paths:

```bash
OBSIDIAN_VAULT_PATH=/Users/<mac-user>/path/to/Obsidian Vault
HERMES_LEDGER_PATH=/Volumes/Hermes
```

If the NAS is not mounted on Mac yet, leave `HERMES_LEDGER_PATH` unset and run the dashboard locally first.

## Run the AI Office dashboard on Mac

```bash
cd ~/dev/hermes-agent
source .venv/bin/activate
hermes dashboard --host 127.0.0.1 --port 8765 --no-open
```

Open:

```text
http://127.0.0.1:8765/office
```

Stop/status:

```bash
cd ~/dev/hermes-agent
source .venv/bin/activate
hermes dashboard --status
hermes dashboard --stop
```

## Local verification on Mac

Short focused tests:

```bash
cd ~/dev/hermes-agent
source .venv/bin/activate
scripts/run_tests.sh \
  tests/hermes_cli/test_office_redaction.py \
  tests/hermes_cli/test_office_state_adapters.py \
  tests/hermes_cli/test_office_api.py \
  -q --tb=short
```

Frontend build check:

```bash
cd ~/dev/hermes-agent/web
npm run build
```

## Important behavior differences

The Mac dashboard shows Mac-local Hermes state:

```text
Mac ~/.hermes config/session/cron/kanban/log state
Mac-visible NAS/Obsidian paths
Mac-local provider/tool config
```

It does not automatically show WSL's live sessions, WSL cron jobs, WSL gateway state, or WSL local DBs.

To share cross-device knowledge, use the NAS/Obsidian ledger. To share code, use the GitHub branch. Keep runtime state local per device.

## Remote viewing alternative

If the goal is to view the WSL dashboard from Mac, use an SSH tunnel rather than copying runtime to NAS or binding the dashboard publicly. Do not use `--insecure` unless explicitly accepting LAN exposure.

## Current official PR status at handoff time

PR: https://github.com/NousResearch/hermes-agent/pull/21813

The PR is for upstream review/CI. The Mac can use the fork branch before the PR is merged.
