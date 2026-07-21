---
name: dreamcycle-hermes-memory
description: Install and wire DreamCycle as a Hermes memory provider for Hermes users.
version: 1.0.0
author: kenjix217
license: MIT
platforms: [linux, macos, windows]
category: software-development
---

# DreamCycle memory provider for Hermes

## What this skill does

This skill gives other Hermes users a one-command way to install the DreamCycle
memory plugin, point Hermes at a running DreamCycle sidecar, and run a quick
validation smoke check.

## Prerequisites

- Git + shell access for the target user
- Running DreamCycle sidecar (`dreamcycle-server`) with key and namespace configured
- Optional but recommended: PostgreSQL + pgvector endpoint reachable by DreamCycle

## One-command install

```bash
# from this package checkout
bash scripts/install_dreamcycle_hermes_plugin.sh
```

The script installs:

- `~/.hermes/plugins/dreamcycle/__init__.py`

## Configure Hermes

Set these variables in your Hermes runtime config (equivalent to your profile env):

- `DREAMCYCLE_BASE_URL`
- `DREAMCYCLE_API_KEY`
- `DREAMCYCLE_NAMESPACE`
- `DREAMCYCLE_USER_ID`
- `DREAMCYCLE_SOURCE=hermes`

Then:

```bash
hermes config set memory.provider dreamcycle
hermes memory status
```

## Smoke check

Run smoke script after env is set:

```bash
python scripts/hermes_sidecar_smoke.py --dry-run
```

Expected:

- `health ok`
- `/v1/memory/search` and `/v1/memory/turns` return successful responses

## Verify install from this package

From this package:

- If running locally in this repo, place a copy under this profile's skill path (e.g. `~/.hermes/skills/software-development/dreamcycle-hermes-memory`) and run:

```bash
hermes skills install dreamcycle-hermes-memory
```

- If publishing to registry, users install via repo URL:

```bash
hermes skills install https://raw.githubusercontent.com/kenjix217/dreamcycle/main/skills/dreamcycle-hermes-memory/SKILL.md
```

## Release

### Release checklist

1. Freeze changes and bump version tags
2. Run packaging checks
3. Create GitHub tag and release artifacts:

```bash
bash scripts/release_dreamcycle_for_hermes.sh <VERSION>
```

4. Optionally mirror version in `CHANGELOG.md` and publish install notes.