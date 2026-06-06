---
title: Hermes Agent Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: tech-tools-hermes-agent
---

# Hermes Agent

## Identity

| Field | Value |
|---|---|
| Category | `Tech Tools` |
| Relative path | `Tech Tools/Hermes Agent` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent` |
| Stack | `node` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[tech-tools-hermes-agent]]` |

## Package Scripts

| Script | Command |
|---|---|
| `postinstall` | `echo '✅ Browser tools ready. Run: python run_agent.py --help'` |

## Top-Level Directories

- `.github/`
- `.plans/`
- `acp_adapter/`
- `acp_registry/`
- `agent/`
- `assets/`
- `cron/`
- `datagen-config-examples/`
- `docker/`
- `docs/`
- `gateway/`
- `hermes_agent.egg-info/`

## Top-Level Files

- `.dockerignore`
- `.env.example`
- `.envrc`
- `.gitattributes`
- `.gitignore`
- `.mailmap`
- `AGENTS.md`
- `batch_runner.py`
- `cli-config.yaml.example`
- `cli.py`
- `constraints-termux.txt`
- `CONTRIBUTING.md`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env.example`
- `.envrc`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`inspect package scripts and README`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
