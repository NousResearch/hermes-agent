---
title: EmailHunter Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: tech-tools-emailhunter
---

# EmailHunter

## Identity

| Field | Value |
|---|---|
| Category | `Tech Tools` |
| Relative path | `Tech Tools/EmailHunter` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Tech Tools/EmailHunter` |
| Stack | `docker/mixed` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[tech-tools-emailhunter]]` |

## Package Scripts

| Script | Command |
|---|---|
| none detected | - |

## Top-Level Directories

- `.claude/`
- `api/`
- `backups/`
- `dashboard/`
- `n8n-workflows/`
- `scripts/`
- `searxng/`

## Top-Level Files

- `.env`
- `.env.template`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `AGENTS.md`
- `deploy.sh`
- `docker-compose.yml`
- `GUIDE.md`
- `PHASE_REVIEW.md`
- `stats.json`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.template`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`inspect package scripts and README`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
