---
title: VPS Server Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: tech-tools-vps-server
---

# VPS Server

## Identity

| Field | Value |
|---|---|
| Category | `Tech Tools` |
| Relative path | `Tech Tools/VPS Server` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Tech Tools/VPS Server` |
| Stack | `unknown/mixed` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `critical` |
| Primary role | `architect` |
| Obsidian note | `[[tech-tools-vps-server]]` |

## Package Scripts

| Script | Command |
|---|---|
| none detected | - |

## Top-Level Directories

- `Docs/`
- `jsoncrack-vscode/`
- `logs/`
- `nginx/`
- `scripts/`
- `templates/`

## Top-Level Files

- `.cursorignore`
- `.env`
- `.env.example`
- `.gitignore`
- `.hermes-projects.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docker-compose.monitoring.yml`
- `README.md`
- `server_health_dashboard.sh`
- `ssh_diagnostic.sh`
- `ssh_monitor.plist`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`inspect package scripts and README`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
