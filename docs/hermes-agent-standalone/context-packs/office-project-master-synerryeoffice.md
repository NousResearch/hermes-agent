---
title: Master SynerryEoffice Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: office-project-master-synerryeoffice
---

# Master SynerryEoffice

## Identity

| Field | Value |
|---|---|
| Category | `Office Project` |
| Relative path | `Office Project/Master SynerryEoffice` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Office Project/Master SynerryEoffice` |
| Stack | `node/vite` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[office-project-master-synerryeoffice]]` |

## Package Scripts

| Script | Command |
|---|---|
| `audit` | `npm audit --audit-level=high` |
| `build` | `turbo build` |
| `check-env` | `bash scripts/check-env.sh` |
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `conform:verbose` | `bash scripts/code-conform.sh --verbose` |
| `db:down` | `docker compose down` |
| `db:migrate` | `cd apps/api && pnpm db:migrate` |
| `db:reset` | `cd apps/api && pnpm db:reset` |
| `db:seed` | `cd apps/api && pnpm db:seed` |
| `db:studio` | `cd apps/api && pnpm db:studio` |
| `db:up` | `docker compose up -d` |

## Top-Level Directories

- `--version/`
- `.claude/`
- `.cursor/`
- `.git-hooks/`
- `.gitlab/`
- `.husky/`
- `.pm2-logs/`
- `.turbo/`
- `apps/`
- `Blog/`
- `Docs/`
- `HTML/`

## Top-Level Files

- `.cursorignore`
- `.cursorrules`
- `.depcheckrc`
- `.env`
- `.env.example`
- `.gitignore`
- `.gitlab-ci.yml`
- `.gitleaksignore`
- `.hermes-projects.md`
- `.mcp.json`
- `.npmrc`
- `.pnpm-approve-builds.json`

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
