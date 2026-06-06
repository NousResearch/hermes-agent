---
title: Master_GodeysDB Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-master-godeysdb
---

# Master_GodeysDB

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/Master_GodeysDB` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master_GodeysDB` |
| Stack | `node/vite` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-master-godeysdb]]` |

## Package Scripts

| Script | Command |
|---|---|
| `audit` | `npm audit --audit-level=high` |
| `build` | `turbo run build` |
| `check-env` | `bash scripts/check-env.sh` |
| `clean` | `turbo run clean && rm -rf node_modules` |
| `db:generate` | `npm run generate --workspace=@gods-eye/db` |
| `db:migrate` | `npm run migrate --workspace=@gods-eye/db` |
| `db:push` | `npm run push --workspace=@gods-eye/db` |
| `db:studio` | `npm run studio --workspace=@gods-eye/db` |
| `dev` | `turbo run dev` |
| `doctor` | `node scripts/doctor.mjs` |
| `format` | `prettier --write "**/*.{ts,tsx,js,jsx,json,md}"` |
| `health` | `bash scripts/health-check.sh` |

## Top-Level Directories

- `.claude/`
- `.cursor/`
- `.git-hooks/`
- `.husky/`
- `.prelaunch/`
- `.turbo/`
- `.vscode/`
- `apps/`
- `backups/`
- `BP-MKT/`
- `docker/`
- `Docs/`

## Top-Level Files

- `.cursorignore`
- `.cursorindexingignore`
- `.cursorrules`
- `.dockerignore`
- `.env`
- `.env.example`
- `.env.production.example`
- `.gitignore`
- `.gitlab-ci.yml`
- `.gitleaksignore`
- `.hermes-projects.md`
- `.mcp.json`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`
- `.env.production.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`start local dev server and check localhost`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
