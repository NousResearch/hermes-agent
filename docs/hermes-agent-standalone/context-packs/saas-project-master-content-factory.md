---
title: Master Content Factory Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-master-content-factory
---

# Master Content Factory

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/Master Content Factory` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master Content Factory` |
| Stack | `node/vite` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-master-content-factory]]` |

## Package Scripts

| Script | Command |
|---|---|
| `audit` | `npm audit --audit-level=high` |
| `build` | `tsc && cp -r src/dashboard dist/dashboard` |
| `check-env` | `bash scripts/check-env.sh` |
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `conform:verbose` | `bash scripts/code-conform.sh --verbose` |
| `db:generate` | `prisma generate` |
| `db:migrate` | `prisma migrate dev` |
| `db:migrate:deploy` | `prisma migrate deploy` |
| `db:push` | `prisma db push` |
| `db:seed` | `tsx prisma/seed.ts` |
| `db:studio` | `prisma studio` |

## Top-Level Directories

- `--version/`
- `.credentials/`
- `.cursor/`
- `.git-hooks/`
- `.github/`
- `.gitlab/`
- `.husky/`
- `.playwright-mcp/`
- `backups/`
- `BP-MKT/`
- `config/`
- `data/`

## Top-Level Files

- `.cursorignore`
- `.cursorrules`
- `.cursorrules.content-factory`
- `.depcheckrc`
- `.env`
- `.env.example`
- `.env.production.template`
- `.gitignore`
- `.gitlab-ci.yml`
- `.gitleaksignore`
- `.hermes-projects.md`
- `.mcp.json`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`
- `.env.production.template`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`inspect package scripts and README`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
