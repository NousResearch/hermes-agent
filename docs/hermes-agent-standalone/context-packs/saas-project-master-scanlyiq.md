---
title: Master ScanlyIQ Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-master-scanlyiq
---

# Master ScanlyIQ

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/Master ScanlyIQ` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master ScanlyIQ` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-master-scanlyiq]]` |

## Package Scripts

| Script | Command |
|---|---|
| `audit` | `pnpm audit --audit-level=high` |
| `build` | `next build` |
| `check-env` | `bash scripts/check-env.sh` |
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `conform:verbose` | `bash scripts/code-conform.sh --verbose` |
| `db:generate` | `prisma generate` |
| `db:migrate` | `prisma migrate dev` |
| `db:push` | `prisma db push` |
| `db:seed` | `tsx prisma/seed.ts` |
| `db:studio` | `prisma studio` |
| `dev` | `NODE_OPTIONS='--max-old-space-size=8192' next dev --turbopack -p 5008` |

## Top-Level Directories

- `--version/`
- `.cursor/`
- `.git-hooks/`
- `.github/`
- `.gitlab/`
- `.husky/`
- `.next/`
- `docker/`
- `Docs/`
- `e2e/`
- `engine/`
- `prisma/`

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
- `.mise.toml`
- `.prettierignore`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`start local dev server and check localhost`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
