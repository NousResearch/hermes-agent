---
title: Master ViberQC Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-master-viberqc
---

# Master ViberQC

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/Master ViberQC` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master ViberQC` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-master-viberqc]]` |

## Package Scripts

| Script | Command |
|---|---|
| `audit` | `npm audit --audit-level=high` |
| `build` | `next build` |
| `check-env` | `bash scripts/check-env.sh` |
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `conform:verbose` | `bash scripts/code-conform.sh --verbose` |
| `db:generate` | `drizzle-kit generate` |
| `db:migrate` | `drizzle-kit migrate` |
| `db:push` | `drizzle-kit push` |
| `db:studio` | `drizzle-kit studio` |
| `dev` | `next dev --port 3030` |
| `format` | `prettier --write .` |

## Top-Level Directories

- `--version/`
- `.claude/`
- `.cursor/`
- `.git-hooks/`
- `.gitlab/`
- `.handoff/`
- `.husky/`
- `.next/`
- `.playwright-mcp/`
- `.viberqc/`
- `artifacts/`
- `backups/`

## Top-Level Files

- `.coderabbit.yaml`
- `.cursorignore`
- `.cursorrules`
- `.depcheckrc`
- `.env`
- `.env.example`
- `.env.local`
- `.env.production.example`
- `.gitignore`
- `.gitlab-ci.yml`
- `.gitleaks.toml`
- `.hermes-projects.md`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`
- `.env.local`
- `.env.production.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`start local dev server and check localhost`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
