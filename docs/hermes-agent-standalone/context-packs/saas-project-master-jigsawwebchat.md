---
title: Master JigsawWebChat Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-master-jigsawwebchat
---

# Master JigsawWebChat

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/Master JigsawWebChat` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master JigsawWebChat` |
| Stack | `node/vite` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-master-jigsawwebchat]]` |

## Package Scripts

| Script | Command |
|---|---|
| `audit` | `npm audit --audit-level=high` |
| `check-env` | `bash scripts/check-env.sh` |
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `conform:verbose` | `bash scripts/code-conform.sh --verbose` |
| `format` | `prettier --write .` |
| `health` | `bash scripts/health-check.sh` |
| `lint` | `eslint . && prettier --check .` |
| `lint:fix` | `eslint --fix . && prettier --write .` |
| `migrate:verify` | `bash scripts/migration-verify.sh` |
| `prepare` | `husky` |
| `quality` | `bash scripts/quality-gate.sh` |

## Top-Level Directories

- `--version/`
- `.claude/`
- `.cursor/`
- `.git-hooks/`
- `.gitlab/`
- `.husky/`
- `.playwright-mcp/`
- `apps/`
- `deploy/`
- `Docs/`
- `packages/`
- `playwright-report/`

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
- `.prettierignore`
- `.prettierrc`

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
