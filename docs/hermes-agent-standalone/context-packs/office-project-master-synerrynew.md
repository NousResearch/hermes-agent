---
title: Master SynerryNew Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: office-project-master-synerrynew
---

# Master SynerryNew

## Identity

| Field | Value |
|---|---|
| Category | `Office Project` |
| Relative path | `Office Project/Master SynerryNew` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Office Project/Master SynerryNew` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[office-project-master-synerrynew]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `check-env` | `bash scripts/check-env.sh` |
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `conform:verbose` | `bash scripts/code-conform.sh --verbose` |
| `dev` | `next dev` |
| `format` | `prettier --write .` |
| `health` | `bash scripts/health-check.sh` |
| `lint` | `eslint . --ext .js,.ts,.jsx,.tsx && prettier --check .` |
| `lint:fix` | `eslint . --ext .js,.ts,.jsx,.tsx --fix && prettier --write .` |
| `migrate:verify` | `bash scripts/migration-verify.sh` |
| `prepare` | `husky` |

## Top-Level Directories

- `--version/`
- `.claude/`
- `.cursor/`
- `.git-hooks/`
- `.gitlab/`
- `.husky/`
- `.next/`
- `.playwright-mcp/`
- `_reference/`
- `Docs/`
- `new-project-files/`
- `public/`

## Top-Level Files

- `.cursorignore`
- `.cursorrules`
- `.env`
- `.env.example`
- `.gitignore`
- `.gitlab-ci.yml`
- `.mcp.json`
- `.next_dev.log`
- `.prettierignore`
- `.prettierrc`
- `.tmp.html`
- `AGENTS.md`

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
