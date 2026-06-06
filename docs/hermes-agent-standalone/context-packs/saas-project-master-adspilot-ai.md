---
title: Master AdsPilot-AI Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-master-adspilot-ai
---

# Master AdsPilot-AI

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/Master AdsPilot-AI` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master AdsPilot-AI` |
| Stack | `node` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-master-adspilot-ai]]` |

## Package Scripts

| Script | Command |
|---|---|
| `conform` | `bash scripts/code-conform.sh` |
| `conform:fix` | `bash scripts/code-conform.sh --fix` |
| `migrate:verify` | `bash scripts/migration-verify.sh` |
| `quality` | `bash scripts/quality-gate.sh` |
| `quality:all` | `bash scripts/quality-gate.sh && bash scripts/security-scan.sh && bash scripts/dependency-audit.sh` |
| `quality:code` | `bash scripts/code-quality-scan.sh` |
| `quality:deps` | `bash scripts/dependency-audit.sh` |
| `security` | `bash scripts/security-scan.sh` |
| `session:end` | `bash scripts/plan-update.sh session-end` |
| `sync` | `bash scripts/sync-check.sh` |
| `sync:fix` | `bash scripts/sync-check.sh --fix` |

## Top-Level Directories

- `.cursor/`
- `.git-hooks/`
- `.gitlab/`
- `apps/`
- `BP-MKT/`
- `docs/`
- `infra/`
- `packages/`
- `scripts/`
- `services/`
- `TEMP/`
- `templates/`

## Top-Level Files

- `.cursorignore`
- `.cursorrules`
- `.editorconfig`
- `.env`
- `.env.example`
- `.gitignore`
- `.gitlab-ci.yml`
- `.golangci.yml`
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
