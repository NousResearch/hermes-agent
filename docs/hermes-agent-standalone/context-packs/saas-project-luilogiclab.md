---
title: LuiLogicLab Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: saas-project-luilogiclab
---

# LuiLogicLab

## Identity

| Field | Value |
|---|---|
| Category | `SaaS Project` |
| Relative path | `SaaS Project/LuiLogicLab` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/SaaS Project/LuiLogicLab` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[saas-project-luilogiclab]]` |

## Package Scripts

| Script | Command |
|---|---|
| `analyze` | `ANALYZE=true next build` |
| `build` | `NODE_OPTIONS='--max-old-space-size=4096' next build` |
| `dev` | `WATCHPACK_POLLING=true WATCHPACK_POLLING_INTERVAL=3000 NODE_OPTIONS='--max-old-space-size=4096' next dev -p 3003` |
| `lint` | `next lint` |
| `prebuild` | `rm -rf .next` |
| `predev` | `lsof -ti:3003 / xargs kill -9 2>/dev/null; true` |
| `prepare` | `husky` |
| `seed` | `npx tsx scripts/seed-data.ts` |
| `start` | `next start -p 3003` |
| `test` | `npx playwright test` |
| `test:chrome` | `npx playwright test --project=chromium` |
| `test:headed` | `npx playwright test --headed` |

## Top-Level Directories

- `.claude/`
- `.gitlab/`
- `.husky/`
- `.next/`
- `.vscode/`
- `Blog/`
- `deploy/`
- `docs/`
- `messages/`
- `playwright-report/`
- `public/`
- `qc-toolkit/`

## Top-Level Files

- `.cursorrules`
- `.env`
- `.env.local`
- `.env.local.backup-2026-04-19`
- `.env.local.example`
- `.env.production.example`
- `.eslintrc.json`
- `.gitignore`
- `.gitlab-ci.yml`
- `.hermes-projects.md`
- `.mcp.json`
- `.nvmrc`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.local`
- `.env.local.backup-2026-04-19`
- `.env.local.example`
- `.env.production.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`run package test script`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
