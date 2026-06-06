---
title: Jigsaw AI Team Mock 2026 Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: office-project-jigsaw-ai-team-mock-2026
---

# Jigsaw AI Team Mock 2026

## Identity

| Field | Value |
|---|---|
| Category | `Office Project` |
| Relative path | `Office Project/Jigsaw AI Team Mock 2026` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Office Project/Jigsaw AI Team Mock 2026` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[office-project-jigsaw-ai-team-mock-2026]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `dev` | `next dev -p 3001` |
| `dev:safe` | `rm -rf .next && WATCHPACK_POLLING=true WATCHPACK_POLLING_INTERVAL=1000 next dev -p 3005 -H 127.0.0.1` |
| `dev:stable` | `WATCHPACK_POLLING=true WATCHPACK_POLLING_INTERVAL=1000 next dev -p 3001 -H 127.0.0.1` |
| `lint` | `next lint` |
| `start` | `next start -p 3001` |
| `test` | `jest` |
| `test:coverage` | `jest --coverage` |
| `test:e2e:automation` | `playwright test -c playwright.automation.config.ts` |
| `test:watch` | `jest --watch` |

## Top-Level Directories

- `.claude/`
- `.cursor/`
- `.go/`
- `.next/`
- `.playwright/`
- `.swc/`
- `.vscode/`
- `__tests__/`
- `_archive/`
- `app/`
- `backend/`
- `BAK/`

## Top-Level Files

- `.cursorignore`
- `.cursorrules`
- `.env`
- `.env.example`
- `.env.local`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `.tmp-backend.log`
- `AGENTS.md`
- `AUDIT_REPORT_AI_AGENTS.md`
- `AUDIT_REPORT_SETTINGS_MODALS.md`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`
- `.env.local`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`run package test script`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
