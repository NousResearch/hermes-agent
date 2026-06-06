---
title: bted-demo Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: customer-project-bted-demo
---

# bted-demo

## Identity

| Field | Value |
|---|---|
| Category | `Customer Project` |
| Relative path | `Customer Project/bted-demo` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Customer Project/bted-demo` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `critical` |
| Primary role | `architect` |
| Obsidian note | `[[customer-project-bted-demo]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `check:images` | `node scripts/check-image-sizes.mjs --check-only` |
| `dev` | `next dev -p 4000` |
| `dev:clean` | `rm -rf .next && next dev -p 4000` |
| `lint` | `next lint` |
| `optimize:images` | `node scripts/check-image-sizes.mjs` |
| `start` | `next start` |
| `test` | `vitest run` |
| `test:a11y` | `playwright test e2e/accessibility.spec.ts` |
| `test:all` | `vitest run && playwright test` |
| `test:e2e` | `playwright test e2e/security.spec.ts` |
| `test:watch` | `vitest` |

## Top-Level Directories

- `.claude/`
- `.github/`
- `.next/`
- `__tests__/`
- `app/`
- `components/`
- `Docs/`
- `e2e/`
- `i18n/`
- `lib/`
- `logs/`
- `messages/`

## Top-Level Files

- `.dockerignore`
- `.env.example`
- `.env.local`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `AGENTS.md`
- `CLAUDE.md`
- `docker-compose.yml`
- `Dockerfile`
- `eslint.config.mjs`
- `lighthouserc.js`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env.example`
- `.env.local`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`run package test script`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
