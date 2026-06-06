---
title: BOI Web Main Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: customer-project-boi-web-main
---

# BOI Web Main

## Identity

| Field | Value |
|---|---|
| Category | `Customer Project` |
| Relative path | `Customer Project/BOI Web Main` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Customer Project/BOI Web Main` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `critical` |
| Primary role | `architect` |
| Obsidian note | `[[customer-project-boi-web-main]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `dev` | `next dev -p 4000` |
| `lint` | `eslint` |
| `optimize` | `node scripts/optimize-images.mjs` |
| `optimize:dry` | `node scripts/optimize-images.mjs --dry` |
| `resize` | `node scripts/resize-image.mjs` |
| `start` | `next start -p 4000` |

## Top-Level Directories

- `.cursor/`
- `.next/`
- `Docs/`
- `messages/`
- `public/`
- `scripts/`
- `src/`

## Top-Level Files

- `.cursorrules`
- `.dockerignore`
- `.env`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `AGENTS.md`
- `docker-compose.yml`
- `Dockerfile`
- `eslint.config.mjs`
- `next-env.d.ts`
- `next.config.ts`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`start local dev server and check localhost`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
