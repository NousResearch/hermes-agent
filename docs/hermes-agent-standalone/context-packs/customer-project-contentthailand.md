---
title: Contentthailand Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: customer-project-contentthailand
---

# Contentthailand

## Identity

| Field | Value |
|---|---|
| Category | `Customer Project` |
| Relative path | `Customer Project/Contentthailand` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Customer Project/Contentthailand` |
| Stack | `node` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `critical` |
| Primary role | `architect` |
| Obsidian note | `[[customer-project-contentthailand]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `pnpm --filter web build` |
| `dev` | `pnpm --filter web dev` |
| `lint` | `pnpm --filter web lint` |
| `typecheck` | `pnpm --filter web tsc --noEmit` |

## Top-Level Directories

- `apps/`
- `Docs/`
- `nginx/`
- `packages/`
- `scripts/`

## Top-Level Files

- `.cursorrules`
- `.env.example`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `AGENTS.md`
- `API_SPEC.md`
- `DB_SCHEMA.md`
- `DESIGN_SYSTEM.md`
- `docker-compose.prod.yml`
- `docker-compose.yml`
- `memory.md`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`start local dev server and check localhost`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
