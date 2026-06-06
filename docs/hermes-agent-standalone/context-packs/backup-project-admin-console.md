---
title: Admin Console Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: backup-project-admin-console
---

# Admin Console

## Identity

| Field | Value |
|---|---|
| Category | `Backup Project` |
| Relative path | `Backup Project/Admin Console` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Backup Project/Admin Console` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[backup-project-admin-console]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `dev` | `next dev -p 3002 -H 127.0.0.1` |
| `lint` | `eslint .` |
| `start` | `next start` |

## Top-Level Directories

- `.cursor/`
- `.next/`
- `Admin-ConsolWF/`
- `app/`
- `components/`
- `contexts/`
- `docs/`
- `hooks/`
- `lib/`
- `public/`
- `styles/`
- `types/`

## Top-Level Files

- `.cursorrules`
- `.gitignore`
- `.hermes-projects.md`
- `BUGFIX_CRM_APIKEYS.md`
- `components.json`
- `DESIGN_TOKENS.md`
- `middleware.ts`
- `MIGRATION_PLAN.md`
- `next-env.d.ts`
- `next.config.mjs`
- `package-lock.json`
- `package.json`

## Secret Hygiene

Secret-bearing files detected by name only:

- none detected

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`start local dev server and check localhost`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
