---
title: AI In Office Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: tech-tools-ai-in-office
---

# AI In Office

## Identity

| Field | Value |
|---|---|
| Category | `Tech Tools` |
| Relative path | `Tech Tools/AI In Office` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Tech Tools/AI In Office` |
| Stack | `node/next` |
| Git | `false` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[tech-tools-ai-in-office]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `db:deploy` | `prisma migrate deploy` |
| `db:generate` | `prisma generate` |
| `db:migrate` | `prisma migrate dev` |
| `db:seed` | `tsx prisma/seed.ts` |
| `db:studio` | `prisma studio` |
| `dev` | `next dev` |
| `lint` | `eslint` |
| `start` | `next start` |
| `test` | `vitest run` |
| `typecheck` | `tsc --noEmit` |

## Top-Level Directories

- `.next/`
- `app/`
- `components/`
- `docs/`
- `lib/`
- `prisma/`
- `public/`
- `tests/`

## Top-Level Files

- `.dockerignore`
- `.env`
- `.env.example`
- `.env.production.example`
- `.env.vps.example`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `docker-compose.prod.yml`
- `docker-compose.yml`
- `Dockerfile`
- `eslint.config.mjs`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`
- `.env.production.example`
- `.env.vps.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`run package test script`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
