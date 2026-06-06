---
title: AIControlCenter Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: tech-tools-aicontrolcenter
---

# AIControlCenter

## Identity

| Field | Value |
|---|---|
| Category | `Tech Tools` |
| Relative path | `Tech Tools/AIControlCenter` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Tech Tools/AIControlCenter` |
| Stack | `node/next` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `high` |
| Primary role | `devex` |
| Obsidian note | `[[tech-tools-aicontrolcenter]]` |

## Package Scripts

| Script | Command |
|---|---|
| `build` | `next build` |
| `check:production` | `tsx scripts/production-readiness.ts` |
| `connector:chatgpt` | `tsx scripts/chatgpt-web-connector.ts` |
| `connector:chrome` | `bash scripts/open-chrome-debug.sh` |
| `connector:claude` | `tsx scripts/claude-web-connector.ts` |
| `connector:gemini` | `tsx scripts/gemini-web-connector.ts` |
| `connector:kimi` | `tsx scripts/kimi-web-connector.ts` |
| `connector:open:kimi` | `bash scripts/open-kimi-debug-profile.sh` |
| `connector:start` | `tsx scripts/web-connector-supervisor.ts start` |
| `connector:status` | `tsx scripts/web-connector-supervisor.ts status` |
| `connector:stop` | `tsx scripts/web-connector-supervisor.ts stop` |
| `connector:supervisor` | `tsx scripts/web-connector-supervisor.ts` |

## Top-Level Directories

- `.next/`
- `.web-sessions/`
- `app/`
- `components/`
- `docs/`
- `lib/`
- `prisma/`
- `public/`
- `scripts/`
- `tests/`

## Top-Level Files

- `.dockerignore`
- `.env`
- `.env.example`
- `.env.production`
- `.env.production.example`
- `.env.vps`
- `.env.vps.example`
- `.gitignore`
- `.hermes-projects.md`
- `.mcp.json`
- `docker-compose.prod.yml`
- `docker-compose.yml`

## Secret Hygiene

Secret-bearing files detected by name only:

- `.env`
- `.env.example`
- `.env.production`
- `.env.production.example`
- `.env.vps`
- `.env.vps.example`

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`inspect package scripts and README`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
