---
title: Hermes Labs Context Pack
tags:
  - hermes-agent
  - context-pack
  - workspace-40
status: active
updated: 2026-05-20
project_slug: tech-tools-hermes-labs
---

# Hermes Labs

## Identity

| Field | Value |
|---|---|
| Category | `Tech Tools` |
| Relative path | `Tech Tools/Hermes Labs` |
| Absolute path | `/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Labs` |
| Stack | `node` |
| Git | `true` |
| `.hermes` | `true` |
| Risk | `isolated` |
| Primary role | `sunset` |
| Obsidian note | `[[tech-tools-hermes-labs]]` |

## Package Scripts

| Script | Command |
|---|---|
| `cli` | `tsx src/cli/index.ts` |
| `dev` | `tsx src/daemon.ts` |
| `health` | `[legacy runtime command omitted: forbidden in standalone workflow]` |
| `install:service` | `bash scripts/install.sh` |
| `logs` | `tail -f ~/.hermes-launchd/daemon.out.log ~/.hermes-launchd/daemon.err.log` |
| `mcp` | `tsx src/mcp/server.ts` |
| `restart` | `launchctl kickstart -kp gui/$(id -u)/com.hermes.daemon && sleep 2 && npm run health` |
| `smoke` | `tsx scripts/smoke.ts` |
| `smoke:all` | `tsx scripts/smoke.ts && tsx scripts/incidents-smoke.ts && tsx scripts/incidents-search-smoke.ts && tsx scripts/preflight-smoke.ts && tsx scripts/security-smoke.ts && tsx scripts/pattern-smoke.ts` |
| `typecheck` | `tsc --noEmit` |

## Top-Level Directories

- `audit/`
- `backups/`
- `bin/`
- `Blog/`
- `config/`
- `docs/`
- `logs/`
- `memory/`
- `modules/`
- `profiles/`
- `scripts/`
- `skills/`

## Top-Level Files

- `.gitignore`
- `.hermes-projects.md`
- `AGENTS.md`
- `FINAL-REPORT-2026-04-21.md`
- `HANDOFF-2026-04-20.md`
- `HANDOFF-2026-04-21.md`
- `LEAD-DEV-BRIEFING.md`
- `package-lock.json`
- `package.json`
- `PROJECT-SPEC.md`
- `README.md`
- `TEAM-ONBOARDING.md`

## Secret Hygiene

Secret-bearing files detected by name only:

- none detected

No secret file contents were read or copied into this pack.

## First Safe Verification Gate

`reference-only; no runtime calls to this project`

## Runtime Boundary

- This project context pack is owned by Hermes Agent.
- HermesNous and Hermes Labs are historical/reference-only for this standalone system.
- Do not call HermesNous port 7422 or Hermes Labs port 7421 from new workflows.
