---
title: Hermes Agent Standalone Audit
tags:
  - hermes-agent
  - audit
  - hermesnous
  - obsidian
status: complete
updated: 2026-05-19
---

# 01 Audit

## Scope

Read-only audit of the old HermesNous system, the current Hermes Agent install, Obsidian integration, and local project candidates. No files outside Hermes Agent were changed.

## Verified Counts

| Item | Count | Evidence |
|---|---:|---|
| System agents | 9 | `~/.claude/agents/*.md` |
| Business agents | 20 | `~/.claude/agents/business/**/*.md` |
| HermesNous `nous-*` skills | 14 | `HermesNous/skills/{conductor,core,domain}/**/nous-*.md` |
| Extra HermesNous skill packages | 2 | `prompt-shortcuts`, `wow-design-master` |
| Candidate top-level project folders | 42 | filesystem scan under `~/Documents/Viber Project/*/*` |
| Canonical projects for Agent onboarding | 40 | excludes folders with no git and no `.hermes`: `Private Project/EA Factoring`, `Tech Tools/OpenClaw2` |

## Old HermesNous Operating Model

```text
User prompt
  -> conductor-router
  -> business agent
  -> nous-conductor
  -> core phase skill
  -> optional domain skill
  -> optional system subagent
  -> trace/report/decision
```

Core phase chain:

```text
spec -> plan -> build -> test -> review -> ship
```

Domain overlays:

```text
frontend-ui, api-design, security-hardening, debugging-recovery,
code-simplification, documentation-adrs, browser-testing
```

## Old Obsidian Integration

Existing vault:

```text
~/ObsidianVault/HermesNous
```

Observed symlinks:

| Vault entry | Target |
|---|---|
| `docs` | `HermesNous/docs` |
| `skills` | `HermesNous/skills` |
| `knowledge` | `HermesNous/knowledge` |
| `lessons` | `HermesNous/lessons` |
| `patterns` | `HermesNous/patterns` |
| `playbooks` | `HermesNous/playbooks` |
| `review-queue` | `HermesNous/review-queue` |
| `reports` | `HermesNous/reports` |

This is useful as a historical knowledge vault, but it is not suitable as the new Hermes Agent runtime source because it preserves live links to the old HermesNous folder.

## Localhost Verification

| Target | Result | Interpretation |
|---|---:|---|
| Hermes Agent dashboard `/` | 200 HTML | running |
| Hermes Agent dashboard `/chat` | 200 | browser chat route available |
| HermesNous `/api/health` | ok | old monitor alive |
| HermesNous `/api/status` | degraded | registry shows 0 and usage checks error on old Labs path permissions |
| Hermes Labs `/health` | ok | old runtime alive |
| Hermes Labs `/api/atlas` | returns 40 projects | historical reference only; not used as new Agent runtime dependency |
| VPS | not checked | no VPS target was specified for this standalone design pass |

## Risk Findings Carried Forward

| Risk | Standalone Design Response |
|---|---|
| Old HermesNous/Labs boundary drift | no live runtime dependency from Hermes Agent |
| Old cross-scope shell calls | no shell calls into HermesNous or Hermes Labs from Agent workflows |
| Old secret and daemon drift | no copying secrets into docs; `.hermes` stays gitignored |
| Duplicate docs and stale counts | one Agent registry file becomes the new count source for this workspace |
| Obsidian symlinks into old system | new Obsidian bridge uses export/import packs, not live symlinks into old runtime folders |

## Audit Decision

HermesNous and Hermes Labs remain available as old independent systems. Hermes Agent becomes the new primary operator by owning its own registry, profiles, kanban, docs, and Obsidian bridge.

