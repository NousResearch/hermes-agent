---
title: Hermes Agent Operating System
tags:
  - hermes-agent
  - operating-system
status: active
updated: 2026-05-24
---

# Hermes Agent Operating System

This folder is the source of truth for making **Hermes Agent** the primary working system for the 40-project laptop workspace.

## 2026-05-24 Knowledge Center Decision

The Obsidian Knowledge Center is no longer a separate standalone vault built
from scratch. User chose migration option 1: restore/rename the existing
HermesNous graph as `~/ObsidianVault/HermesAgent`.

The source-of-truth Obsidian entry points are:

- `~/ObsidianVault/HermesAgent/MOC.md`
- `~/ObsidianVault/HermesAgent/AI_MEMORY.md`
- `~/ObsidianVault/HermesAgent/docs/OBSIDIAN_LINK_INDEX.md`
- `~/ObsidianVault/HermesAgent/docs/AI_SKILL_ROUTER.md`
- `~/ObsidianVault/HermesAgent/docs/SKILL_GRAPH.md`

The short-lived generated `HermesAgent` vault is backup/staging only.

## Runtime Boundary

Hermes Agent is the primary orchestrator. It may use the restored HermesAgent
Obsidian graph as its knowledge source, but it must not call obsolete HermesNous
or Hermes Labs runtime services.

| System | New Role | Runtime Link From Hermes Agent |
|---|---|---:|
| Hermes Agent | Primary operator, dashboard, kanban, project registry, skill routing | yes, self only |
| Restored HermesAgent vault | Knowledge Center from the old HermesNous graph | yes, files only |
| HermesNous runtime service | Legacy runtime/service identity | no |
| Hermes Labs | Legacy memory/runtime source and historical atlas source | no |

Allowed:
- Read and write the restored `~/ObsidianVault/HermesAgent` Knowledge Center.
- Read project folders directly from the filesystem.
- Store all new operating knowledge in Hermes Agent docs and `.hermes`.

Forbidden:
- No runtime calls to `127.0.0.1:7421` or `127.0.0.1:7422`.
- No MCP dependency on Hermes Labs.
- No disconnected new Knowledge Center graph when an existing HermesAgent layer can be merged.
- No writes into the 40 projects during onboarding unless a later task explicitly asks for a per-project change.

## Artifact Index

| File | Purpose |
|---|---|
| `01-audit.md` | Read-only findings from HermesNous, Obsidian, localhost, and filesystem inventory |
| `02-architecture.md` | Standalone architecture, boundary contract, data ownership, verification gates |
| `03-project-registry.md` | Canonical 40-project registry generated from local filesystem candidates |
| `04-roles-skills-kanban.md` | Required roles, role skills, Hermes Agent profiles, skill/routing model |
| `05-obsidian-bridge.md` | Current Obsidian bridge: restored HermesNous graph renamed as HermesAgent |
| `06-implementation-compliance.md` | Phase issues, completion percentages, localhost/VPS verification status |
| `07-runtime-skills.md` | Local runtime skills installed into `.hermes/skills` and mirrored to role profiles |
| `08-operator-runbook.md` | How to operate Hermes Agent across the 40-project workspace |
| `09-phase-delivery-report.md` | Current phase compliance report with numeric completion |
| `11-knowledge-center-plan.md` | 3-tier Knowledge Center implementation plan |
| `12-knowledge-intake-router.md` | Skill/Agent/KB classifier and Obsidian routing layer |
| `context-packs/` | One generated onboarding pack per canonical local project |

## Operating Principle

Hermes Agent should feel like a cockpit, not a passive archive:

1. Project registry answers "what exists".
2. Role profiles answer "who should work on it".
3. Kanban answers "what is moving now".
4. Obsidian bridge answers "what knowledge is reusable".
5. Verification gates answer "is this real or just claimed".
