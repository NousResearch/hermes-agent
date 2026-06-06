---
title: Hermes Agent 40 Project Registry
tags:
  - hermes-agent
  - project-registry
status: complete
updated: 2026-05-19
---

# 03 Project Registry

## Registry Rule

This registry is generated from the local filesystem, not from Hermes Labs atlas. It is the Agent-owned source of truth for this standalone operating system.

Discovery found 42 top-level candidates under `~/Documents/Viber Project/*/*`. The canonical 40 exclude folders that have neither git nor `.hermes`:

- `Private Project/EA Factoring`
- `Tech Tools/OpenClaw2`

## Summary

| Category | Count |
|---|---:|
| Backup Project | 1 |
| Customer Project | 4 |
| Office Project | 7 |
| SaaS Project | 12 |
| Tech Tools | 16 |
| Total | 40 |

| Stack | Count |
|---|---:|
| Node | 20 |
| Python | 2 |
| Unknown/mixed | 18 |

## Canonical Projects

| # | Category | Project | Stack | Git | `.hermes` | Primary Agent Role | Risk |
|---:|---|---|---|:---:|:---:|---|---|
| 1 | Backup Project | Admin Console | node | yes | yes | legacy-support-lead | high |
| 2 | Customer Project | BOI Web Main | node | yes | yes | head-of-engineering | critical |
| 3 | Customer Project | Contentthailand | node | yes | yes | head-of-engineering | critical |
| 4 | Customer Project | DRA | unknown | yes | yes | head-of-engineering | critical |
| 5 | Customer Project | bted-demo | node | yes | yes | ux-ui-product-lead | critical |
| 6 | Office Project | Jigsaw AI Team Mock 2026 | node | yes | yes | ux-ui-product-lead | high |
| 7 | Office Project | MD Assist by AI | unknown | yes | yes | system-transformation-lead | high |
| 8 | Office Project | Master SynerryEoffice | node | yes | yes | system-transformation-lead | critical |
| 9 | Office Project | Master SynerryNew | node | yes | yes | head-of-engineering | high |
| 10 | Office Project | Master WebEngine | unknown | yes | yes | head-of-engineering | critical |
| 11 | Office Project | Support Center | unknown | no | yes | legacy-support-lead | medium |
| 12 | Office Project | jigsaw-ai-team-website | node | yes | yes | frontend-engineer | high |
| 13 | SaaS Project | LuiLogicLab | node | yes | yes | product-growth-lead | high |
| 14 | SaaS Project | MQ5 Market | unknown | yes | yes | quant-ea-specialist | critical |
| 15 | SaaS Project | Master AdsPilot-AI | node | yes | yes | backend-engineer | critical |
| 16 | SaaS Project | Master Content Factory | node | yes | yes | backend-engineer | high |
| 17 | SaaS Project | Master Fundamental | node | yes | yes | quant-ea-specialist | critical |
| 18 | SaaS Project | Master JigsawWebChat | node | yes | yes | frontend-engineer | critical |
| 19 | SaaS Project | Master ScanlyIQ | node | yes | yes | backend-engineer | critical |
| 20 | SaaS Project | Master ViberQC | node | yes | yes | qa-verification-lead | high |
| 21 | SaaS Project | Master_GodeysDB | node | yes | yes | quant-ea-specialist | critical |
| 22 | SaaS Project | SaaS Web Engine | unknown | yes | no | head-of-engineering | high |
| 23 | SaaS Project | Venture Radar | unknown | yes | yes | backend-engineer | high |
| 24 | Tech Tools | AI In Office | node | no | yes | system-transformation-lead | medium |
| 25 | Tech Tools | AI on Premis | python | yes | no | devex-automation-lead | high |
| 26 | Tech Tools | AIControlCenter | node | yes | yes | system-transformation-lead | high |
| 27 | Tech Tools | Asana Labs | unknown | no | yes | system-transformation-lead | medium |
| 28 | Tech Tools | AutoGPT Platform Lab | unknown | no | no | research-sandbox-lead | medium |
| 29 | Tech Tools | Canvas Labs | unknown | no | yes | knowledge-obsidian-lead | medium |
| 30 | Tech Tools | EmailHunter | unknown | yes | yes | backend-engineer | medium |
| 31 | Tech Tools | Hermes Agent | python | yes | yes | hermes-agent-architect | critical |
| 32 | Tech Tools | Hermes Labs | node | yes | yes | legacy-reference-only | isolated |
| 33 | Tech Tools | HermesNous | unknown | yes | yes | legacy-reference-only | isolated |
| 34 | Tech Tools | Lark Labs | unknown | no | yes | integration-ops-lead | medium |
| 35 | Tech Tools | Main Server | unknown | no | yes | network-infra-architect | high |
| 36 | Tech Tools | Notion Labs | unknown | no | yes | knowledge-obsidian-lead | medium |
| 37 | Tech Tools | OpenClaw | unknown | yes | yes | migration-sunset-lead | high |
| 38 | Tech Tools | PaperClip | unknown | yes | yes | system-transformation-lead | high |
| 39 | Tech Tools | VPS Server | unknown | yes | yes | network-infra-architect | critical |
| 40 | Tech Tools | thClaw | unknown | no | yes | migration-sunset-lead | medium |

## Onboarding Policy

| Project Type | Default Action |
|---|---|
| Critical customer or SaaS | read-only context pack first, no automation until verified |
| Tech Tools legacy systems | isolate, document boundary, no runtime dependency |
| No git but has `.hermes` | read-only registry only until owner asks for changes |
| No `.hermes` but has git | create Agent-side context pack only; do not write into project |

## First Pilot Set

| Pilot | Reason |
|---|---|
| Hermes Agent | self-hosted system being built now |
| Venture Radar | SaaS project with backend ownership and manageable scope |
| bted-demo | customer-facing UI project that benefits from WOW and browser verification |

## Rollout Batches

| Batch | Projects | Gate |
|---|---:|---|
| Batch A pilot | 3 | dashboard + context + kanban verified |
| Batch B high-value SaaS | 10 | no critical secret exposure in docs |
| Batch C customer/office | 10 | role ownership mapped |
| Batch D tech tools | 12 | boundary isolation checked |
| Batch E low/no-git workspaces | 5 | read-only unless explicitly changed |

