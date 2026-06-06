---
title: Hermes Agent Roles Skills Kanban Model
tags:
  - hermes-agent
  - roles
  - skills
  - kanban
status: complete
updated: 2026-05-19
---

# 04 Roles, Skills, And Kanban

## Role System

These are the Hermes Agent role profiles required to replace the useful parts of the old 20 business agents and 14 HermesNous skills.

| Role Profile | Main Skill | Owns |
|---|---|---|
| `hermes-agent-architect` | systems architecture, boundary contracts, roadmap | Agent OS architecture and design decisions |
| `knowledge-obsidian-lead` | taxonomy, Obsidian, lessons/patterns/playbooks | standalone knowledge bridge |
| `agent-orchestration-lead` | routing, profiles, skill selection, kanban | task orchestration model |
| `security-boundary-lead` | secrets, isolation, runtime dependency review | no-cross-runtime enforcement |
| `devex-automation-lead` | CLI, project scans, commands, onboarding | project registry and automation |
| `qa-verification-lead` | tests, browser checks, localhost/VPS gates | verification matrix and compliance |
| `wow-operator-designer` | dashboard UX, operator workflow, clarity | high-impact browser dashboard experience |
| `migration-sunset-lead` | legacy freeze, migration sequence, fallback | old system isolation and retirement |

## Legacy Agent Mapping

| Old Business Area | New Hermes Agent Profile |
|---|---|
| conductor-router | agent-orchestration-lead |
| head-of-engineering | hermes-agent-architect |
| ux-ui-designer | wow-operator-designer |
| frontend-engineer | wow-operator-designer |
| backend-engineer | devex-automation-lead |
| network-infra-architect | qa-verification-lead + security-boundary-lead |
| qc-qa-engineer | qa-verification-lead |
| quant-ea-specialist | hermes-agent-architect |
| chief-strategist | hermes-agent-architect |
| crisis-growth-analyst | security-boundary-lead |
| system-ai-transformation-lead | agent-orchestration-lead |
| people-culture-architect | knowledge-obsidian-lead |
| cmo / marketing roles | wow-operator-designer |
| finance/accounting roles | migration-sunset-lead |
| legacy-tech-support | migration-sunset-lead |

## Phase Skills

Hermes Agent uses the old phase chain as an operating discipline, not as a live dependency.

| Phase | Agent Skill Behavior | Output |
|---|---|---|
| define | turn request into context/problem/goal/scope | spec block |
| plan | break into issues, owners, sequence, risks | issue checklist |
| build | create artifact or code in the approved workspace | changed files |
| test | run local checks and browser checks when applicable | evidence |
| review | risk, security, quality, boundary check | findings |
| ship | final handoff, rollback, compliance table | done report |

## Domain Skills

| Domain | Trigger | Lead Role |
|---|---|---|
| frontend-ui | UI, responsive, accessibility, visual polish | wow-operator-designer |
| api-design | endpoints, contracts, integration | devex-automation-lead |
| security-hardening | auth, secret, PII, permission, deploy | security-boundary-lead |
| debugging-recovery | repeated failure, regression, incident | qa-verification-lead |
| code-simplification | complexity, duplication, dead code | hermes-agent-architect |
| documentation-adrs | decisions, README, runbook, memory | knowledge-obsidian-lead |
| browser-testing | localhost, screenshots, console, keyboard | qa-verification-lead |

## Kanban Board Model

Board slug:

```text
workspace-40
```

Column contract:

| Column | Meaning |
|---|---|
| triage | rough idea or imported project observation |
| todo | scoped and waiting for role assignment |
| scheduled | waiting on time or external dependency |
| ready | runnable by a role profile |
| running | actively claimed |
| blocked | needs user/VPS/credential/human input |
| done | verified and summarized |
| archived | intentionally retired |

## Standard Task Template

```markdown
## Context
Project:
Role:
Phase:
Domain:

## Work
- Issue:
- Files allowed:
- Files forbidden:
- Runtime dependencies forbidden:
  - Hermes Labs 7421
  - HermesNous 7422

## Verification
- localhost:
- tests:
- browser:
- VPS:

## Completion
- Done %:
- Remaining %:
```

## Dispatch Rules

| Rule | Requirement |
|---|---|
| K-01 | one task has one primary role profile |
| K-02 | every task has an explicit project or `Hermes Agent` as target |
| K-03 | every task declares forbidden files/systems |
| K-04 | every UI task must include browser verification |
| K-05 | every delivery must include numeric done/remaining values |
| K-06 | no worker may write to HermesNous, Hermes Node/Nous, or Hermes Labs inside this standalone program |
