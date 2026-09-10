---
name: loop-engineering
description: 'Design, scaffold, audit, and operate agent loops — scheduled automations that prompt agents on your behalf. Use when setting up automated workflows like daily triage, PR babysitter, CI sweeper, or dependency sweeper.'
version: 1.0.0
author: Hermes Agent (adapted from cobusgreyling/loop-engineering)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [loops, automation, scheduling, infrastructure, governance]
    related_skills: [loopy, cronjob, self-learning, safety-guard]
---

# Loop Engineering

Design the system that prompts your agents. Stop prompting manually — build loops that triage, fix, verify, and report on a schedule. This skill covers the **infrastructure** side: scaffolding, scoring, scheduling, worktree isolation, drift detection, and governance. For loop _content_ (discovering, crafting, running, debriefing individual loops), use the **loopy** skill.

## When to use this skill

- "Set up automated PR review on this repo"
- "I want daily issue triage that runs unattended"
- "Run loop doctor on this project"
- "What's my Loop Ready score?"
- "Add a CI sweeper loop"
- "Wire up loop-engineering to this codebase"

## The five building blocks + memory

| Block                    | What it is                              | Hermes Equivalent                     |
| ------------------------ | --------------------------------------- | ------------------------------------- |
| Automations / Scheduling | Discovery + triage on a cadence         | `cronjob` tool                        |
| Worktrees                | Safe parallel execution per fix attempt | `delegate_task` with isolated context |
| Skills                   | Persistent project knowledge            | `skills/` directory                   |
| Plugins & Connectors     | Reach into real tools (MCP)             | MCP servers in `config.yaml`          |
| Sub-agents               | Maker / checker split                   | `delegate_task` with leaf role        |
| + Memory / State         | Durable spine outside any conversation  | `memory` tool + `hermes_state.py`     |

## The 7 production patterns

| Pattern            | Cadence   | Autonomy        | Token cost |
| ------------------ | --------- | --------------- | ---------- |
| Daily Triage       | 1d–2h     | L1 report       | Low        |
| PR Babysitter      | 5–15m     | L1 watch        | High       |
| CI Sweeper         | 5–15m     | L2 cautious     | Very high  |
| Dependency Sweeper | 6h–1d     | L2 patch-only   | Medium     |
| Changelog Drafter  | 1d or tag | L1 draft        | Low        |
| Post-Merge Cleanup | 1d–6h     | L1 off-peak     | Low        |
| Issue Triage       | 2h–1d     | L1 propose-only | Low        |

## Quickstart: install loop-engineering into a project

### Step 1: Pick a pattern

Map the user's problem to the right pattern:

| Symptom                            | Pattern              |
| ---------------------------------- | -------------------- |
| Morning chaos / unclear priorities | `daily-triage`       |
| PRs stalling                       | `pr-babysitter`      |
| CI red / flakes                    | `ci-sweeper`         |
| CVE / Dependabot noise             | `dependency-sweeper` |
| Post-merge TODOs piling up         | `post-merge-cleanup` |
| Stale release notes                | `changelog-drafter`  |
| Noisy issues                       | `issue-triage`       |

### Step 2: Scaffold with Hermes cron

Use Hermes's `cronjob` tool to schedule the loop:

```
cronjob add --schedule "every 1d" --prompt "Triage open issues..." --skills loopy
```

### Step 3: Doctor check

Review the loop configuration:

- Is the schedule appropriate?
- Are the right skills loaded?
- Is the autonomy level correct?
- Are safety guards enabled?

### Step 4: Week-one rule

**Report only.** No auto-fix, no auto-merge. Start at L1 and phase up.

## Autonomy levels

| Level  | What it does                      | When to use                                                  |
| ------ | --------------------------------- | ------------------------------------------------------------ |
| **L1** | Report only. No changes.          | Week one. Always start here.                                 |
| **L2** | Assisted fixes with human gate.   | After L1 is stable for 2+ weeks.                             |
| **L3** | Unattended fixes within denylist. | Loop Ready ≥ 80, verifier passing, human explicitly opts in. |

**Never jump straight to L3.** Phase through L1 → L2 → L3.

## Safety rules

1. **Week one is always L1 (report-only).** No exceptions.
2. **No auto-merge on main** except trivial dependency patches.
3. **All unattended code changes run in isolated contexts** via `delegate_task`.
4. **Token caps are enforced.** Check budget before enabling high-cadence loops.
5. **Kill switch:** `cronjob pause` or disable in config.
6. **Enable `safety-guard` skill** for all L2+ autonomous loops.

## Relationship to loopy

|                 | loopy                                           | loop-engineering                                          |
| --------------- | ----------------------------------------------- | --------------------------------------------------------- |
| **Focus**       | Loop _content_ — discover, craft, run, debrief  | Loop _infrastructure_ — scaffold, score, schedule, govern |
| **Output**      | A loop definition (LOOPS.md entry)              | A scored, scheduled, isolated loop system                 |
| **When to use** | "I want to turn this repeated task into a loop" | "I want automated loops running on this repo"             |

They work together: use **loop-engineering** to scaffold the infrastructure, then use **loopy** to refine the loop content within that infrastructure.

## Hermes Integration

- Use `cronjob` tool for all loop scheduling
- Use `delegate_task` for isolated loop execution
- Use `safety-guard` skill for autonomous loop protection
- Use `memory` tool for loop state persistence
- Use `terminal` for git operations within loops
- Use `write_file` to create `LOOPS.md` and `STATE.md`
- Combine with `self-learning` to harvest successful loop patterns as skills
