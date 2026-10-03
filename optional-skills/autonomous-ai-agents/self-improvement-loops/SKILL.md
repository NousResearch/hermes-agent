---
name: self-improvement-loops
description: Hypothesis, lesson, guardrail and fitness loops.
version: 1.0.0
author: ljluestc (ljluestc), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [self-improvement, experiments, hypotheses, governance, audit, fitness, lessons, cron, kanban]
    category: autonomous-ai-agents
    related_skills: [dynamic-workflow]
---

# Self-Improvement Loops Skill

Runs the long-lived "system that evolves" loops — hypothesis tracking, learning from failure,
guardrails with an audit trail, and weighted fitness scoring — on primitives Hermes already ships:
kanban, `/goal`, cron, memory, skills, shell hooks and approvals. It adds one helper,
`scripts/fitness.py`, for the only piece with no native home. It does not add a daemon, a new
tool, or a parallel JSON store beside memory and skills.

## When to Use

- The user wants to try an approach, measure it, and keep or discard it (a hypothesis ledger).
- The user wants lessons from failures captured so the next attempt starts smarter.
- The user wants dangerous actions blocked or every tool action auditable.
- The user wants a goal scored across weighted dimensions over time (speed 0.4, safety 0.3, ...).
- The user describes an "autonomous subsystem" design; `references/subsystem-map.md` maps each
  proposed subsystem to the surface below.

Skip it for one-off tasks: just do the task.

## Prerequisites

- `terminal` for `hermes ...` commands and `scripts/fitness.py` (Python 3.8+, stdlib only).
- The `kanban` toolset for the hypothesis ledger, or run `hermes kanban` through `terminal`.
- The `cronjob` toolset (`cronjob_manage`) for scheduled reviews.
- Store fitness specs and history under the profile's Hermes home, never a hardcoded home path:
  the local terminal exports `HERMES_HOME`, so use `"$HERMES_HOME/fitness/"`.

## How to Run

Pick the loop from the user's intent, run its Procedure section, and report real tool output.
When a loop should run unattended, schedule it with `cronjob_manage` and put this skill in
`skills` so each run loads it.

## Quick Reference

| Loop | Primitive | Key calls |
|---|---|---|
| Hypothesis | kanban task per hypothesis | `kanban_create`, `kanban_comment`, `kanban_complete(metadata=...)` |
| Single measurable hypothesis | `/goal` contract | `/goal ... verification: ...`, `/goal gate add <cmd>` |
| Lessons | skills + memory | `/refine`, `skill_manage`, `memory`, `session_search` |
| Guardrails | approvals + shell hooks | `/approvals smart`, `approvals.deny`, `hooks.pre_tool_call`, `hermes approvals test -- <cmd>` |
| Audit | session DB | `hermes sessions export --session-id ID --format jsonl`, `hermes logs --component tools` |
| Fitness | `scripts/fitness.py` + cron | `python scripts/fitness.py SPEC SCORES --log HISTORY` |
| Health | status commands | `hermes status`, `hermes cron status`, `hermes kanban stats`, `hermes hooks doctor` |

## Procedure

### 1. Hypothesis loop (ScienceLoop)

1. Create one task per hypothesis with `kanban_create`: `title` = the claim, `body` = how it will
   be tested and what result would falsify it. Use one `tenant` (for example `"hypotheses"`) so
   `kanban_list(tenant="hypotheses")` shows the ledger; kanban has no tags.
2. After each experiment, `kanban_comment` the command run, the measured result and the outcome
   (`success`, `failure` or `inconclusive`). Quote real output; never summarise a run you did not do.
3. Close with `kanban_complete(task_id, summary=..., metadata={"verdict": "retain" | "discard" |
   "modify", "evidence": [...]})`. For `modify`, create the revised hypothesis as a new task and
   `kanban_link` the old one to it.
4. For one hypothesis with a pass/fail check, `/goal` fits better: `/goal <claim>` with
   `verification: <evidence>` and `stop when: <blocker>` lines, plus `/goal gate add <command>` so
   the goal cannot close until the command passes.

### 2. Lessons from failure (ReflectiveEvolution)

1. Before retrying a failed task, `session_search` for the error text or task name to recall
   earlier attempts.
2. After the fix, save the root cause and the working procedure where it will load next time:
   `skill_manage` patch into the skill used for the task (Pitfalls section), or `/refine` to
   review the whole conversation. Use `memory` only for facts that hold in every session.
3. Write lessons as facts with their cause ("`uv sync` fails behind the proxy unless
   `UV_NATIVE_TLS=1`"), not as imperatives.

### 3. Guardrails and audit (Governance)

1. Check what the current policy does with a command before proposing a rule:
   `hermes approvals test -- <command>` (exit 0 allow, 2 ask, 3 deny).
2. Hard denials go in `approvals.deny` (fnmatch globs, enforced even under `/yolo`). Custom
   checks go in a `hooks.pre_tool_call` entry in `config.yaml` with a `matcher` regex and
   `fail_closed: true`; the script reads the call as JSON on stdin and blocks by exiting 2 or
   printing `{"action": "block", "message": "..."}`. Confirm with `hermes hooks test pre_tool_call
   --for-tool terminal`, then `hermes hooks doctor`.
3. For an audit trail, export the session: `hermes sessions export audit.jsonl --session-id <id>
   --format jsonl`. Add a `hooks.post_tool_call` script only when the user needs a separate log.
4. Changing `config.yaml` guardrails is the user's decision: show the exact YAML and ask first.

### 4. Fitness scoring (FitnessBuilder)

1. Write a spec with `write_file` to `"$HERMES_HOME/fitness/<name>.json"`:
   `{"name": "<name>", "target": "<goal>", "dimensions": [{"name": "speed", "weight": 0.5},
   {"name": "accuracy", "weight": 0.5}]}`. Weights must be positive and sum to 1.
2. Measure each dimension with real tool output and map it to a score in [0, 1], using a rule
   written into the spec's `target` so every run scores the same way.
3. Score with the helper. Never compute a weighted sum in prose:

   ```bash
   python scripts/fitness.py "$HERMES_HOME/fitness/latency.json" '{"speed": 0.8, "accuracy": 0.6}' --log "$HERMES_HOME/fitness/latency.jsonl"
   ```

4. Report `score` and `delta` (change since the last logged run). A falling score is a new
   hypothesis for loop 1.
5. To track it over time, `cronjob_manage(action="create", schedule="every monday 9am",
   prompt="Measure and score the latency fitness spec", skills=["self-improvement-loops"])`.

## Pitfalls

- Don't create `goals.json`, `learnings.json` or `governance_audit.json` files. Kanban, skills,
  memory and the session DB already hold that state and are what future sessions read.
- Don't hardcode `~/.hermes`: profiles live elsewhere. Use `$HERMES_HOME`.
- `approvals.deny` and fail-closed hooks can block the user's own work; test with
  `hermes approvals test` and `hermes hooks test` before telling them it's done.
- Shell hooks need one-time consent (recorded in the profile's `shell-hooks-allowlist.json`);
  a hook that never fires has usually not been accepted yet — run `hermes hooks doctor`.
- `fitness.py` rejects specs whose weights don't sum to 1 and scores that are missing a
  dimension or out of range. Fix the input; don't bypass the check.

## Verification

- Hypothesis loop: `kanban_show(task_id)` shows the experiment comments and the verdict metadata.
- Lessons: `skill_view` on the patched skill shows the new pitfall.
- Guardrails: `hermes approvals test -- <command>` returns the intended exit code.
- Fitness: the JSONL history gains one line per run, and the second run reports a non-null `delta`.
