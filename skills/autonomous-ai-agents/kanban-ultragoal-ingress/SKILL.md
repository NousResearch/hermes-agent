---
name: kanban-ultragoal-ingress
description: Use when Chris says `ULTRAGOAL로 진행해` or equivalent; fail-closed into hermes-execution-routing's Kanban Ultragoal lane without falling back to Autopilot or generic Hermes direct.
version: 1.0.0
author: Hayase Yuuka
license: MIT
metadata:
  hermes:
    tags: [kanban, ultragoal, ingress, routing, execution-boundary]
    related_skills: [hermes-execution-routing, kanban-native-work-execution]
    requires_toolsets: [terminal, file]
---
# Kanban Ultragoal Ingress

Use this skill only as the thin natural-language ingress for explicit Ultragoal operator commands such as `ULTRAGOAL로 진행해`, `ultragoal로 구현해`, `BO-123 ultragoal로 진행`, or `이 parent ultragoal로 계속해`.

## Contract

This skill is **not** a second SSOT and not the executor itself.

1. Load and apply `hermes-execution-routing` first.
2. Record/respect the top-level lane as `Kanban Ultragoal`.
3. Preserve current runtime wire compatibility when needed: `routing_verdict=direct-kanban` may still be required by `kanban-ultragoal` even though the human lane is `Kanban Ultragoal`.
4. Keep Kanban as execution authority, Done Criteria, lifecycle, and audit SSOT.
5. Treat `.hermes/goal-runs/<id>/ledger.jsonl` as execution journal/proof only.
6. Confirm target card/parent, execution approval, allowed mutations, and forbidden side effects before calling `kanban-ultragoal`.
7. If any prerequisite is missing, fail closed with a blocker. Do **not** fall back to Autopilot, generic Hermes direct, free-floating Codex, or ordinary conversation.

## Execution shape

When prerequisites are present:

```text
natural phrase
→ kanban-ultragoal-ingress
→ hermes-execution-routing verdict: Kanban Ultragoal
→ current wire verdict if needed: direct-kanban
→ kanban-ultragoal pilot-check/run/status/resume/review-ready
→ verifier/cleanup proof
→ Kanban evidence
```

## Required preflight

Before mutation or run creation, verify:

- live Kanban authority for the target card/parent;
- `execution_approved=true` or equivalent current-turn Chris authorization;
- Done Criteria are present and durable;
- no forbidden side effect is needed: merge, deploy/live apply, gateway restart/reload, prod/customer/env/secret/provider mutation, external send;
- the requested lane is Ultragoal, not Autopilot.

## Blocker format

```text
Ultragoal ingress blocked
Reason: <missing target | missing Kanban authority | execution not approved | forbidden side effect | runtime unavailable>
Lane: Kanban Ultragoal
Wire compatibility: direct-kanban if current CLI requires it
No fallback: Autopilot/Hermes direct/Codex not used
Needed next: <exact approval/fact/action>
```

## Verification checklist

- [ ] `hermes-execution-routing` loaded before executor selection.
- [ ] Lane recorded as `Kanban Ultragoal`.
- [ ] Current wire value compatibility handled explicitly.
- [ ] Kanban authority re-read.
- [ ] Target card/parent resolved.
- [ ] Execution approval and side-effect boundaries checked.
- [ ] `kanban-ultragoal` used only after preflight.
- [ ] Missing prerequisites fail closed, not into Autopilot.
