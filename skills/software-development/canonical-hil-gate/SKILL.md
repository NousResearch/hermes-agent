---
name: canonical-hil-gate
description: Validate human-in-the-loop authority decisions through a canonical event path instead of YAML, dashboard text, or chat fragments. Use when reviewing HIL, approvals, Telegram/web/CLI decisions, authority boundaries, self-improvement approval, or governance flows.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hil, governance, authority, approval, autonomy]
    related_skills: [run-scoped-causality-gate, autonomy-verb-proof-gate]
---

# Canonical HIL Gate

## Purpose

Prevent projected or informal human approval from becoming authority. Notification is transport; canonical decision events are authority.

## Canonical decision requirements

A valid HIL decision must include:

- exact proposal/capability/action id;
- decision: approve/reject/defer/narrow/suppress/etc.;
- approver identity or session;
- channel: CLI/web/Telegram parser that emits the same normalized event shape;
- granted authority and constraints;
- timestamp;
- append-only event id;
- schema validation;
- decision application record when executed.

## Reject as approval

Do not accept:

- raw YAML edited by an agent;
- dashboard labels;
- Telegram text that was not parsed into canonical event;
- `proposal_id: auto`;
- broad approval without target id;
- approval records created by the same process that needs approval;
- fixture/human_fixture decisions outside explicit test mode.

## Review output

```text
Claimed approval:
Canonical store/event:
Target id:
Authority granted:
Constraints:
Decision applied event:
Verdict: canonical | notification-only | projection-only | fixture-only | invalid
```

## Pitfall

A HIL notification can be real and still not be approval. Keep transport, normalization, authority, and execution as separate states.
