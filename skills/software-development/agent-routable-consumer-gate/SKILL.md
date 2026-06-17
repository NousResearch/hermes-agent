---
name: agent-routable-consumer-gate
description: Require agent-routable/proposal/backlog labels to have real consumers that claim, execute, verify, and retire work. Use when reviewing attention queues, improvement backlogs, proposal_required, agent_routable, repair candidates, or autonomous work queues.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [queues, agents, work-routing, autonomy, anti-theater]
    related_skills: [autonomy-verb-proof-gate, behavioral-verifier-gate]
---

# Agent-Routable Consumer Gate

## Purpose

Stop queue labels from masquerading as autonomous handling. `agent_routable` means nothing unless an agent/runtime can consume the item and close it with evidence.

## Required consumer contract

For each routable item class, identify:

- consumer name/entrypoint;
- claim/lock mechanism;
- input schema;
- authority policy;
- action executor;
- verifier;
- retry/rollback behavior;
- terminal states;
- retirement/update record;
- monitoring/eval that detects stuck items.

## Required lifecycle

```text
queued
-> claimed
-> policy_checked
-> executed | deferred | rejected
-> verified
-> retired | updated | escalated
```

If the lifecycle is absent, the item is not agent-routable; it is merely labeled.

## Failure conditions

Fail when:

- queue has open items but no consumer;
- consumer exists but cannot execute the recommended next action;
- verifier is artifact-only;
- item remains open across runs without escalation;
- dashboard hides backlog because no new proposals were created this run.

## Review output

```text
Queue/item:
Claimed routing:
Consumer present:
Lifecycle state:
Verifier:
Stuck/retired/escalated:
Verdict: routable | label-only | stuck | needs HIL | invalid
```

## Pitfall

Do not count “0 new proposals this run” as an empty backlog. Existing unresolved items remain open work.
