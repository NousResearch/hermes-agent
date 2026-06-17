---
name: capability-actuator-gate
description: Prevent capability registries or specs from being treated as deployed capabilities without executable actuators and effect verification. Use when reviewing capability registries, self-improvement outputs, plugins, tools, shadow capabilities, or deployment claims.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [capabilities, deployment, actuators, self-improvement, anti-theater]
    related_skills: [behavioral-verifier-gate, autonomy-verb-proof-gate]
---

# Capability Actuator Gate

## Purpose

A capability is not deployed because it appears in a registry. Deployment requires an invocation path and verified effect.

## Deployment minimum

Do not call a capability deployed unless it has:

- executable entrypoint or callable tool;
- declared input/output contract;
- authority boundary and effect surface;
- run command or invocation example;
- independent effect verifier;
- rollback/disable path;
- at least one successful run trace on a non-trivial case;
- tests for boundary violations and failure modes.

## State vocabulary

Use precise states:

- `proposed`: idea/spec exists;
- `registered`: metadata exists;
- `implemented`: code/entrypoint exists;
- `invocable`: caller can execute it;
- `verified_shadow`: effect proven in bounded/shadow mode;
- `deployed`: used by the real runtime under policy;
- `retired`: disabled with rollback record.

A registry row alone is at most `registered`.

## Review output

```text
Capability claim:
Registry/spec present: yes/no
Actuator present: yes/no
Invocation path:
Effect verifier:
Run trace:
Rollback path:
Correct state:
```

## Anti-theater check

If the declared effect surface only writes registry/evaluation files, the capability is not built. It is documentation about a possible capability.
