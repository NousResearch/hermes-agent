# Wayfinder Handoff Profile

Produce an execution packet only after product intent is stable enough.

## Triggers

build plan, engineering readiness, execution packet, wayfinder handoff, implementation packet.

## Ownership split

| Neon Genie owns | Wayfinder owns |
|-----------------|----------------|
| What should be built and why | Work decomposition |
| Target user and blocked transition | Dependency sequence |
| Product boundary and system behavior | Milestones and eng validation |
| Success criteria and proof requirements | Implementation status |

## Required fields

product authority, version, objective, non-goals, target outcome, acceptance criteria, workstreams, dependencies, constraints, canonical interfaces, artifacts, validation gates, regression risks, deferred scope, and change control.

## Change control (Gate H)

```yaml
product_intent_changes_require_neon_genie_review: true
```

Any proposed change to product intent returns to Neon Genie as a change request. The handoff packet **must not** rewrite intent — see `evals/cases/wayfinder-change-control.json`.

## Outputs

- `WayfinderExecutionPacket`
- `NeonGenieRunReceipt`

Schema: `schemas/wayfinder-execution-packet.schema.json`

## Authority

Handoff is still **advisory**. It does not authorize spend, deploy, or repo mutation. Wayfinder runtime is optional; absence never blocks emitting a local packet.
