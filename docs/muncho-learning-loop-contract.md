# Muncho Operational Learning Loop Contract

Status: report-only implementation contract.

This contract adds Muncho-specific operational learning without replacing or
weakening Hermes' built-in self-improvement loop.

## Boundary

Hermes self-improvement remains intact:

- `agent/background_review.py` may still run after turns.
- The standard `💾 Self-improvement review` may still update memory/skills.
- Mainstream Hermes skill-management behavior is not blocked by this contract.
- This contract does not alter gateway runtime behavior or the model prompt.

Muncho operational learning is a separate Adventico/SkyVision overlay:

- It creates reviewable learning packet drafts from explicit evidence.
- It stores private case detail outside the public repository by default.
- It may recommend future promotion, but does not promote automatically.
- It must not introduce keyword-router or classifier authority.

## Learning Packet

A learning packet is a draft record for one operational case. It contains:

- case id
- exact source/evidence refs
- requester and involved people
- business area
- problem and expected action
- actual Muncho action
- what went wrong / what worked
- final status
- lesson candidate
- promotion recommendation
- confidence and missing evidence
- safety boundary flags

The safety flags must remain:

- `report_only=true`
- `runtime_behavior_change=false`
- `durable_promotion_performed=false`
- `standard_hermes_self_improvement_preserved=true`
- `keyword_router_authority=false`

## Knowledge Classes

Allowed review metadata classes:

- `case_note_only`
- `business_knowledge`
- `operational_process`
- `team_routing_knowledge`
- `it_access_or_infra`
- `customer_support_pattern`
- `tooling_gap`
- `product_process_improvement`
- `skill_update_candidate`
- `reject`

These classes are not runtime routers. They are only review labels for humans
and later promotion gates.

## Promotion

Auto-promotion is forbidden for:

- team/channel routing
- access maps
- deploy/runtime behavior
- Cloud SQL write contracts
- customer/provider processes
- permissions
- production operations

Hermes self-improvement remains allowed for:

- reusable workflow phrasing
- report formatting
- evidence collection hygiene
- runbook clarity
- local skill hygiene

Durable promotion requires an explicit later approval gate.

## Storage

Public repository artifacts contain only tooling, schemas, tests, and this
contract. Real business cases should be generated into private paths such as:

```text
~/.hermes/private_reports/muncho_learning_loop/<date-or-run-id>/
```

Public-safe digests must omit raw case transcripts, customer/provider/private
identifiers, and secrets.

## First Gate

The first implementation gate is:

1. add report-only packet tooling;
2. generate private pilot packets for four real cases;
3. generate private and public-safe digests;
4. do not change runtime behavior;
5. do not update durable knowledge;
6. do not disable standard Hermes self-improvement.
