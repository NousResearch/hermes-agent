# Canonical Response Policy Resolver Pattern

Use when creating a single canonical authority that maps canonical intent → processing policy, without classifying, generating responses, or executing actions.

## Context

After consolidating multiple parallel intent classifiers into one canonical authority, the next step is to create a resolver that decides HOW each intent should be processed. The resolver's sole responsibility is policy resolution — it does nothing else.

## Architecture position

```
Canonical Intent Classifier
    ↓
Canonical Response Policy Resolver  ← THIS
    ↓
Existing Routers (consume policy, do not create it)
```

## Policy contract output

The resolver produces a frozen policy object:

```javascript
{
  response_class: 'A'|'B'|'C'|'D',
  provider_required: boolean,
  retrieval_required: boolean,
  evidence_required: boolean,
  governance_required: boolean,
  execution_allowed: boolean,
  approval_required: boolean,
  deterministic_allowed: boolean,
  answer_builder: string|null,
  reason_codes: string[]
}
```

## Response classes

| Class | Name | Provider | Retrieval | Evidence | Governance | Execution | Approval |
|-------|------|----------|-----------|----------|------------|-----------|----------|
| A | Deterministic | No | No | No | No | No | No |
| B | Provider-enriched | Yes | Yes | Yes | No | No | No |
| C | Governed preparation | Yes | Yes | Yes | Yes | No | Maybe |
| D | Governed execution | No | Yes | Yes | Yes | No | Yes |

## Mapping intent → policy (19-intent example)

| Intent | Class | Rationale |
|--------|-------|-----------|
| identity | A | Deterministic sufficient |
| capability_inquiry | A | Built-in inventory |
| memory_query | B | Provider synthesizes past context |
| diagnostic | B | Provider analyzes error patterns |
| business_analysis | B | Provider synthesizes data |
| mutating | D | Execution requires approval |
| campaign_action | C | Governed — provider prepares, approval gates |
| unauthorized_action | A | Deterministic blocked |
| unknown | B | Provider fallback |

## Low-confidence upgrade

When the classifier reports LOW confidence for a Class A intent, upgrade to Class B:

```javascript
if (context.confidence === 'LOW' && policy.response_class === 'A') {
  return makePolicy('B', {
    provider_required: true,
    reason_codes: [...policy.reason_codes, 'low_confidence_upgrade_to_provider'],
  });
}
```

Do NOT double-upgrade (B/C/D with LOW confidence stays as-is).

## Legacy adapters

```javascript
policyToNaturalActionIntentClass(policy) → 'ANSWER'|'BLOCKED'|'APPROVAL_REQUIRED'|'GOVERNED_PLAN'
policyToSafetyClass(policy) → 'READ_ONLY'|'APPROVAL_REQUIRED'|'GOVERNED_PLAN'|'BLOCKED'
```

## What the resolver must NOT do

- Classify requests
- Generate responses
- Execute actions
- Modify routers
- Store state — pure/stateless only

## Pitfalls

- **ordering**: Do not implement the resolver before consolidating classifiers. Without canonical inputs it becomes another authority layer.
- **scope creep**: Resist adding routing logic or execution decisions.
- **approval semantics**: Class C with `approval_required: true` → APPROVAL_REQUIRED. Class C with `approval_required: false` → GOVERNED_PLAN.
- **crawl special case**: Non-private crawls = GOVERNED_PLAN (Class C, no approval). Private/authenticated crawls blocked at router level.
