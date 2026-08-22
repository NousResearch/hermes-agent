# Retrieval & Evidence Context Builder Pattern

Use when building a structured evidence pipeline that supplies downstream consumers (provider, answer composer) with verified runtime knowledge rather than relying on model knowledge alone.

## Context

After the canonical intent classifier, policy resolver, and router unification are in place, the next layer is the retrieval/evidence pipeline. The builder assembles a structured evidence dossier with explicit categories — facts remain facts, unknowns remain unknowns, conflicts stay explicit.

## Architecture position

```
Canonical Intent Classifier
    ↓
Canonical Response Policy Resolver
    ↓
Canonical Retrieval & Evidence Context Builder  ← THIS
    ↓
Provider Bridge / Answer Composer (future)
```

## Evidence dossier output

```javascript
{
  schema: 'cloe_evidence_dossier_v1',
  generated_at: '<ISO timestamp>',
  request: { intent, policy },
  retrieval_plan: { required_categories: [...] },
  evidence: {
    by_category: {
      FACT: [{ key, value, source, provenance, confidence }],
      UNKNOWN: [{ key, description, confidence }],
    },
    items: [...],  // flattened
    total: 42,
    categories_present: { FACT: 10, UNKNOWN: 2 }
  },
  summary: { fact: 10, unknown: 2 },
  provenance: { sources_consulted: [...], evidence_item_count: 42 },
  governance: { read_only: true, no_provider_calls: true, no_execution: true },
  annotations: {
    unknowns: [{ key, description }],
    stales: [{ key, description }],
    conflicts: [{ key, description }]
  }
}
```

## Nine evidence categories

| Category | Meaning | Example |
|----------|---------|---------|
| FACT | Verified runtime fact | process.node_version |
| SOURCE | Source of the evidence | Filesystem path, API |
| EVIDENCE | Proven operator evidence | Trial verdict, receipt |
| UNKNOWN | Data is unavailable | Registry file not found |
| CONFLICT | Evidence conflicts | Two sources disagree |
| STALE | Data may be outdated | WIRED but untested capability |
| PERMISSION | Access permission | safety.read_only: true |
| GOVERNANCE | Governance constraint | security.constraints: 5 |
| ASSUMPTION | Documented assumption | Inferred from available data |

## Retrieval planning

Determine which evidence categories to collect based on the policy:

```javascript
function planEvidenceRetrieval(policy) {
  const categories = ['FACT', 'SOURCE'];
  if (policy.retrieval_required) categories.push('EVIDENCE');
  if (policy.evidence_required) categories.push('EVIDENCE', 'CONFLICT', 'STALE');
  if (policy.governance_required) categories.push('GOVERNANCE', 'PERMISSION');
  categories.push('UNKNOWN', 'ASSUMPTION');
  return [...new Set(categories)];
}
```

## Evidence item factory

Every item must have traceable provenance:

```javascript
function fact(category, key, value, source, metadata = {}) {
  return {
    category, key, value, source,
    provenance: metadata.provenance || source,
    confidence: metadata.confidence || 'HIGH',
    collected_at: new Date().toISOString(),
    ...metadata,
  };
}
```

## Common evidence sources

| Source | Provides |
|--------|----------|
| aggregateNativeIntrospection() | Runtime facts, governance, safety, route registry |
| Canonical capability registry (JSON) | Capability count, taxonomy, state distribution |
| reviewBackgroundContinuity() | Continuity verdict, next action |
| Operator trial verdicts | Trial results, gates, case counts |

## Rules

- Every item retains its origin (source + provenance mandatory)
- Unknowns never become facts
- Conflicting evidence stays explicit (two sources → CONFLICT)
- The builder never calls the provider
- The builder never formats responses

## Pitfalls

- **fs in ESM**: Use readFileSync/existsSync/readdirSync from node:fs. Do NOT use createRequire or dynamic import() for sync ops.
- **Missing registry**: If the canonical capability registry file doesn't exist, handle with UNKNOWN entries not crashes.
- **Introspection errors**: aggregateNativeIntrospection() can throw. Wrap in try/catch.
- **Trial dir not found**: Operator trial directories may not exist. Return null, don't crash.
- **Circular deps**: Keep the builder as a leaf module — it should not be imported by routers or response builders.
