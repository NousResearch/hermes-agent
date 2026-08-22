# Retrieval → Provider Bridge Integration Pattern

Use when verifying that a newly-built component (retrieval builder, evidence dossier, etc.) is actually consumed by the existing provider pipeline, and wiring it in when it's not.

## Context

After building the Retrieval & Evidence Context Builder (or any new downstream component), the component may exist in the codebase but never be called during real runtime execution. This pattern covers:

1. **Audit** — trace whether the component is actually consumed
2. **Wiring** — inject the component's output into the existing provider bridge
3. **Verify** — confirm end-to-end that the output reaches the provider payload

## Audit: tracing the provider path

The canonical provider bridge is `buildBrainAskResponse` in `readonly-operator-cli-client.js`.

### Step 1: Find the bridge

Search for `buildBrainAskResponse` in `src/services/readonly-operator-cli-client.js`. It has callers in `readonly-conversation-router.js` and `server.js`.

### Step 2: Identify context injection points

`buildBrainAskResponse` builds a `cognitiveContextPack` via `buildCognitiveContextPack()`. This pack has an `available_items` array — the standard mechanism for injecting context into the provider payload. Items in this array are included in the system prompt.

Other injection points exist:
- `runtimeSnapshot` — runtime facts
- `cognitiveContextPack.compact_summary` — human-readable summary
- `cognitiveContextPack.intent_tags` — tags for the answer planner

### Step 3: Search for your component

```bash
grep -n "buildEvidenceDossier\|retrieval-context-builder\|evidence_dossier" src/services/readonly-operator-cli-client.js
```

If no matches, the component is NOT consumed by the provider bridge.

## Wiring: inject via cognitiveContextPack.available_items

```javascript
import { buildEvidenceDossier } from '../cognitive/retrieval-context-builder.js';
import { classifyCanonicalIntent } from '../cognitive/canonical-intent-classifier.js';
import { resolveResponsePolicy } from '../cognitive/canonical-response-policy-resolver.js';

// Inside buildBrainAskResponse, AFTER cognitiveContextPack is built:
try {
  const canonical = classifyCanonicalIntent(promptText);
  const policy = resolveResponsePolicy(canonical.intent, {
    confidence: canonical.confidence,
    subkind: canonical.subkind
  });
  const dossier = buildEvidenceDossier({
    intent: canonical.intent,
    policy,
    introspection: runtimeSnapshot,
  });
  if (dossier && dossier.evidence && dossier.evidence.total > 0) {
    cognitiveContextPack.available_items.push({
      kind: 'canonical_evidence_dossier',
      name: 'canonical_evidence_dossier_v1',
      source: 'retrieval-context-builder',
      available: true,
      read_only: true,
      safety_flags: ['read_only', 'no_network', 'no_write', 'no_execute'],
      description: `Canonical evidence dossier: ${dossier.evidence.total} items`,
      metadata: {
        by_category: dossier.evidence.categories_present,
        provenance: dossier.provenance,
        items_preview: dossier.evidence.items
          .filter(i => i.category === 'FACT')
          .slice(0, 15)
          .map(i => ({ key: i.key, value: String(i.value).slice(0, 120), source: i.source })),
        annotations: dossier.annotations,
      }
    });
    cognitiveContextPack.compact_summary += ` | Canonical evidence: ${dossier.evidence.total} items`;
  }
} catch {
  // Non-blocking — evidence failure must not break the provider call
}
```

## Rules for safe injection

| Rule | Why |
|------|-----|
| Wrap in try/catch | Provider call must not fail if evidence collection fails |
| Use existing injection mechanisms | `available_items`, `compact_summary` — no new parallel paths |
| Verify the module loads | `await import('...')` in test catches broken imports |
| Keep items read_only | Evidence is for provider consumption, not mutation |
| Add safety_flags | `['read_only', 'no_network', 'no_write', 'no_execute']` |
| Truncate long values | Use `.slice(0, 120)` on previews to avoid token overflow |

## Verify end-to-end

```javascript
// Module loads
test('provider module loads', async () => {
  const mod = await import('../src/services/readonly-operator-cli-client.js');
  assert.ok(mod);
});

// Injection works
test('evidence dossier items push into cognitiveContextPack', () => {
  const canonical = classifyCanonicalIntent(prompt);
  const policy = resolveResponsePolicy(canonical.intent);
  const dossier = buildEvidenceDossier({ intent: canonical.intent, policy });
  const items = [{
    kind: 'canonical_evidence_dossier',
    metadata: { by_category: dossier.evidence.categories_present }
  }];
  assert.ok(items[0].metadata.by_category.FACT > 0);
});

// Categories preserved
test('categories preserved', () => {
  const dossier = buildEvidenceDossier({ intent: 'test', policy: {} });
  assert.ok('STALE' in dossier.evidence.categories_present);
  assert.ok('CONFLICT' in dossier.evidence.categories_present);
});
```

## Pitfalls

- **Circular imports**: Provider bridge imports retrieval builder, NOT vice versa.
- **Static capabilities list**: The bridge may use a separate `STATIC_CAPABILITIES` list. Don't remove it — the dossier is additive.
- **Non-blocking**: If dossier build throws, provider must still work. Always try/catch.
- **Double classification**: `buildBrainAskResponse` already has `classifyIntent()`. The canonical classifier adds a second classification. Both coexist until migration complete.
- **Runtime snapshot reuse**: `buildBrainAskResponse` builds a `runtimeSnapshot`. Pass it as `preloadedIntrospection` to `buildEvidenceDossier` to avoid duplicate introspection work.
