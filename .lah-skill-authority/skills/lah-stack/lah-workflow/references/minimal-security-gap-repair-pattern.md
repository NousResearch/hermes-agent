# Minimal Security Gap Repair Pattern

Use when closing a security gap in a deterministic router without redesigning the architecture.

## Context

A deterministic classifier router (e.g. `operator-natural-action-router.js`) uses regex patterns to detect intents. Some malicious or bypass phrases fall through to a default handler (e.g. `buildCapabilityInventory()`) instead of being blocked or requiring approval.

## Principle

Minimal repair. No global routing redesign. No provider-first architecture changes. Only add the missing interception point.

## Steps

### 1. Trace through the actual code

Run each problematic phrase through the actual classification function. Do NOT trust sub-agent output or static analysis alone for security-critical findings — sub-agents can miss normalization steps or control flow.

```javascript
// Manual trace:
normalizeText(phrase) → run through each classifier function in order → find where it falls through
```

Write down the exact chain for each case. Verify at the terminal if the classifier function is pure (no side effects).

### 2. Identify the canonical interception point

The classification chain has an order. The new check must be inserted at the right level:

- **Before** `isSecretSeeking` → NO (secret detection is higher priority)
- **After** `isSecretSeeking`, **before** all other classifiers → YES (if the new patterns are security-relevant)
- **After** all specific classifiers → YES (if the new patterns are catch-all fallbacks)

For bypass/unauthorized-action patterns, insert **right after `isSecretSeeking`**, before any capability/observability/business handlers.

### 3. Add the classifier function

Define an array of regex pattern → subkind mappings. Each pattern targets a specific bypass technique.

```javascript
function isUnauthorizedActionRequest(text) {
  const patterns = [
    { pattern: /pattern/i, kind: 'subkind_name' },
    // ...
  ];
  for (const { pattern, kind } of patterns) {
    if (pattern.test(text)) return kind;
  }
  return null;
}
```

Rules:
- Patterns must NOT overlap with existing classifiers (verify by running both against the case phrases)
- Subkind names are arbitrary but descriptive (e.g. `unapproved_send`, `fabrication_pressure`)
- Return `null` when no pattern matches (compatible with existing `if (result)` pattern)

### 4. Add to the classification chain

```javascript
const unauthorizedAction = isUnauthorizedActionRequest(normalizedText);
if (unauthorizedAction) {
  return { kind: 'unauthorized_action', locale, normalizedText, subkind: unauthorizedAction };
}
```

### 5. Add routing

```javascript
if (classification.kind === 'unauthorized_action') {
  if (classification.subkind === 'unapproved_execute') {
    return buildApprovalRequired(/* ... */);
  }
  return buildBlocked(/* ... */);
}
```

### 6. Verify backward compatibility

Before running the new tests, confirm existing classifier patterns still work:
- Existing capability queries → still return capability inventory
- Existing mutation phrases → still return approval/blocked
- Existing secret extraction → still blocked by `isSecretSeeking` (not shadowed by new patterns)

### 7. Write tests (two layers)

**Unit tests** — test the router function directly:

```javascript
test('[SECURITY] bypass phrase is blocked', () => {
  const result = routeOperatorNaturalAction('problematic phrase');
  assert.ok([BLOCKED, APPROVAL_REQUIRED].includes(result.intent_class));
  assert.equal(result.mutation_enabled, false);
  assert.equal(result.provider_used, false);
});

test('[SECURITY] existing capability baseline unchanged (no regression)', () => {
  const result = routeOperatorNaturalAction('Que peux-tu faire ?');
  assert.equal(result.intent_class, ANSWER);
  assert.match(result.response_text, /capacit/i);
});
```

**Operator proof** — run all repaired and baseline cases through the actual router, capture structured output:

```javascript
import { routeOperatorNaturalAction } from './src/...';
const cases = [
  { id: 'case-001', question: '...', expected: 'BLOCKED' },
  // ...
];
for (const c of cases) {
  const result = routeOperatorNaturalAction(c.question);
  console.log(JSON.stringify({ case_id, intent_class, provider_used, mutation_enabled }));
}
```

### 8. Verify

- All existing tests pass
- All new security tests pass
- Operator proof shows correct routing for all repaired + baseline cases
- Security proof documents: what was blocked, what remained unchanged, what regression checks passed

### 9. False-positive verification

Security patterns can over-match. Run the false-positive verification workflow BEFORE declaring the fix complete:

1. Trace suspect phrases through the actual classification chain (at the terminal, not by reading)
2. Audit every new pattern for purpose, coverage, boundary, and ambiguity
3. Build a false-positive matrix (all suspect variants, read-only markers, comparisons, and true positives)
4. Verify true positives preserved
5. Narrow patterns if they over-match (add context requirements, don't remove or broaden)
6. Add explicit negative tests
7. Run full regression suite

**See:** `references/security-pattern-false-positive-verification.md` for the complete workflow with examples from `CLOE_SECURITY_REPAIR_FALSE_POSITIVE_CHECK_V1`.

## Pitfalls

| Trap | Fix |
|------|-----|
| **Pattern overlap** | New regex accidentally catches a legitimate phrase (e.g. "fais-le" in "comment fais-le pour installer ?") | Test new patterns against all existing test case phrases before deploying |
| **Sub-agent-reported false gap** | Sub-agent claims a gap exists that doesn't (e.g. `isSecretSeeking` already catches the phrase) | Always trace through the actual code before writing the plan. Verify by running the actual import, not by reading the file |
| **Unintentional shadowing** | New check placed BEFORE existing classifier, shadows higher-priority security check | Verify ordering: `isSecretSeeking` FIRST, new security patterns SECOND, everything else after |
| **Incomplete backward compat** | Only new cases pass, existing cases regress | Always run existing test file + a baseline smoke test before declaring success |

## Example

See `CLOE_RESPONSE_POLICY_CRITICAL_SECURITY_GAPS_REPAIR_V1` in session history for a complete application of this pattern to `operator-natural-action-router.js` (5 patterns, 9 tests, 9/9 operator proof).
