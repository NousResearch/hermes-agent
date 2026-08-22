# Canonical Router Unification Pattern

Use when updating existing runtime routers to consume canonical contracts (intent classifier + policy resolver) instead of performing their own classification and policy decisions.

## Context

After creating a canonical intent classifier (M4) and a canonical response policy resolver (M5), the existing routers still contain legacy classification and policy decision paths. This pattern describes how to make the canonical authorities the effective runtime path while preserving backward compatibility.

## Target flow

```
Front Router
    ↓
Canonical Intent Classifier
    ↓
Canonical Response Policy Resolver
    ↓
Existing Router (consumes policy, dispatches to builders)
    ↓
Provider Bridge / Governance
```

## Steps

### 1. Audit runtime entry points

Identify every code path that performs:
- Intent classification (regex matching, keyword detection)
- Policy decisions (should this be blocked? approved? governed?)
- Router selection (which builder to call)

Common entry points: TUI, AionUI, ACP client, Gateway, front router.

### 2. Security interceptors stay in routers

Security checks (secret detection, unauthorized action detection, mutation pre-checks) must run BEFORE the canonical pipeline. These are NOT general intent classification — they are gate checks that answer binary yes/no questions:

```
Security interceptor → if match → return blocked/approval directly
Mutation pre-check → if match → return approval/blocked directly
Canonical classifier + resolver → everything else
```

Keep these in the router:
- `isSecretSeeking(text)` — from canonical classifier
- `isUnauthorizedActionRequest(text)` — from canonical classifier
- `isBroadMutationRequest(text)` — local router function (mutation keywords)
- `isCampaignAction(text)` — local router function (campaign keywords)

### 3. Replace switch/case with canonical dispatch

Before (old pattern — router decides):
```javascript
switch (classification.kind) {
  case 'capabilities': return buildCapabilityInventory();
  case 'morning': return buildMorningRoutine();
  case 'gaps': return buildGapBrief();
  // ...
}
```

After (router consumes policy):
```javascript
const canonical = classifyCanonicalIntent(rawPrompt);
const policy = resolveResponsePolicy(canonical.intent, { confidence, subkind });

// Use policy's answer_builder to dispatch
if (policy.response_class === 'A' && policy.answer_builder) {
  switch (policy.answer_builder) {
    case 'buildCapabilityInventory': return buildCapabilityInventory(context);
    case 'buildBlocked': return buildBlocked(...);
    // ...
  }
}

// Class B/C/D — use legacy adapter
const legacyClass = policyToNaturalActionIntentClass(policy);
if (legacyClass === 'BLOCKED') return buildBlocked(...);
if (legacyClass === 'APPROVAL_REQUIRED') return buildApprovalRequired(...);
```

### 4. Legacy classifier stays exported

Keep the old `classifyOperatorNaturalActionPrompt()` function exported for backward compatibility. External consumers may still import it. Mark it as legacy in JSDoc but do not change its behavior.

### 5. Update intent-classifier.js to delegate

The secondary classifier (used by the cognitive front router) should become a thin wrapper:

```javascript
export function classifyCognitiveIntent(text) {
  const canonical = classifyCanonicalIntent(prompt);
  const legacyType = CANONICAL_TO_LEGACY[canonical.intent] || 'unknown';
  return makeIntent(legacyType, prompt, {
    provider_requested: canonical.intent === 'provider_opt_in',
    governance_overlay_required: isGovernanceSensitive,
    // ...
  });
}
```

## Patterns that need adjustment

When moving from a legacy router to the canonical pipeline, some patterns may not match correctly. Common fixes:

| Legacy pattern | Canonical fix |
|---------------|---------------|
| `isBroadMutationRequest` catches "restart docker" but `stack_observability` has "docker" | Keep mutation pre-check BEFORE canonical pipeline |
| `isStackObservabilityPrompt` catches "blockers" but canonical doesn't | Add "blockers" to stack_observability pattern |
| `isCrawlRequest` returns GOVERNED_PLAN but canonical returns APPROVAL_REQUIRED | Set `crawl_request.approval_required = false` in policy |
| `isSecretSeeking` and `isUnauthorizedActionRequest` duplicated locally | Remove local versions, import from canonical classifier |

## What NOT to do

- Do NOT remove the old `classifyOperatorNaturalActionPrompt` function (backward compat)
- Do NOT remove the response builders (buildBlocked, buildApprovalRequired, etc.)
- Do NOT modify the V5 decision classifier
- Do NOT change the Gateway readonly-conversation-router (it has its own provider path)
- Do NOT implement retrieval, answer composer, or provider redesign

## Pitfalls

- **duplicate export**: If the router defined isSecretSeeking/isUnauthorizedActionRequest locally AND the canonical classifier exports them, remove the local versions to avoid "already declared" errors.
- **mutation vs observation**: "restart docker" is a mutation (should be blocked). "docker ps" is observability. The canonical classifier can't distinguish these — keep `isBroadMutationRequest` as a mutation pre-check in the router.
- **campaign overlap**: "campaign" and "campagne" match BOTH `campaign_action` and `business_analysis` patterns. Keep campaign pre-check before canonical pipeline.
- **test expectations**: Some tests assert specific `intent_class` or `safety_class` values. After unification, verify these still match. Use `policyToNaturalActionIntentClass`/`policyToSafetyClass` adapters to produce the same values.
