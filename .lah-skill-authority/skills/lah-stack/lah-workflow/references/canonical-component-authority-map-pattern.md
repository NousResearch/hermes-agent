# Canonical Component Authority Map Pattern

Use when determining the authoritative component for every critical responsibility in a layered architecture with parallel components.

## When to Use

After completing a capability inventory (M2 pattern), before designing component consolidation. Answers: "which component is the authority for each responsibility, and what should happen to the others?"

## Steps

### 1. Identify the components to audit

List every component that participates in the critical responsibility. Sources:
- Capability registry (from capability inventory)
- CodeGraph exploration of entry points and callers
- Route/endpoint enumeration (server.js, route directories)

### 2. For each component, determine:

| Field | Source |
|-------|--------|
| Purpose | File header comment or module-level documentation |
| Runtime usage | Callers (from CodeGraph or grep) |
| Authority | Is this the intended/canonical component for this task? |
| Callers | Who imports or calls this component |
| Callees | What does this component import or call |
| Duplication | Does another component serve the same purpose |
| Replacement candidate | If this should be archived, what replaces it |
| Recommended status | KEEP / MERGE / REPLACE / ARCHIVE / DELETE / INVESTIGATE |

### 3. Build caller/callee trail

For each component, trace:
- **Who calls it** — `codegraph_node(symbol).importers` or `grep -rl 'from.*component' src/`
- **Who it calls** — `codegraph_node(symbol).callees` or read imports in the file header
- **Where it's mounted** — server.js route mounting or entry point registration

Example from CLOE routing audit:

```
operator-natural-action-router.js
  Callers: openclaw-tui.js, openclaw-aionui.js, openclaw-acp-client.js
  Callees: aggregateNativeIntrospection(), formatStackObservabilityBrief(), reviewBackgroundContinuity(), isStackObservabilityPrompt()

readonly-conversation-router.js
  Callers: Gateway websocket handler
  Callees: classifyReadonlyConversationIntent(), buildBrainAskResponse(), aggregateNativeIntrospection(), buildAllOperatorQuerySurfaces()
```

### 4. Map against target pipeline

Lay the current components onto the target architecture pipeline. Identify:

- **✅ Correct** — component that fits the target architecture (e.g. a single provider bridge)
- **❌ Missing** — component that the target pipeline requires but doesn't exist yet (e.g. no Front Router)
- **⚠️ Duplicate** — multiple components serving the same role (e.g. 3+ intent classifiers)
- **🔄 Partial** — component exists but lacks essential features (e.g. context builder exists but doesn't produce evidence dossiers)

### 5. Determine status per component

| Status | Meaning | Example |
|--------|---------|---------|
| KEEP | Correctly placed, keep as-is | provider bridge |
| MERGE | Duplicate that should fold into canonical component | parallel intent classifiers |
| REPLACE | Inadequate — needs replacement | response-formatter → Answer Composer |
| ARCHIVE | Functionally superseded, keep for reference | V3 pipeline (if V5 replaces it) |
| DELETE | Dead code, no callers, no tests | orphan module |
| INVESTIGATE | Cannot determine from available evidence | module with no tests and no callers |

### 6. Build convergence roadmap

Order the changes by dependency:

**Phase A — Foundation**: Components that everything else depends on (classifier consolidation)
**Phase B — Integration**: Components that use the foundation (router unification)
**Phase C — New components**: Components that don't exist yet (Answer Composer)
**Phase D — Polish**: Components that can wait (extend existing to add features)

Each phase must specify:
1. What changes
2. Prerequisites (phases that must complete first)
3. Risk level
4. Effort estimate (relative: small/medium/large)

### 7. Write deliverables

| File | Content |
|------|---------|
| `docs/superpowers/plans/canonical-routing-authority-map-v1.md` | Full authority map (component audit + duplicates + convergence roadmap) |
| `docs/superpowers/plans/CLOE_ROUTING_AUTHORITY_MAP_V1_OPERATOR_PACKET.md` | Operator packet (ready/blocked/remaining) |

### 8. Verify internal consistency

Cross-check:
- Every component has a status (not just listed)
- Every duplicate has a target (what it merges into or is replaced by)
- Convergence roadmap phases are dependency-ordered (no phase N+1 that requires phase N+2's output)
- All findings trace to specific CodeGraph observations (not general impressions)
