# Canonical Intent Classifier Consolidation Pattern

Use when consolidating multiple parallel intent classifiers (from different code paths) into a single canonical authority while preserving backward compatibility.

## Context

Multiple entry points (TUI, Gateway, ACP client, front router) each have their own intent classifier with:
- Different intent taxonomies
- Different normalization functions
- Different prioritization/ordering
- Different output formats

This pattern creates ONE canonical classifier, then uses adapter functions to map back to legacy formats so existing consumers don't break.

## Steps

### 1. Inventory existing classifiers

List every intent classifier in the codebase, its file, its consumed entry points, and its intent taxonomy.

| File | Consumer | Intents |
|------|----------|---------|
| `classifier-A.js` | Entry point X | identity, status, ... |
| `classifier-B.js` | Entry point Y | capabilities, gaps, ... |

### 2. Define canonical taxonomy

Create a superset taxonomy that subsumes all existing intents. Each canonical intent maps to one or more legacy intents.

19 intents is typical for an operator-assistant system:
- identity, status, capability_inquiry, stack_observability, memory_query, diagnostic, tracking, business_analysis, morning_routine, system_analysis, campaign_action, mutating, governance_question, provider_opt_in, action_preparation, unauthorized_action, crawl_request, skill_install, unknown

### 3. Security interceptors FIRST

Security-related patterns (secret extraction detection, unauthorized action detection) must run BEFORE intent classification. They are pre-classification hooks, not intents.

```javascript
if (isSecretSeeking(text)) {
  return { intent: UNAUTHORIZED_ACTION, subkind: 'secret_extraction' };
}
```

### 4. Priority ordering

Order intent checks from most specific to most general. A pattern that matches many things (like "analyse") should come after a pattern that matches something specific (like "analyse le repository").

**Specific-before-general rule:**
- skill_install → capability_inquiry (because "install skill" contains "skill")
- system_analysis → status (because "état du système" contains "état")
- stack_observability → business_analysis (because "repository" is more specific than "analyse")
- campaign_action → business_analysis (because "campagne" is more specific than general analysis)

### 5. Legacy adapter map

Create a mapping from each canonical intent back to the legacy format each consumer expects.

```javascript
const LEGACY_INTENT_MAP = {
  canonical_intent: {
    consumer_a: 'legacy_intent_name',
    consumer_b: 'legacy_intent_other_name'
  }
};
```

### 6. Exported API

The canonical classifier should export:
- `classifyCanonicalIntent(text)` → `{ intent, confidence, matched_pattern, locale, subkind }`
- `CANONICAL_INTENTS` — frozen object of all intent constants
- `isSecretSeeking(text)` / `isUnauthorizedActionRequest(text)` — security interceptors (importable)
- `LEGACY_INTENT_MAP` — for adapters/wrappers

### 7. Tests

Test every canonical intent with a representative query. Test security interceptors. Test edge cases. Verify all existing consumer tests still pass (backward compat).

## What NOT to do
- Do NOT modify existing classifier files during consolidation (preserve backward compat)
- Do NOT merge the V5 decision classifier — it's additive/subdomain, not general-purpose
- Do NOT remove security interceptors from their original files — the canonical exports them too
- Do NOT reorder the priority without checking all edge cases
- Do NOT add "analyse" to business_analysis without ensuring stack_observability and system_analysis come first
