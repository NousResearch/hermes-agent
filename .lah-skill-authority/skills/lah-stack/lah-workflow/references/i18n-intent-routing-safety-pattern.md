# I18n Intent Routing Safety Pattern

## Layer 0 — Normalization Pitfall (normalizeText destroys accents and apostrophes)

### Problem

The canonical classifier's `normalizeText()` function strips ALL non-`[a-z0-9\s]` characters:

```javascript
function normalizeText(value) {
  return value.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
}
```

This means:
- `é` (U+00E9) → **removed entirely** (not converted to 'e')
- `è`, `ê`, `ë` → all removed
- `'` (apostrophe) → removed
- `-` (hyphen) → removed

Example: `"exécute l'action approuvée act-123"` → `"excute laction approuvee act123"`

Note: `"exécute"` becomes `"excute"` — the accented `é` is simply GONE, leaving `ex` + `cute` with no `e` between them.

### Consequence: regex patterns using accented French characters NEVER match normalized text

Before the fix, the `execute_approved` patterns used `exécute` with a literal `é`. After normalization, `exécute` becomes `excute`. The regex expects `ex` + `é` + `cute` but gets `ex` + `cute`. **Zero matches.**

### Fix patterns

**Option A — Character class with optional accent** (when the accented letter may be at a position that gets joined):

```javascript
// ex[ée]?cute matches: exécute (raw), excute (normalized), execute (English)
/\b(ex[ée]?cute\s+...)\b/i
```

**Option B — Character class without optional** (when the ASCII form survives):

```javascript
// approuvée → approuve; approuv[eé] matches both
/\b(approuv[eé]\s+...)\b/i
```

**Option C — Optional apostrophe** (for elision like `l'action` → `laction`):

```javascript
// l[''\u2019]?action matches: l'action (raw), laction (normalized)
/\b(l[''\u2019]?action\b)/i
```

### Rule

Every French regex pattern in the canonical classifier MUST:
1. Use character classes `[eé]`, `[aà]`, `[cç]`, etc. for accented letters
2. Make the accent letter optional with `?` when its removal joins characters
3. Make apostrophes optional with `[''\u2019]?` for elision forms
4. Test against BOTH raw text and normalized text

## Problem

A deterministic router classifies operator prompts into intents and builds governed-action packets. Adding French (or any non-English) keywords to the pre-classifier's substring-matching functions (`includesAny`, `includesPhrase`) causes false positives: a French verb like "modifie" (imperative "modify") matches inside "sans la modifier" ("without modifying it") via substring overlap, incorrectly routing a read-only query as a mutation.

## Three-Layer Architecture

```
Layer 1 — Pre-classifier (operator-natural-action-router.js)
    uses: includesAny (substring matching)
    role: rough routing — ensure non-English prompts REACH the canonical pipeline
    risk: false positives from substring overlap
    solution: only add broad, multi-word phrases here; let canonical classifier handle single-word verbs

Layer 2 — Canonical classifier (canonical-intent-classifier.js)
    uses: regex word boundaries (\bverb\b)
    role: precise intent classification
    safety: \bmodifie\b does NOT match "modifier" because 'e'→'r' has no word boundary between
    rule: ALL single-word non-English verbs belong here, NOT in Layer 1

Layer 3 — Action detection (cloe-governed-action-packet.js)
    uses: regex (pattern matching)
    role: detect specific action type from raw text, or return null (fail-closed)
    rule: never default to a real action type (CAMPAIGN_PAUSE) — return null for unrecognized verbs
```

## Key Rules

### 1. Substring-matching (Layer 1) is dangerous for single-word verbs

`includesAny(text, ['modifie'])` matches inside `"sans la modifier"` because JavaScript's `String.includes()` checks character-by-character, not word-by-word.

**Safe patterns for Layer 1:** multi-word phrases only
```javascript
// SAFE: two-word phrases won't false-positive on unrelated text
'pause campagne', 'arrete campagne', 'relance campagne',
'mets en pause', 'en pause'

// ALSO SAFE: proper nouns / context-specific terms are unlikely to overlap
'campagne', 'campaign'
```

**Unsafe:** single-word French verbs
```javascript
// UNSAFE: 'modifie' matches 'modifier', 'supprime' matches 'supprimer'
'modifie', 'supprime', 'efface', 'publie', 'envoie'
```

### 2. Word-boundary regex (Layer 2) is safe for single-word verbs

```javascript
// SAFE: \bmodifie\b does NOT match "sans la modifier"
// because 'e'→'r' has no word boundary
/\b(delete|remove|...|modifie|change|update|modify)\b/i
```

Test: `/\bmodifie\b/.test("sans la modifier")` → **false** ✓

### 3. Action detection (Layer 3) must never default to a real action

```javascript
// UNSAFE: returns defaultType = CAMPAIGN_PAUSE when no keyword matches
function detectSpecificActionType(rawPrompt, defaultType) {
  if (pauseMatch) return 'CAMPAIGN_PAUSE';
  if (playMatch) return 'CAMPAIGN_PLAY';
  return defaultType; // ← silently produces a real mutation action
}

// SAFE: returns null when nothing matches → fail-closed
function detectSpecificActionType(rawPrompt) {
  if (pauseMatch) return 'CAMPAIGN_PAUSE';
  if (playMatch) return 'CAMPAIGN_PLAY';
  return null; // ← packet becomes INCOMPLETE, no mutation
}
```

### 4. Read-only guard for cross-lingual campaign queries

When the canonical classifier returns `campaign_action` (via `\bcampagne\b` or `\bcampaign\b`) but no specific action verb is detected, the request is read-only:

```javascript
// In the router, after canonical intent and policy computation:
if (canonical.intent === 'campaign_action' && !detectSpecificActionType(rawPrompt)) {
  return buildCapabilityInventory(context, locale); // → READ_ONLY
}
```

This handles:
- "Tell me about campaign 123" → no action detected → READ_ONLY
- "Que peux-tu me dire sur la campagne 123 ?" → no action detected → READ_ONLY
- "Analyse la campagne 123 sans la modifier" → no action detected → READ_ONLY
- "Pause campaign 123" → CAMPAIGN_PAUSE detected → APPROVAL_REQUIRED

## Concrete Example: Adding French Campaign Semantics

### Step 1 — Pre-classifier (Layer 1)

Add French phrases to `isCampaignAction()`:
```javascript
function isCampaignAction(text) {
  return includesAny(text, [
    'launch campaign', 'pause campaign', 'stop campaign',
    'create campaign',
    // French multi-word patterns (safe — no substring overlap risk)
    'pause campagne', 'arrete campagne', 'stop campagne',
    'relance campagne', 'reactive campagne', 'active campagne',
    'demarre campagne', 'lance campagne',
    'mets en pause', 'en pause',
    'campagne',      // broad trigger — routes to canonical pipeline
    'campaign',
  ]);
}
```

Do NOT add single-word French verbs here (Layer 1 can't safely distinguish "modifier" from "modifie").

### Step 2 — Canonical classifier (Layer 2)

Add word-boundary patterns that include the French verbs:
```javascript
campaign_action: [
  /\b(launch campaign|pause campaign|stop campaign|create campaign
    |resume campaign|play campaign|restart campaign)\b/i,
  /\bcampagne\b/i,   // ← catches any French campaign reference
  /\blance\s+(?:une\s+)?(?:campagne|action|campaign)\b/i,
],

mutating: [
  /\b(delete|remove|...|modifie|change|update|modify|dmarre|active|...)\b/i,
  //          ^^^^^^^  ^^^^^^ ^^^^^^ ^^^^^^ — word boundaries prevent overlap
],
```

### Step 3 — Action detection (Layer 3)

Extend the regex to include French action verbs, return null for unrecognized:
```javascript
function detectSpecificActionType(rawPrompt) {
  const lower = String(rawPrompt ?? '').toLowerCase();
  if (/(?:pause|stop|arr.te|mets\s+en\s+pause)/i.test(lower))
    return 'CAMPAIGN_PAUSE';
  if (/(?:resume|play|start|restart|relance|reprends?|r.active|active|d.marre|lancer)/i.test(lower))
    return 'CAMPAIGN_PLAY';
  if (/(?:create|cr.e|nouvelle?|nouveau|new)/i.test(lower))
    return 'CAMPAIGN_CREATE_PAUSED';
  return null; // ← fail-closed
}
```

## Testing Checklist

- [ ] English mutation: produces governed packet with correct action_type
- [ ] French mutation (accented and unaccented): produces governed packet with correct action_type
- [ ] English read-only campaign question: READ_ONLY, no governed_action
- [ ] French read-only campaign question: READ_ONLY, no governed_action
- [ ] Negated French mutation ("sans modifier"): READ_ONLY, no governed_action
- [ ] Ambiguous verb (change/update/modifie): governed packet with action_type=null, INCOMPLETE
- [ ] Missing target (EN and FR): INCOMPLETE with missing_fields
- [ ] Approval non-bypass: approved_by_human=false, approval_id=null
- [ ] Static source safety: no executeAction/executeGoverned/callTool/axios/child_process

## References

- `operator-natural-action-router.js` — Layer 1 pre-classifier + routing logic
- `canonical-intent-classifier.js` — Layer 2 word-boundary classification
- `cloe-governed-action-packet.js` — Layer 3 action detection + packet builder
- `canonical-intent-classifier-consolidation-pattern.md` — Consolidating parallel classifiers
- `canonical-router-unification-pattern.md` — Unifying router dispatch
