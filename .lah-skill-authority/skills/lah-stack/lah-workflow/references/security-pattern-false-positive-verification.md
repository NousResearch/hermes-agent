# Security Pattern False-Positive Verification

Use AFTER adding new security detection patterns to a deterministic router. This reference documents how to verify the new patterns don't over-match legitimate requests.

## When to Use

Always. Any time you add new regex patterns or classifier branches to a security-critical routing path, run this verification BEFORE declaring the fix complete.

## Verification Steps

### Step 1: Trace exact matching for SUSPECT phrases

For any phrase that your intuition says might be a false positive, trace through the actual classification chain:

```
Question: "Analyse mes données."
→ normalizeText: "analyse mes donnees"
→ isSecretSeeking: ["token","secret"...] → NO MATCH
→ isUnauthorizedActionRequest: [...pattern list...]
    → /analyse\s+(?:mes|mon)\s+(?:donn[ée]es|historique)/i → MATCH (FALSE POSITIVE!)
→ kind: 'unauthorized_action' → BLOCKED ❌
```

Do this at the terminal with the actual import, not by reading the file. Normalization steps (stripDiacritics, toLowerCase, non-alphanumeric replacement) can change how patterns match in ways that aren't obvious from the regex alone.

### Step 2: Audit every new regex pattern

For EACH pattern in the new classifier function, document:

| Field | What to record |
|-------|----------------|
| **Purpose** | What specific threat does this pattern target? |
| **Intended coverage** | Examples that SHOULD match |
| **Clean boundary** | Examples that SHOULD NOT match (but are dangerously close) |
| **Overlap** | Does this pattern overlap with any existing classifier? Run both against the case set. |
| **Ambiguity** | Could a legitimate phrase match this pattern? If yes, document which one and whether it's acceptable. |

### Step 3: Build a false-positive matrix

Run at minimum the following categories of test requests:

- **Exact suspect phrase** — the phrase you're worried about (e.g. "Analyse mes données")
- **Near variants** — same grammar, different object ("Analyse mes campagnes", "Analyse les logs")
- **Read-only markers** — phrases explicitly saying "sans modifier", "sans rien changer", "uniquement les infos"
- **Comparison markers** — "Compare X et Y", "Différence entre X et Y"
- **All true positives** — every phrase that should be blocked (must remain blocked)

For each request, record:

| Column | Value |
|--------|-------|
| Request | The exact phrase |
| Detected intent | kind from classify function |
| Matched regex | Which pattern (or none) |
| Routing decision | BLOCKED / APPROVAL_REQUIRED / ANSWER / etc. |
| Expected policy | What it SHOULD be |
| Verdict | CORRECT / FALSE_POSITIVE / FALSE_NEGATIVE |

### Step 4: Verify true positives preserved

Re-run every case that the original security fix intended to protect. Use the same classification + routing function. Confirm:
- All bypass phrases still blocked
- All secret extraction still blocked
- All execution still requires approval
- All fabrication requests still blocked

### Step 5: If false positives exist — narrow, don't remove

**Rule:** When a pattern over-matches, narrow the regex to add context requirements. Do NOT:
- Remove the pattern entirely (this re-opens the gap)
- Add exception lists (brittle, impossible to maintain)
- Add a post-classification exemption check (adds complexity, separate from the detection logic)
- Broaden routing (changing blocked → allowed is wrong direction)

**Narrowing strategy:**

```
// BEFORE — too broad, matches any "analyse mes données"
/analyse\s+(?:mes|mon)\s+(?:donn[ée]es|historique)/i

// AFTER — requires context-exploitation prefix first
/(?:maintenant\s+que|puisque|comme\s+tu|vu\s+que).*analyse\s+(?:mes|mon)\s+(?:donn[ée]es|historique)/i
```

The narrowing principle: **make the pattern more specific about WHEN the threat applies, not about WHAT words it matches.** Add prefixes, suffixes, or structural requirements that distinguish the dangerous usage from the legitimate one.

### Step 6: Add explicit negative tests

Each false-positive scenario gets its own test asserting it is NOT blocked:

```javascript
test('[FP] "Analyse mes donnees" alone is NOT blocked (read-only analysis)', () => {
  const result = routeOperatorNaturalAction('Analyse mes données.');
  assert.equal(result.intent_class, ANSWER);  // not BLOCKED
  assert.equal(result.provider_used, false);
  assert.equal(result.mutation_enabled, false);
});
```

These tests are as important as the positive security tests. Without them, a future refactor could re-introduce the false positive without detection.

### Step 7: Full regression suite

Run: original tests + security repair tests + new negative tests + true-positive regression tests. All must pass.

## Pitfalls

| Trap | Fix |
|------|------|
| **Only tested true positives, not negatives** | The false positive is invisible in CI because no test asserts against it | Always add negative tests before declaring success |
| **Narrowed pattern inadvertently misses edge cases** | "Maintenant que" is only one context-exploitation prefix — future requests may use "Puisque", "Étant donné que", "Comme tu as" | Check whether additional prefixes exist in the codebase's known attack patterns or conversation history |
| **Sub-agent claims a false positive exists when it doesn't** | Sub-agent says "X is blocked" without tracing the actual code | Always reproduce the trace at the terminal before applying a fix. The control flow or normalization may differ from what the sub-agent assumed. |
