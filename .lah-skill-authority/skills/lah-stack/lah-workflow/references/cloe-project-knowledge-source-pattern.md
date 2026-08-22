# CLOE Project Knowledge Source Pattern

## Purpose

Bounded read-only project knowledge access layer that lets CLOE answer operator questions about roadmap, QA, certification, memory locks, and project state — without creating unrestricted filesystem access.

## Architecture

### The Knowledge Source Module

**File:** `src/cognitive/cloe-project-knowledge-source.js`

Scans allowlisted directories and extracts concise structured facts with provenance. Key properties:

- **Static allowlist**: `docs/architecture`, `docs/operator/receipts`, `docs/mcporter`, `test/reports`, `.cloe/memory-locks`, `.cloe/continuity`, `.cloe/operator-packets`, `docs/roadmap`, `docs/governance`
- **Path traversal rejection**: Resolves `realpathSync` and verifies against allowed roots
- **Symlink escape detection**: Rejects symlinks whose resolved path escapes an allowed root
- **Bounds**: 100KB per file, 50 total files, 5000 total chars, 800 per fact value (815 with truncation suffix)
- **Forbidden patterns**: `.env`, `secrets`, `credentials`, `tokens`, `node_modules`, `.git`
- **9 knowledge targets**: roadmap, certification, architecture_review, qa_reports, memory_locks, continuity, known_risks, next_mission, baseline_metrics
- **Provenance**: Every fact has `{source, layer, confidence, uncertain, timestamp}`

### Knowledge Targets

| Target | Label | Scan Path |
|--------|-------|-----------|
| roadmap | ROADMAP_COMPLETION_STATE | `.cloe/memory-locks/*roadmap*`, `docs/roadmap/*`, `docs/mcporter/*CLOE_V3*` |
| certification | CLOE_CANONICAL_CERTIFICATION_STATUS | `docs/mcporter/*CERTIFICATION*`, `docs/mcporter/*CONTINUITY*`, `docs/architecture/*CERTIFICATION*`, `docs/mcporter/*CLOE_V3*` |
| architecture_review | CLOE_ARCHITECTURAL_REVIEW | `docs/architecture/*ARCHITECTURAL_REVIEW*`, `docs/architecture/*ARCHITECTURE_GATE*`, `docs/architecture/*CLOE_V3*`, `docs/architecture/*CERTIFIED_BASELINE*` |
| qa_reports | RECENT_QA_REPORTS | `test/reports/*.json` (2 most recent) |
| memory_locks | RECENT_MEMORY_LOCKS | `.cloe/memory-locks/*` (5 most recent) |
| continuity | CERTIFIED_PROJECT_CONTINUITY | `docs/mcporter/*.json` (3 most recent) |
| known_risks | KNOWN_RISKS | Heuristic keyword extraction from architecture docs and memory locks |
| next_mission | RECOMMENDED_NEXT_MISSION | `.cloe/operator-packets/*`, `.cloe/memory-locks/*roadmap*` |
| baseline_metrics | CERTIFIED_BASELINE_METRICS | `test/reports/*.json`, `docs/operator/receipts/*.json` |

## The Two-Path Problem (Critical)

CLOE has **two distinct runtime paths** that produce answers, and project knowledge must be injected in BOTH:

### Path A: Gateway/CLOE Path

```
Gateway → createReadonlyConversationRouter().route()
  → classifyReadonlyConversationIntent()
  → buildCloeAssistantResponse()
    → createCloeRuntimeAdapter().runAssistantLoop()
    → fromCloeEnvelope() → ReasoningContextEnvelope
    → *** scanProjectKnowledge() *** ← WIRED IN
    → reasoningContextToBrainPack() (legacy .context format)
    → buildOpenClawBrainContext()
    → renderOperatorAnswer()
```

**Entry point:** `src/services/gateway/readonly-conversation-router.js:704` (`buildCloeAssistantResponse`)
**Wiring location:** Line 722-734 (after `fromCloeEnvelope`, injects into `rcEnvelope.verified_facts`)
**Smoke test:** `tools/ci/cloe-real-operator-9-question-smoke-test-v2.mjs` — imports router directly
**Used by:** Telegram bot, HTTP brain-ask endpoint, Gateway chat adapter

### Path B: CLI `brain ask` Path

```
scripts/openclaw-operator-cli.js
  → buildBrainAskResponse()
    → buildLocalReadOnlyBrainAskResponse() ← only ~24 exact English phrases
      → createReadonlyConversationRouter().route() → Path A
    → (fall through) resolveBrainProviderConfig()
    → buildCognitiveContextPack() (NEW format, no .context field)
    → *** scanProjectKnowledge() *** ← WIRED IN (PR #595)
    → buildOpenClawBrainContext()
    → POST to LLM provider
```

**Entry point:** `src/services/readonly-operator-cli-client.js:844` (`buildBrainAskResponse`)
**Wiring location:** Lines 1014-1031 (after `buildCognitiveContextPack`, injects as `available_items`)
**Used by:** `node scripts/openclaw-operator-cli.js brain ask "<prompt>"`
**Problem:** The provider path was NOT wired in PR #590 — only fixed in PR #595

### Key Difference: Context Pack Formats

| Aspect | Path A (Gateway) | Path B (CLI) |
|--------|------------------|--------------|
| Context builder | `reasoningContextToBrainPack()` | `buildCognitiveContextPack()` |
| Pack format | Legacy: `.context` (raw text) | New: `.available_items` (structured) |
| Envelope model | ReasoningContextEnvelope V1 | None |
| Answer source | `renderOperatorAnswer()` → deterministic text | LLM provider completion |

## Knowledge Injection Gate (Critical Pitfall)

### The Wrong Gate

When wiring into Path B (CLI provider path), the initial gate was:

```javascript
if (knowledge.facts.length > 0 && cognitiveContextPack.needs_runtime_context)
```

**Problem:** `selectCognitiveContextIntent()` only sets `needs_runtime_context = true` when TAG_RULES match runtime keywords (skills, mcp, agents, runtime, etc.). NONE of the 9Q prompts match any TAG_RULE, so `needs_runtime_context` is false for ALL of them — including CLOE roadmap, QA, and blockers questions.

### The Correct Gate

```javascript
if (knowledge.facts.length > 0 && !cognitiveContextPack.intent_tags.includes('general_knowledge'))
```

This injects project knowledge for ANY prompt that isn't pure general knowledge (matched by GENERAL_KNOWN_PATTERNS: "explain X", "what is Y", "describe Z", etc.). CLOE-specific prompts like "Où en est-on dans la roadmap CLOE ?" are correctly classified as NOT general knowledge and get knowledge injection.

**PL/SQL of the fix:**
- PR #590: Wired `scanProjectKnowledge` into Path A only ✓
- PR #592: Wired into Path B BUT with wrong gate (`needs_runtime_context`) ✗
- PR #595: Fixed gate to `!intent_tags.includes('general_knowledge')` ✓

## Request Failure Fallback

When the LLM provider is unreachable in Path B (timeout, network error, auth failure), the original code returned raw `ok: false` JSON (`REQUEST_FAILED` / `REQUEST_TIMEOUT`) which the CLI outputs as unformatted error text — a terrible UX.

### The Fix

Replace the `catch` block with a natural language fallback that uses available project knowledge:

```
I encountered a connection issue reaching the LLM provider. Based on available project knowledge:
• ROADMAP_COMPLETION_STATE: ... [source: ..., confidence: 95%]
• RECENT_MEMORY_LOCKS: ... [source: ..., confidence: 95%]
• ...

Some answers may be limited because the LLM provider is unavailable.
Check your provider configuration (OPENCLAW_BRAIN_PROVIDER, API key, network).
```

**Wiring location:** `src/services/readonly-operator-cli-client.js:1154-1180` (the entire `catch` block)

**Key details:**
- Return `ok: true` so the CLI passes through `formatHumanResult` instead of `writeError`
- Set `provider_error` field in `data` so the error is still visible in JSON mode
- Use `safety: createBrainSafetyEnvelope(false)` (no external network used for fallback)

## The Local-Only Gate Problem

The `buildLocalReadOnlyBrainAskResponse` function has a gate `isLocalReadOnlyStackPrompt()` that checks **exact equality** after accent-stripped normalization against ~24 hardcoded English phrases:

```
'stack map', 'stack overview', 'stack inventory', 'stack index',
'hierarchical index', 'current blockers', 'latest blockers',
'latest continuity', 'latest verdicts', 'operator packets',
'continuity records', 'superpowers plans'
```

**Problem:** This is exact-match only — "current blockers" works but "what are the current blockers?" does not. All French prompts fail. Without a configured LLM provider, there is no way to get a CLOE answer for these prompts.

**Future work:** Expand this to use fuzzy matching or keyword detection to handle French/natural-language variants.

## Project Knowledge Refresh Procedure

When CLOE's certification state changes (e.g., V2⟶V3), the project knowledge source must be explicitly updated. Module tests pass but CLOE's answers will reference the old canonical state until the knowledge source is refreshed.

### Audit Checklist

1. **Check knowledge labels** in `KNOWLEDGE_TARGETS`: are `label` values still accurate?
   - `CLOE_V1_CERTIFICATION` → `CLOE_CANONICAL_CERTIFICATION_STATUS`
   - `CLOE_V1_ARCHITECTURAL_REVIEW` → `CLOE_ARCHITECTURAL_REVIEW`
   - If stale labels exist, CLOE will inject "V1" into answers even when reading V2/V3 data.

2. **Check scan patterns**: do pattern globs match the new files?
   - Add `docs/mcporter/*CLOE_V3*` for certification/roadmap targets
   - Add `docs/architecture/*CLOE_V3*` for architecture targets
   - Add `docs/architecture/*CERTIFIED_BASELINE*` for baseline docs

3. **Check V3 field extraction** in `scanCertification`: does the scanner extract V3-specific fields?
   - `baseline_freeze.tag`, `baseline_freeze.freeze_commit`
   - `v3_relationship.builds_on`, `v3_relationship.additive_only`
   - `pr.merge_commit`
   - `verdict` containing `CLOE_V3`

4. **Check leak patterns**: run the benchmark runner and look for `internal_leak` signals.
   - `selected_collectors`: fix in `src/brain/cognitive-context-formatters.js` — remove the `selected_collectors` line from `formatCognitiveContextPack()` output
   - `safety_envelope`: may need filtering in the answer renderer

5. **Verify via knowledge source scan**:
   ```bash
   node -e "
   import('./src/cognitive/cloe-project-knowledge-source.js').then(m => {
     const r = m.scanProjectKnowledge(process.cwd());
     for (const f of r.facts) console.log(f.type, '→', (f.value || '').slice(0, 120));
   });"
   ```

6. **Verify via provider smoke test**:
   ```bash
   npm run openclaw -- brain ask \
     "Quel est le statut de certification actuel ? Reponds en une phrase." --json
   ```
   Expected: answer references V3 canonical certification, not V1.

### PR Workflow for Knowledge Refresh

The refresh is a self-contained fix that touches only 2-3 files:
- `src/cognitive/cloe-project-knowledge-source.js` — label/pattern/extraction updates
- `src/brain/cognitive-context-formatters.js` — leak fix (if applicable)
- `docs/operator/receipts/local-ci-verified-*.json` — new receipt

Follow the standard LOCAL_CI_VERIFIED workflow: commit → push → PR → receipt → merge → post-merge verify. No V2/V3 module changes, no behavioral logic changes.

## Internal Leak Audit

The benchmark runner (`tools/ci/cloe-operator-benchmark-runner.mjs`) detects these leaks automatically in provider answers:

| Pattern | Detection | Fix Location |
|---------|-----------|-------------|
| `selected_collectors` | Internal cognitive context pack field | `src/brain/cognitive-context-formatters.js` — remove from `formatCognitiveContextPack()` |
| `safety_envelope` | Policy structure exposed | `src/cognitive/cloe-user-facing-answer-renderer.js` or `tools/ci/cloe-real-operator-9-question-smoke-test-v2.mjs` |
| `ReasoningContextEnvelope` | Reasoning internals | Answer renderer post-processing |
| `system prompt` / `brain pack` | Prompt framing | LLM prompt sanitization |
| `mutation_hardstop` / `governed_action_block` | Policy dump | Context engine filter |

### Why leaks happen

The `formatCognitiveContextPack()` function in `cognitive-context-formatters.js` builds the LLM context prompt with implementation metadata. If the LLM reproduces internal field names in its answer, the benchmark flags it as a leak.

**Fix pattern:** Remove internal-only fields from the user-facing context prompt. The `selected_collectors` field is implementation metadata that provides no value to the LLM's reasoning — removing it from the prompt prevents the leak without affecting answer quality.

### Testing the leak fix

```bash
# Run the full benchmark and check for internal_leak signals
npm run benchmark:cloe-operator
# Then check the report:
python3 -c "
import json
with open('test/reports/cloe-operator-benchmark-latest.json') as f:
    d = json.load(f)
for q in d['questions']:
    leaks = q.get('signals',{}).get('internal_leak',[])
    if leaks: print(q['id'], 'LEAKS:', leaks)
"
```

## Files Reference
- `src/services/gateway/readonly-conversation-router.js:704-768` — buildCloeAssistantResponse (Path A)
- `src/services/readonly-operator-cli-client.js:844-1181` — buildBrainAskResponse (Path B)
- `src/services/readonly-operator-cli-client.js:314-350` — isLocalReadOnlyStackPrompt gate
- `src/brain/cognitive-context-selectors.js:217-324` — selectCognitiveContextIntent
- `src/brain/cognitive-context-engine.js:372-464` — buildCognitiveContextPack
- `src/brain/openclaw-brain-context-builder.js:583-619` — selectContextSections (legacy format fix)
- `test/cloe-project-knowledge-source.test.js` — 14 tests
- `tools/ci/cloe-real-operator-9-question-smoke-test-v2.mjs` — 9Q smoke test
- `docs/operator/cloe-brain-ask-runtime-path-audit-v1.md` — Full audit report from this session

## PR History

| PR | Purpose | Status |
|----|---------|--------|
| #590 | CLOE_READONLY_KNOWLEDGE_ACCESS_EXPANSION_V1 — knowledge source + Path A wiring | Merged |
| #592 | Path B wiring (wrong gate: `needs_runtime_context`) | Merged |
| #593 | Test bounds fix (truncation suffix) | Merged |
| #595 | Gate fix (`!general_knowledge`) + provider fallback | Merged |
| #611 | V3 canonical status refresh: update stale V1 labels, add V3 patterns, fix selected_collectors leak | Merged |
