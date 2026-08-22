# Provider Bridge Propagation Proof Pattern

Use when you need to prove (or complete) end-to-end metadata propagation through a canonical provider bridge. This pattern is designed for **MIXED** missions (READ_ONLY_AUDIT baseline → bounded CODE_CHANGE repair → runtime proof).

## When to use

- A function/feature exists that transforms metadata but you don't know if it reaches the real provider payload
- Previous missions recommended wiring something into the provider path but left it unverified
- You need runtime proof, not just static code inspection

## Phase 1 — Baseline capture (no repair)

Capture the unmodified propagation chain before any code changes.

### Method: Payload intercept

Use a mock/intercept `fetchImpl` to capture the provider-bound request body without sending it:

```javascript
const captureFetch = async (url, options) => {
  capturedPayload = JSON.parse(options.body);
  return new Response(JSON.stringify({
    choices: [{ message: { content: '[BASELINE CAPTURE]' } }],
  }), { status: 200, headers: { 'content-type': 'application/json' } });
};

const result = await buildBrainAskResponse({
  env: process.env,
  prompt: QUESTION,
  fetchImpl: captureFetch,   // ← intercept, never reaches real API
  timeoutMs: 15000,
});
```

### Checks to perform

In the captured `system` message content, search for:
- Target metadata keywords (e.g. `FRESHNESS`, `evidence_freshness_context`, `CURRENT_VERIFIED`)
- Whether the expected item appears in `cognitiveContextPack.available_items`
- Whether the format item (description + preview_items) renders the metadata

### Output

Save:
- Full system content to `CLOE_*_BASELINE_SYSTEM_CONTENT.txt`
- Full payload to `CLOE_*_BASELINE_PAYLOAD.json`
- Structured JSON report with per-stage preservation/loss analysis

## Phase 2 — Bounded repair

Implement the minimal change needed to make the metadata reach the provider. Max 3 repair cycles.

### Typical repair locations

| Target | Typical change |
|--------|---------------|
| `cognitiveContextPack.available_items.push()` | Add missing metadata fields to the item's `metadata` object |
| New available_items entry | Push a new item with `preview_items` containing the rendered metadata string |
| Import missing helper | e.g. `import { buildFreshnessContext } from '...'` |

### Verify repair

Same intercept pattern as Phase 1 — re-capture the payload and check the same keywords now appear.

## Phase 3 — Runtime proof (real provider)

Run actual provider calls through the canonical bridge to prove the metadata affects real answers.

### Pitfall: Fetch intercept chain

Do NOT stack multiple interceptors in a loop. Each new `interceptFetch` captures the current `globalThis.fetch` at creation time. After N iterations, the call stack is N deep. Instead:

```javascript
// SAFE: pass globalThis.fetch directly, no intercept
const result = await buildBrainAskResponse({
  env: process.env,
  prompt: question,
  fetchImpl: globalThis.fetch,   // ← real call, no intercept
  timeoutMs: 45000,
});
```

For payload capture, run a separate targeted test with interception (Phase 2 pattern). Verify freshness in the runtime answer content instead.

### Minimum runtime scenarios

| # | Scenario | Purpose |
|---|----------|---------|
| 1 | Superseded data | Old failure superseded by current success — provider must distinguish |
| 2 | Unknown freshness | Item without trustworthy timestamp — provider must qualify |
| 3 | Conflicting evidence | Two items with equal authority, no supersession — provider must surface conflict |
| 4 | Fact vs inference | Direct evidence vs derived conclusion — provider must separate |

### Regression replay

If the system has established strategic questions (e.g. 9 operator questions), re-run them all and verify:
- All still produce answers (no regression)
- Metadata still reaches provider payload for all questions
- No governance violations

### Report structure

Required artifacts:
- `test/reports/<mission>/CLOE_*_REPORT.md`
- `test/reports/<mission>/CLOE_*_REPORT.json`
- `docs/mcporter/CLOE_*_CONTINUITY.json`
- `test/reports/<mission>/runtime-answers/` — one file per scenario

Required sections:
1. Environment (SHA, provider, date)
2. Baseline propagation proof (each stage, preserved/lost)
3. Root cause
4. Files changed
5. Controlled fixture results
6. Provider payload proof (sanitized)
7. Runtime scenarios (full questions + answers)
8. Original replay results
9. Test totals
10. Before/after metrics
11. Residual risks

## Script template

Use this pattern for the runtime proof script:

```javascript
import { buildBrainAskResponse } from '...';
import { buildFreshnessContext } from '...';

async function runScenario(scenario) {
  const result = await buildBrainAskResponse({
    env: process.env,
    prompt: scenario.question,
    fetchImpl: globalThis.fetch,   // ← REAL call, no intercept
    timeoutMs: 45000,
  });

  return {
    id: scenario.id,
    answer: result.data?.answer || '',
    ok: result.ok,
    duration_ms: Date.now() - t0,
    // ... other metadata
  };
}
```

## See also

- `retrieval-context-builder-pattern.md` — building the evidence dossier
- `retrieval-provider-integration-pattern.md` — wiring components into the provider bridge
- `cloe-canonical-pipeline-e2e-certification-pattern.md` — full pipeline certification
