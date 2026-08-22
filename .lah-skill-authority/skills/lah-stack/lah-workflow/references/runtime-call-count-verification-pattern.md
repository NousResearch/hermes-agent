# Runtime Call Count Verification Pattern

Prove that a production function is called exactly N times per request, with no duplicate injection, no hidden alternate paths, and no conflicting regeneration.

Use this pattern for READ_ONLY_AUDIT missions where you need to verify wiring integrity after a repair, or to investigate suspected duplicate calls.

---

## When to use

- After adding a new production function call, verify it fires exactly once per request
- Investigating suspected duplicate injection (same output appearing twice in payload)
- Verifying that a fallback path doesn't also call the target function
- Proving backward compatibility after refactoring a call chain

## Procedure

### Phase 1 — Static exhaustivity

Use multiple search tools to find every occurrence. Don't rely on one tool alone.

```bash
# rg for comprehensive coverage
rg -n 'functionName' --include='*.js' --include='*.mjs' src/ test/

# Verify no re-exports in index files
rg -n 'functionName' src/**/index.js

# CodeGraph for call graph
codegraph_explore({ query: "functionName", maxFiles: 10 })
```

Build a classification table with columns: file, line, classification, role.

| Classification | Meaning |
|----------------|---------|
| PRODUCTION_DEFINITION | The canonical `export function` or `export const` |
| PRODUCTION_IMPORT | Static `import { ... }` in a production source file |
| PRODUCTION_DIRECT_CALL | A call expression `fn(...)` in production code |
| PRODUCTION_INDIRECT_CALL | A wrapper, callback, or event handler that calls it |
| REEXPORT | `export { fn } from './module'` in an index barrel |
| TEST_CALL | Any call inside `test/` or `test-*.js` files |
| TEST_MOCK_OR_SPY | `jest.spyOn`, `sinon.stub`, `vi.mock` |
| DOCUMENTATION_ONLY | Reports, plans, continuity JSONs, README |
| GENERATED_OR_ARCHIVED | Output files in `reports/`, `artifacts/` — skip |
| DEAD_OR_UNUSED_IMPORT | Import exists but no call site in same file |
| AMBIGUOUS | Unclear — requires runtime verification |

### Phase 2 — Production call graph

Trace the complete chain from entry point to target function:

```
entrypoint()
  → intermediate()
    → targetFunction()    ← count this
```

For each caller in the graph, note:
- Is it a production path or test-only?
- Is it reachable during a normal request?
- Does it run in the same tick/request or a background path?

Check for **alternate entry points**. A common pattern is a gateway router that ALSO calls the same build function:

```
Gateway path → buildBrainAskResponse → targetFunction   (1 call)
CLI path     → buildBrainAskResponse → targetFunction   (same 1 call)
```

Both paths converge on the same function — count = 1 per request regardless of entry point. List both entry points in the report to prove you checked.

### Phase 3 — Runtime payload inspection (definitive proof)

Create a verification script that intercepts the provider-bound payload and counts occurrences:

```javascript
let capturedPayload = null;

const captureFetch = async (url, options) => {
  capturedPayload = JSON.parse(options.body);
  // Return mock to avoid real API cost
  return new Response(JSON.stringify({
    id: 'mock-verify',
    choices: [{ message: { content: '[verify]' } }],
    usage: { prompt_tokens: 0, completion_tokens: 0 }
  }), { status: 200, headers: { 'content-type': 'application/json' } });
};

await entryPoint({ fetchImpl: captureFetch, ... });
```

Then parse the payload's system message to count unique items and blocks:

```javascript
// Count unique [kind] markers (each item starts with - [kind])
const itemPattern = /- \[(\w+)\]/g;
const itemKinds = [];
let match;
while ((match = itemPattern.exec(systemContent)) !== null) {
  itemKinds.push(match[1]);
}
const targetCount = itemKinds.filter(k => k === 'target_kind').length;

// Count specific blocks
const blockCount = (systemContent.match(/=== UNIQUE BLOCK HEADER ===/g) || []).length;
```

**Pitfall — string counting**: Don't count raw string occurrences of a kind/name value with `String.match()` — both `kind` and `name` fields often contain the same term, double-counting a single item. Parse item boundaries instead.

**Pitfall — fetch intercept chain breaking**: When running multiple interception scenarios in a loop, each `captureFetch` closure captures the current `globalThis.fetch` at creation time. If the entry point receives `fetchImpl` as a parameter (the common pattern), this chains correctly. But if it reads `globalThis.fetch` internally, the interceptor must be assigned to `globalThis.fetch` instead. Verify the entry point's fetch strategy before designing the test.

### Phase 4 — Semantic duplication check

If the target function produces text in the payload, verify it appears exactly once by:

1. Counting unique header/block markers (e.g. `=== FRESHNESS DES PREUVES ===`)
2. Checking for near-duplicate lines in the payload section
3. Verifying that no OTHER component independently generates equivalent text
4. Checking metadata-only fields (which are NOT rendered as provider-visible text)

Distinguish **complementary** representations (same data, different form — one rendered, one in metadata) from **duplicative** representations (same data, same form, both rendered).

### Phase 5 — Regressions

Run the smallest set that covers the target function:

| Coverage | When |
|----------|------|
| Unit tests for the function | Always |
| Module-level tests (e.g. provider composer, retrieval builder) | When function interacts with these |
| E2E certification | When function is in the canonical pipeline |
| Strategic replay | When function affects provider-bound content |

## Report template

```
# FunctionName Single Call Verification

## Verdict: FUNCTION_NAME_SINGLE_CALL_VERIFICATION_V1_CERTIFIED

- Production calls per request: **N**
- Payload insertions per request: **N**
- Re-exports / aliases / hidden paths: **N**
- Dead or unused imports: **N**
- Semantic duplication: COMPLEMENTARY_ONLY (or DUPLICATE)
- Files modified: N
```
