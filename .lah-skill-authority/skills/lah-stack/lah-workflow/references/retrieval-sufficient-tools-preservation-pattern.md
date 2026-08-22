# Retrieval-Sufficient Tools Preservation Pattern (CLOE)

Established during `CLOE_RETRIEVAL_ANSWER_SUFFICIENT_NATIVE_TOOLS_PRESERVATION_REPAIR_V1` (PR #701, 2026-08-05).

## Defect class

Retrieval-first branches inside a chat-completions service that strip client-supplied
tools to force synthesis:

```js
// FIRST DIVERGENCE — ANSWER_SUFFICIENT branch
effectiveTools = undefined;
effectiveToolChoice = 'none';
forcedSynthesis = true;
```

Failure chain observed in production (OurDream turns):

```
SPA webchat → OpenClaw runtime → provider "cloe" → POST /chat/completions
→ buildNativeChatCompletions → retrieval decision = ANSWER_SUFFICIENT
→ effectiveTools = undefined → effectiveToolChoice = none
→ provider payload contains no tools → DeepSeek cannot emit structured tool_calls
→ assistant stop=stop → native OpenClaw tool loop never starts
```

Key diagnostic: `stop=stop` on a provider that **does** support native tool_calls is
NOT evidence the provider lacks tool support — it is usually evidence the payload
was stripped of tools. (See memory: "DeepSeek PEUT tool_calls natifs — stop=stop =
strip tools ANSWER_SUFFICIENT").

## Repair shape (hasClientTools guard)

```js
const hasClientTools =
  Array.isArray(effectiveTools) &&
  effectiveTools.length > 0;

if (!hasClientTools) {
  effectiveTools = undefined;
  effectiveToolChoice = 'none';
}

forcedSynthesis = true;
```

Semantics:
- `hasClientTools = true` → do NOT clear tools, do NOT force tool_choice to none.
  The provider payload keeps the client's tools + compatible tool_choice, so the
  runtime tool loop can start.
- `hasClientTools = false` (absent or `[]`) → tools may remain undefined and
  tool_choice=none is allowed (retrieval-first text path unchanged).

Boundaries respected:
- Never inject tools the client did not request (no toolBridge injection).
- Never force `auto` unless the client contract already resolves to it.
- Do not change retrieval decision classification; do not disable ANSWER_SUFFICIENT;
  do not remove retrieval grounding.
- Do not weaken DSML/NFKC guards or PROVIDER_RAW_TOOL_MARKUP handling.

## Scope traps

- **Sibling branches may carry the same defect.** In the same mission the `greeting`
  branch ALSO stripped client tools when supplied (Phase 6 required "tools remain
  available but provider may answer text normally"). Check every retrieval-first /
  synthesis-forcing branch for the same strip, not just the named one.
- **Do NOT touch budget/duplicate-exceeded branches.** Those legitimately force
  synthesis (hard budget caps) and their tests assert `tools undefined` /
  `tool_choice none`. Only retrieval-classification branches get the guard.
- **Existing tests rarely pin the payload strip** — they assert on the RESPONSE
  (`message.tool_calls === undefined`) with a text-mock provider, which stays true
  after the fix. So a patch can be behavior-safe without breaking old tests; new
  tests must capture the provider payload (fetchImpl that parses `options.body`)
  to prove tools survive.

## Test matrix (RED before patch)

| Case | Input | Expected after repair |
|------|-------|----------------------|
| A — tools supplied | ANSWER_SUFFICIENT + non-empty tools + tool_choice | payload retains tools; tool_choice NOT forced to none; grounding injected |
| B — no tools | ANSWER_SUFFICIENT + tools absent | payload has no tools; tool_choice=none allowed; grounding injected |
| C — empty array | `tools=[]` | treated as no usable tools; no tools added; tool_choice=none allowed |
| D — provider tool_calls | ANSWER_SUFFICIENT + tools + provider returns `message.tool_calls` | buildOpenAiResponse preserves tool_calls; finish reason compatible; no text-only conversion |

RED validation: on base commit the new suite fails exactly the tool-preservation
assertions (Case A, D, integration) while no-tools cases pass — proving the test
is specific to the regression, not the whole path.

## Real-stack integration vs mock gateway

Two test layers:
1. Deterministic mock gateway returning `{ ok: true, authority, evidence: [substance] }`
   → classifyRetrievalDecision yields ANSWER_SUFFICIENT (authority != 'NONE' + substance).
2. Real knowledge stack (`createLahKnowledgeRegistry` + `createUnifiedRetrievalGateway`
   + `createIncrementalIndexer` + `seedPilotCorpus`) for OurDream regression fixtures —
   verify with a probe script BEFORE writing tests that the chosen queries actually
   classify ANSWER_SUFFICIENT (some phrasings hit TARGETED_VERIFICATION_REQUIRED or
   RETRIEVAL_INSUFFICIENT).

## Proof-of-contract harness trick

To simulate DeepSeek's real behavior (structured tool_calls only possible when the
payload has tools), make the mock fetch conditional:

```js
function capturingFetch(providerResponse, captured, { conditionalTools = false } = {}) {
  return async (_url, options) => {
    const body = JSON.parse(options.body);
    if (captured) captured.push(body);
    let response = providerResponse;
    if (conditionalTools && (!body || !Array.isArray(body.tools) || body.tools.length === 0)) {
      response = TEXT_RESPONSE; // provider "cannot" emit tool_calls without tools
    }
    return { ok: true, status: 200, text: async () => JSON.stringify(response) };
  };
}
```

This makes Case D RED before the fix at the payload level (not just at the response
builder level), because without tools in the payload the conditional mock returns text.
