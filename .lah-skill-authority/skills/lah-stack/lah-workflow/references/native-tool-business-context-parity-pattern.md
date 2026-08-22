# Native-Tool Business Context Parity Pattern (shared server-side enrichment)

Established 2026-08-08 on HERMES_CLOE_NATIVE_TOOL_BUSINESS_CONTEXT_PARITY_REPAIR_V1
(openclaw-runtime lah-openclaw-mvp, CERTIFIED — deployed 1ddce4d, PR #731, live A/B/C PASS).

## Problem class

A server exposes two entry points for the same conversational capability:
- **non-tool path** (`buildBrainAskResponse` via /brain/ask and /chat/completions without tools) — injects rich read-only business evidence (business runtime, Affiliate Execution Context, configured offer inventory, governed microtest proposal) into the provider prompt;
- **native-tool path** (`buildNativeChatCompletions` via /chat/completions with tools) — forwards the OpenAI tool surface but never composes that evidence.

Result with tools present: provider has no grounded business facts, `governor.business_context=null`, `evidence_present_count=0`, and the model burns its tool budget (observed 4/4 in round 1) rediscovering facts the non-tool path injects for free. Response becomes non-verifiable.

## Diagnosis recipe

1. **Confirm what is actually deployed.** The canonical checkout may be parked on an unrelated stale branch. Verify byte-identity: `docker exec <container> sha256sum /app/src/services/<file>.js` vs `git show origin/main:<path> | sha256sum` — equal hashes mean the container == origin/main and origin/main is the correct base. (Observed: deployed GIT_COMMIT c0c86ef was NOT an ancestor of canonical HEAD b65cc67; origin/main was the base.)
2. **Reproduce A/B/C live**: same business prompt through (A) /brain/ask, (B) /chat/completions no tools, (C) /chat/completions with a harmless tool. Capture `_cloe.governor` (business_context, business_plan, budget), `finish_reason`, tool_calls count, and answer preview. The probe reads the admin key inside the container env and never prints it.
3. **Locate the enrichment**: grep for the injected item `kind`s (`business_runtime_context`, `affiliate_execution_context`, `configured_offer_inventory`, `governed_microtest_proposal`). They are often inline inside the non-tool response builder only, pushing into a cognitive context pack that `attach_to_prompt` renders via a shared formatter.

## Fix pattern

1. **Extract a shared module** `buildCanonicalBusinessContext({ env, fetchImpl, persistDecisionRecord })` composing the EXISTING sources (never rebuild them — the mission hard-scope rule). Return `{ ok, available_items, compact_summary, attach_to_prompt }` plus the raw microtest result so callers can persist decision records.
2. **Non-tool path** calls it and pushes returned items into its existing cognitive pack. With `persistDecisionRecord: true` the behavior is byte-identical to the old inline blocks (LOT 8 decision-record persistence stays brain-path-only).
3. **Native-tool path** calls it and injects the rendered pack as a SYSTEM message — never as a tool definition — so the business evidence does NOT count as a provider tool call and does NOT consume the tool budget. Record `governor.business_context = { composed, attach_to_prompt, evidence_present_count, item_kinds }` for certification checks.
4. **Render with the SAME formatter** the non-tool path uses (`formatCognitiveContextPack` from cognitive-context-formatters.js), so both paths emit identical context text. Inject after all routing/budget branches settle `effectiveMessages`, before payload build, so the system message survives every branch.

## Verification techniques

### Provider-payload capture (assert on what the provider actually receives)
Mock fetch records `calls.push({ url, init })`; assert on `JSON.parse(calls.filter(non-bridge)[0].init.body)`:
- system messages contain the business kinds (`affiliate_execution_context`, `configured_offer_inventory`, `governed_microtest_proposal`);
- `tools[]` names do NOT include business-evidence kinds; the client's harmless tool is still forwarded.
Asserting on the RESPONSE object is wrong — the response does not echo the provider payload.

### Pure-extraction byte parity
After replacing N inline lines with a shared call, prove the refactor is behavior-preserving: parse `git diff` removed lines, normalize (strip comments/whitespace), apply the accumulator rename (`cognitiveContextPack.` → `pack.`), and assert every functional line exists in the new module. 0 missing = pure extraction. (Observed: 286/286 functional lines matched.)

### Env fixture completeness
`buildNativeChatCompletions` fails with `PROVIDER_NOT_READY: missing DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL` when those env vars are absent — set dummy values in the test fixture even though fetch is mocked. Save/restore every env var you touch (LAHB_URL, LAHB_ADMIN_API_KEY, OPENCLAW_BRAIN_PROVIDER, EXOCLICK_LIVE_ENABLED, DEEPSEEK_*) symmetrically.

## Traps

- **Full-suite differential candidate**: a single "only-in-with-diff" failing test that passes isolated on BOTH baseline and with-diff trees is flaky/order-dependent, not a regression (attribution protocol: run it isolated on the baseline worktree). Don't chase it.
- **OOM in the giant entry-point test file** (`readonly-operator-cli-client.test.js`, ~1300 lines) is pre-existing and environmental on the 8GB VPS (see full-suite-test-traps.md Trap 4). Write a NEW focused standalone test file for the changed path — never append to the OOMing file.
- **Baseline capture must use a fresh /tmp worktree at origin/main**, never the dirty canonical checkout (full-suite-test-traps.md).
- **`npm ci` in BOTH worktrees before the full-suite differential**, or import failures collapse test counts and the name comparison is garbage.
