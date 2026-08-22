# CLOE business evidence planning — wiring existing authorities into the native tool loop

Established during `CLOE_EXISTING_RETRIEVAL_GOVERNOR_AND_BUSINESS_CAPABILITY_GRAPH_RUNTIME_WIRING_REPAIR_V1` (PR #703, branch `fix/cloe-business-capability-graph-runtime-wiring-v1`).

## Proven divergence (reproduce first, never assume)

A complex business request (OurDream diagnosis / microtest / governed PAUSED plan) through the native OpenClaw tool loop reached the provider with the FULL client tool set (exec/grep/find/read + mutation tools), never consulting the existing business capability registry, with no required-evidence plan and no canonical-authority selection → uncontrolled loop / generic useless synthesis. The generic budget (4 calls) only stops the loop when the client reinjects the full history each round; when the external OpenClaw runtime compacts history, the budget resets every turn.

Reproduce with a round-by-round harness through `buildNativeChatCompletions` (the exact function behind `POST /chat/completions`): provider mock only, zero network. See `scripts/cloe-native-tool-loop-harness.mjs` (committed on the mission branch).

| Scenario | Tools seen by provider | Calls | Terminal |
|----------|------------------------|-------|----------|
| Baseline (unrestricted) | exec, grep, find, read, list_tools, campaign_memory_query, exoclick_stats | 2–3 wasted | generic synthesis, no business evidence |
| Compaction / no reinject | same | unbounded (12+ until client cap) | rounds_exhausted |
| With business evidence planner | campaign_memory_query, exoclick_stats only | ≤2 targeted | bounded useful synthesis |

## Architecture facts (verified on origin/main 6481f51, deployed 66c1987)

- The native OpenClaw loop passes through the gateway: `~/.openclaw/openclaw.json` → provider `cloe` (`baseUrl: http://127.0.0.1:4000`, `api: openai-completions`) → `POST /chat/completions` → `buildNativeChatCompletions`. The WS `/gateway` on :4000 is the ACP control plane, NOT the tool loop — do not debug the loop there.
- The mounted canonical graph (`data/self-audit/canonical-graph/capability-graph.json`, ~101 nodes / 22 edges) contains self-audit capabilities + skills + authority conflicts but **NO business capabilities**. The business capability registry lives in `src/self-audit/evidence/high-roi-capabilities.mjs` (8 capabilities: business-campaign-memory, business-provider-statistics-read, business-tracking-attribution, business-approval-budget-safeguards, business-governed-campaign-creation, business-zone-site-selection, business-creative-inventory-lineage, business-health) — it exists but was never consulted by the conversational runtime.
- `getKnowledgeStack()` (readonly-operator-cli-client.js) builds the unified retrieval gateway WITHOUT campaignMemory/qdrant/mem0/cartelogic backends → only HOT/EXACT/ALIASES/FTS layers answer; business evidence from campaign memory / provider stats is not retrievable through the native path.

## Repair pattern (reuse-first, no second authority)

New deterministic adapter `src/services/cloe-business-evidence-planner.js`:
1. classify request class (OURDREAM_DIAGNOSIS / MICROTEST_RECOMMENDATION / GOVERNED_PAUSED_CAMPAIGN_PLAN / BUSINESS_HEALTH / GENERAL_BUSINESS / NOT_BUSINESS) with accent-safe FR/EN regex — test BOTH raw lowercase AND NFD-normalized text (the `normalizeText()` accent-strip trap from canonical-intent-classifier);
2. fact→capability doctrine keyed on EXISTING `HIGH_ROI_CAPABILITIES` ids → resolve capabilities + canonical authorities (module + symbols) from the registry, never invent modules;
3. evidence present vs missing from the retrieval result — STRICT per-fact substring match on evidence text, never "retrieval returned something"; `ANSWER_SUFFICIENT` ⇒ all facts present;
4. filter client tools: general repo exploration (exec/grep/find/read/list_tools) dropped; mutation tools (launch/activate/create/spend/…) dropped AND flagged (`mutation_tools_present`); business tools kept only when matching a missing-fact capability hint;
5. reuse the governor budget doctrine (`resolveBudget('conversational')` → max_tool_calls 4 / 20s phase) — never re-derive a parallel budget;
6. deterministic: SORT allowed/removed/dedupe arrays before computing the plan hash (plan must be insensitive to client tool-definition order — a shuffled tool array must produce the same hash);
7. fail-closed (empty prompt → typed code), never throws, never executes tools, never authorizes mutation.

Wiring (2 integration points only):
- `chat-completions-service.js` `buildNativeChatCompletions`: when `governance.businessPlanner` present and request class ≠ NOT_BUSINESS and no earlier gate forced synthesis → filter `effectiveTools` to `plan.allowed_tool_names` (empty ⇒ tools undefined + tool_choice none), inject deterministic `BUSINESS EVIDENCE PLAN` system directive, force synthesis when `evidence_missing` empty; skip the generic TARGETED_VERIFICATION / RETRIEVAL_INSUFFICIENT directives when a business plan is active (add `&& !businessPlan` to those branches); record `governor.business_plan` metadata.
- `server.js` `/chat/completions`: pass `businessPlanner: { planifyBusinessEvidence }` in governance.

## Harness recipe (agent-loop replay)

- Provider mock must be PAYLOAD-AWARE: only emit tool calls whose names are present in the provider payload. A mock that keeps emitting find/grep after the server filters tools HIDES the tool-filtering behavior (harness-fidelity bug).
- Capture per round: tool_calls_emitted, finish_reason, governor forced/reason/decision, business_plan metadata.
- Compare baseline (no planner) vs wired (planner) by `tools_seen_by_provider`.
- Sanitize reports (no secrets/urls/tokens) — they are committed as evidence.

## RED→GREEN proof without a second checkout

`git stash push -- <wiring files only>` (keep the planner + tests), run the suite → the critical tool-filtering assertion fails (RED), `git stash pop` → GREEN. Proves the baseline defect with the exact same tree.

## Behavioral receipt timing

Generate the behavioral simulation receipt AFTER EVERY commit on the branch — HEAD changes with doc commits (design doc, operator packet), making the previous receipt stale. Writer pattern: `tools/write-receipt-<mission>.mjs` computing diff_hash + canonical hash + validating from REPO ROOT. 12-scenario receipt for this mission validated `VALIDATE: VALID`.

## Full-suite classification

Fresh baseline worktree @ origin/main, identical globs, compare by test NAME (`comm -23 withdiff baseline`). This mission: with-diff 81 fails ⊆ baseline 87 fails, 0 only-in-with-diff → no regression; repo-wide failures pre-existing/unrelated (cloe-v5, cognitive, memory, telegram REAL). Suffix `_WITH_LIMITATIONS` when repo-wide failures remain unclassified.
