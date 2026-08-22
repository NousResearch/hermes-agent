# Cloé Business Runtime Facade — Wiring + Live Certification Loop

Class-level pattern for adding a READ-ONLY business capability that Cloé must
consume through the `/brain/ask` path (established CLOE_LAHB_AUTONOMOUS_AFFILIATE_
RUNTIME_E2E_V1, Lots 1–5, 2026-08-08). Recurred on every lot; the wiring shape
and the certification loop are the durable parts.

## Architecture (the full chain)

```
LAH Brain facade (one admin-key HTTP endpoint, composes in-process)
  → openclaw bridge client (timeout-bounded, fail-closed)
  → composition service (e.g. getAffiliateExecutionContext, getConfiguredOfferInventory)
  → cognitiveContextPack.available_items injection in buildBrainAskResponse
  → LLM grounded answer
```

Each lot built one more layer and injected it as its own `available_items` entry,
so the LLM sees per-item provenance. Reuse the previous lot's service as input,
never re-fetch the same data twice.

## The injection block shape (in src/services/readonly-operator-cli-client.js)

Insert after the project-knowledge / evidence-dossier blocks, before web search.
Every new capability gets a `try/catch` that pushes one available_item and
appends to `compact_summary` + sets `attach_to_prompt = true`:

```js
try {
  const result = await someService({ fetchImpl, env });   // pass fetchImpl ALWAYS (DI)
  if (result.ok && <shape-ok>) {
    cognitiveContextPack.available_items.push({
      kind: 'capability_name',
      name: 'capability_name_v1',
      source: 'describe exact provenance',
      available: true, read_only: true,
      safety_flags: ['read_only', 'no_write', 'no_execute'],
      description: 'human summary with preview lines',
      metadata: { ...structured facts the LLM must not re-derive ... }
    });
    cognitiveContextPack.compact_summary += ` | Cap: ...`;
    cognitiveContextPack.attach_to_prompt = true;
  } else {
    // BLOCKED or unavailable branch — available:false, surface reasons verbatim
    cognitiveContextPack.available_items.push({ ... available: false,
      description: `... unavailable (fail-closed): ${result.error || reason} — do NOT fabricate ...` });
  }
} catch (err) { /* same unavailable branch */ }
```

Hard rules carried into every block: missing != zero, configured != approved,
never fabricate business facts in the unavailable branch, `url_redacted` for any
affiliate material, safety envelope `{read_only:true, provider_write:false}`.

## PITFALL — contract-shape mismatch between service return and wiring

Symptom: unit tests PASS (they call the service directly and read `result.sections`)
but the live Cloé answer says "context unavailable (fail-closed)" with no error
detail. The wiring read `affiliateContext.context` while the service returns
`{ok, sections, ...}` at TOP LEVEL — `.context` is undefined → `ok && .context`
falsy → the fail-closed branch fires silently.

Root cause class: the service and the wiring were written with different return
shapes and only the live probe catches it. Fix:
1. Read the service's actual return statement before writing the wiring condition.
2. Assert the shape in the test that exercises the WIRING (buildBrainAskResponse),
   not just the service — a service-level unit test cannot see this bug.
3. Verify live with the service probe BEFORE committing the wiring (see below).

## PITFALL — fetchImpl must be threaded into nested services

`createBusinessStateMemoryInterface({ fetchImpl })` and every composition service
must forward the injected `fetchImpl` down to the bridge. If the wiring calls
`createBusinessStateMemoryInterface()` without `{fetchImpl}`, the bridge falls back
to `globalThis.fetch` — the unit test's mock never sees the call and the test fails
with "bridge must be invoked" even though the code is correct in production.

## Live service probe (before wiring commit)

```bash
# env via bracket-safe extraction; NEVER inline the secret in write_file
KEY=$(cat /tmp/lahb-key.txt); export LAHB_URL=...; export LAHB_ADMIN_API_KEY="$KEY"
node --input-type=module -e "import { getX } from './src/services/x.js'; const r = await getX({}); console.log(JSON.stringify(r, null, 1))"
```
This catches the contract shape, real bridge reachability, and real data
composition in one call — do it before the deploy/cert cycle.

## REAL Cloé certification loop (each lot)

1. Merge PR(s) (admin exact-head merge; REMOTE_CI=UNAVAILABLE when billing blocks,
   document LOCAL_GOVERNANCE=PASS / FOCUSED_TESTS=PASS / REGRESSION=PASS).
2. Build image with LITERAL GIT_COMMIT, verify image ENV, tag `rollout-<sha>`,
   run the exact-SHA deployer → DEPLOYED_EXACT_SHA + ALREADY_DEPLOYED idempotent.
3. health 200 on :4000, `/version` shows the new commit (lah-brain: deploy == merge).
4. POST /brain/ask with a FRESH conversation_id (new session key → tool budget 0),
   the mission's exact prompt, admin key from .env via awk (never printed).
5. PASS = answer contains bridge-grounded facts (offer counts, spend figures,
   section statuses, generated_at, missing/conflicts) — NOT static/memory text
   like "je n'ai pas de visibilité en temps réel". Record evidence lines verbatim.
6. Write receipt JSON to docs/mcporter/CLOE_..._LOTn_CERT.json (verdict, PR/merge
   SHAs, deployed SHA, tests, live evidence, safety flags, next lot).

## Layered composition notes (Lots 2/3/5)

- getAffiliateExecutionContext: 11 canonical sections (offer, traffic, marketplace,
  zones, exclusions, tracking, attribution, economics, creative, campaign,
  governance) with status vocabulary OBSERVED/DERIVED/MISSING/NOT_SUPPORTED +
  context_id, freshness, provenance, conflicts, missing. exclusions derived from
  zone-fast-cut-policy (local), creative NOT_SUPPORTED when no inventory exists.
- getConfiguredOfferInventory: canonical identity provider + numeric_offer_id;
  provider_approval NEVER inferred from configured (stays UNKNOWN/REQUIRES_APPROVAL
  → conflict "configured != approved"); execution_status derived from
  destination/active only.
- generateGovernedMicrotestProposal: candidate selected ONLY from EXECUTABLE offers
  (deterministic preference: known revenue model → approved → first), approval_id
  required, dry-run steps describe would-be API calls with executed:false,
  live_sent:false. No provider calls ever.
