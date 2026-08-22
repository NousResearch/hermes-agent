# In-Container Consumption Probe (CERT E variant)

Session-proven 2026-08-10 on CLOE_CAMPAIGN_PLAYBOOK_RULE_BODY_PROJECTION (PR #757,
deploy 87103e45). Supplements `cloe-live-certification-and-admin-merge.md` §4 (CERT E).

## Problem

The reference CERT E pattern says: "run a probe against the SAME deployed SHA (worktree
at the deployed commit + canonical .env) that builds the pack exactly as /brain/ask
does". For STORE-BACKED modules (canonical memory, campaign playbook, any reader that
reads `data/memory-events` or `data/business`), a HOST-side worktree probe is WRONG:

- The probe resolves `DEFAULT_MEMORY_EVENTS_DIR = resolve(moduleDir, '..', '..', 'data',
  'memory-events')` → the WORKTREE's `data/` directory, which is empty/absent on a fresh
  worktree → `retrieveCampaignPlaybook()` returns nothing → `PLAYBOOK_ITEM_PRESENT=false`.
- The PRODUCTION store lives only inside the running container at `/app/data` (bind mount
  from the canonical checkout), NOT in a temp worktree.

Symptom observed: host probe reported `PLAYBOOK_ITEM_PRESENT=false` while the container
had 47 memory-events files and the same code path returned 5 projected rules.

## Fix: run the probe INSIDE the deployed container

```bash
# 1. Write the probe with imports RELATIVE to /app (run cwd = /app)
# 2. Copy into the container, run with -w /app so './src/...' resolves
docker cp cert-consumption-probe.mjs lah-openclaw-mvp:/app/cert-consumption-probe.mjs
docker exec -w /app lah-openclaw-mvp node cert-consumption-probe.mjs
# 3. Clean up afterwards (probe is a test artifact, not app code)
docker exec lah-openclaw-mvp rm -f /app/cert-consumption-probe.mjs
```

The container has BOTH the deployed code (GIT_COMMIT=deploy-sha) AND the real mounted
store, so the probe exercises the exact production path. Env/secrets come from the
container's own environment — no .env parsing needed.

## Probe shape that works

```js
import { buildCanonicalBusinessContext } from './src/services/cloe-canonical-business-context.js';
import { formatCognitiveContextPack } from './src/brain/cognitive-context-formatters.js';
const ctx = await buildCanonicalBusinessContext({
  requested_offer: 'crakrevenue:10138',
  campaign_context: { provider: 'exoclick', format: 'popunder', geo: 'US', device: 'mobile' },
});
const rendered = formatCognitiveContextPack({ attach_to_prompt: true, intent_tags: ['business_context'],
  compact_summary: ctx.compact_summary, available_items: ctx.available_items,
  safety_flags: ['read_only','no_write','no_execute'], evidence: [] });
// assert: projected_rules length, record IDs (rule[mem-), contract_source,
// AUTHORITY_PRECEDENCE, contract_authority=REFERENCE_ONLY
```

## Companion checks (same session)

- Enumerate the real store BEFORE asserting expectations:
  `docker exec lah-openclaw-mvp sh -c 'grep -h "\"subject\"" /app/data/memory-events/*.json ...'`
  (note: subjects may be nested under `metadata.subject`, not top-level — read full files).
- Verify offer executability to interpret governed verdicts: `getConfiguredOfferInventory`
  in-container; if the requested offer is absent, the governed surface legitimately
  returns `REQUESTED_OFFER_NOT_EXECUTABLE_OR_NOT_GROUNDED` — that is honest fail-closed,
  not a projection defect.
- WS fresh-session probes: the mandated campaign-prep prompt may route `provider_backed`
  (LLM called, projection injected) while a restitution-flavored variant routes
  `supervisor_routed` (governance_question, deterministic) and a pure knowledge question
  routes `memory_boundary` (campaign-memory surface). Capture `cognitive_trace` from the
  chat event's message.metadata to classify which surface answered.
