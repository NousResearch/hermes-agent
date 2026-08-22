# Read-only Marketplace Dataset Reader Pattern (CLOE_READONLY_MARKETPLACE_DATASET_READER_V1)

Established 2026-08-09 on openclaw-runtime lah-openclaw-mvp (PR #740 8b1c152 + fix #741 1d84f1f, deployed 1d84f1f, live cert A/B/C PASS).

## When to use

When Cloé needs bounded read-only access to a durable research dataset (CSV snapshot)
already stored in the canonical repo, WITHOUT injecting the full catalog into every
prompt. First instance: CrakRevenue 2026-08-09 snapshot, 326 consolidated offers,
9 vertical files, `data/business/affiliate-networks/crakrevenue/2026-08-09/`.

Architecture: `data/business/affiliate-networks/ → marketplace-dataset-reader.js →
bounded query result → buildCanonicalBusinessContext item (marketplace_dataset) →
formatItem render → Cloé cognitive context → DeepSeek/Cloé`.

## Components (all additive, zero regressions to configured_offer_inventory_v1)

1. **src/services/marketplace-dataset-reader.js** — pure ESM, no deps beyond node:fs/path/url.
   - Dataset root resolved MODULE-relative (`fileURLToPath(new URL('../../data/...', import.meta.url))`),
     NEVER cwd-dependent (tests and the container run from different cwds).
   - Allowlist files only (manifest.json + vertical CSVs + consolidated.csv); reject absolute
     paths, `..`, symlink escapes, arbitrary filenames, `.env`/secret/token patterns.
   - Quoted-field-aware CSV parser (countries fields contain internal commas) — a naive split breaks.
   - Ops: listVerticals, topOffers(vertical, limit), findOffer(query), filterOffers(criteria),
     compareOffers(titles), shortlist(criteria, limit). Bounds: max files 2/query, max rows 20,
     max compare 10, max evidence chars 6000; expose `{ total_matches, returned_count, has_more }`.
   - Payout semantics: parse numeric at read time only, preserve RAW strings, expose ov AND ap
     separately with `conflict: true` when they differ — NEVER merge, never pick a winner.
   - Provenance on every result: network, snapshot_date, authority (RESEARCH_SNAPSHOT),
     provider_api_verified=false, marketplace_freshness=UNVERIFIED, source_file, source.
   - No write methods, no delete, no shell, no network (prove with a test that swaps
     `globalThis.fetch` for a throwing stub and asserts zero calls).
   - In-memory cache keyed by `(file, mtimeMs, size)` — metadata invalidation, read-only.

2. **src/services/marketplace-query-interpreter.js** — conservative deterministic FR/EN
   interpreter. FAIL-CLOSED: returns null unless a recognized vertical name OR an explicit
   query verb is present — ordinary business prompts (e.g. microtest proposals) must never
   trigger a dataset read. Test the null case explicitly.
   - Regex gotchas learned: `\d{1,2}` misses 3-digit limits ("Liste les 149 offres dating")
     → use `\d{1,3}` + clamp; country detection must exclude vertical-like tokens (`AI`, `IA`);
     split compare lists on `[,;]|\bet\b|\band\b` (FR `et` AND EN `and`).

3. **src/services/cloe-canonical-business-context.js** — additive item `marketplace_dataset`:
   compact summary (verticals + counts + provenance) always when dataset available; query-scoped
   bounded results when `marketplaceQuery` passed in. Fail-closed `available:false` item when
   dataset missing.

4. **src/brain/cognitive-context-formatters.js** — additive branch in `formatItem` for
   `kind === 'marketplace_dataset'` (same pattern as `offers_executable=` from PR #738).
   NEVER render preview_url / URLs in the prompt (url_redacted style).

5. **Wiring (2 call sites)**: `chat-completions-service.js` (native-tools path) and
   `readonly-operator-cli-client.js` (brain path) both derive
   `marketplaceQuery = interpretMarketplaceQuery(latestUserPrompt)` and pass it into
   `buildCanonicalBusinessContext`.

## ⚠ THE TRAP: MAX_ITEMS=10 renderer slice (cost a live-cert failure + fix PR #741)

**Symptom:** live certification B ("top 7 AI") returned "je ne peux pas te sortir un top 7
fiable… le détail n'est pas exposé dans mon contexte" even though the pack contained the query
item. Local tests passed because they rendered the business pack alone (7 items).

**Root cause:** `buildBrainAskResponse` composes its OWN items (canonical_evidence_dossier,
project_knowledge, v5_decision_context, web_search…) BEFORE appending the business pack's
7 items. `formatCognitiveContextPack` renders only `slice(0, MAX_ITEMS=10)`. The query item
pushed LAST with `push()` landed at index ≥10 → invisible to the provider.

**Fix:** inject query-scoped items with `unshift()` (or reorder early in `available_items`).
**Regression test:** build a DENSE pack — 5 dummy items + real business pack items (>10 total) —
and assert `query=`, `total_matches=`, `results=` are still rendered.

## Live certification harness notes

- Container API: `POST /brain/ask` with header `x-admin-api-key` (NOT `x-admin-key`) — 401
  `AUTH_REQUIRED` otherwise. Key read from `process.env.ADMIN_API_KEY` inside the container
  (env_file injection; `.env` file is NOT present at /app/.env).
- Expected live answers: A → "326 consolidated + 9 verticals correct"; B → bounded rows with
  ov/ap separate, conflict flagged (Get-Harder 50/34), sources BOTH/OfferVault preserved,
  "aucune action effectuée"; C → "Non" — RESEARCH_SNAPSHOT, provider_api_verified=false,
  UNVERIFIED, and Cloé distinguishes the research snapshot from the configured offer inventory.
- Cloé may honestly note the vertical-sum-vs-consolidated discrepancy (359 vs 326 — vertical
  files overlap) — that is correct epistemic behavior, not a defect.

## Deployment nuance (canonical exact-SHA deploy)

The canonical deployer (`bin/deploy-lah-openclaw-mvp-exact-sha.mjs`) FORCES the data mount to
`CANONICAL_DATA_SOURCE = /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/data`.
If the canonical checkout is parked on a branch that predates the dataset merge, `data/business/`
does NOT exist there and the running container cannot see the dataset even though the image
contains it. Before deploying a mission that adds tracked data files, ensure the dataset exists
in the canonical data dir (extract from the merge commit) — the container mounts that dir over
`/app/data`. The reader resolves module-relative → `/app/data/business/...` at runtime.

## Tests to keep

- Reader A–Q (manifest, count=326, list_verticals, top_offers deterministic, find partial/
  case-insensitive, GEO filter, payout numeric filter, conflicting payouts separate, provenance
  fields, max bound + has_more, path traversal rejected, arbitrary fs impossible, CSVs never
  mutated via content hash before/after, no network, no secret/URL field leak).
- Integration 1–5 (41 AI, top-5 bounded + provenance, compare observed-only, NOT guaranteed,
  149 dating → bounded + has_more) + dense-pack renderer regression.
- Regression: cloe-offer-inventory-context-renderer, cloe-native-tool-business-context-parity,
  exoclick-provider-detail-surfacing, exoclick-provider-context-to-aec (all call
  buildCanonicalBusinessContext — the additive item must not break their `every available` checks;
  they pass because the dataset is present in the worktree, so the summary item is available:true).
