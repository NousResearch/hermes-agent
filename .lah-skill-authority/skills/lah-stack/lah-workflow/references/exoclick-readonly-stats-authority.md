# ExoClick Read-Only Stats Authority & Historical Corpus Facts

Mission lineage: CLOE_EXOCLICK_HISTORICAL_COST_RECONCILIATION_V1 (2026-08-01, verdict **BLOCKED**), built on
CLOE_EXOCLICK_OURDREAM_ZONE_INTELLIGENCE_PREFILTER_V1 and the clickstream RMS analysis (LAH_HISTORICAL_CLICKSTREAM_RMS_ANALYSIS_V2).

## 1. Read-only stats access boundary (root cause of the BLOCKED verdict)

ExoClick financial statistics are NOT retrievable in read-only missions:

- Endpoints `/statistics/a/campaign`, `/statistics/a/site`, `/statistics/a/zone` require an **access token** obtained by
  exchanging `EXOCLICK_API_TOKEN` via **POST `/v2/login`** (`exoclick-login.js` → `loginExoClickWithApiToken()`).
- POST is explicitly prohibited in read-only mission specs (Gate 0: "Explicitly prohibit: POST; PUT; PATCH; DELETE").
  The POST /login is an identity exchange, not a resource mutation — but the mission spec forbids the method itself,
  so the mission must fail closed rather than treat login as a safe exception.
- `EXOCLICK_LIVE_ENABLED=false` is a **permanent doctrine** (AGENTS.md: "Opening requires explicit operator
  instruction + explicit gate window"), not a config flag. It gates mutations in `executor.js` / `exoclick-client.js`
  (`LIVE_DISABLED` / `DRY_RUN_BLOCKED`). The stats functions (`exoclick-stats.js`) are NOT gated by it — they are
  blocked only by the missing access token.
- `GET /collections/*` (`exoclick-collections.js`) sends the raw `EXOCLICK_API_TOKEN` as Bearer without login —
  technically callable read-only, but collections contain only reference data (categories, countries, ad types),
  NO financial stats.
- Local campaign data only covers 8293490 (protected), 8304842 (protected), 8308460 (canonical clean). Never touch
  the first two.

Correct handling when cost data is required in a read-only mission (the `EXOCLICK_READ_ONLY_AUTHORITY_MAP.md` pattern):
1. Document the authority map (client, auth adapter, stats endpoints, mutation gate, local-data authority).
2. Prove whether authenticated GET can run independently of the mutation gate (here: collections yes, statistics no).
3. Do NOT open the gate. Stop before any capability that could authorize mutation.
4. Produce artifacts with explicit nulls; classify every zone `INSUFFICIENT_FINANCIAL_EVIDENCE` (fail-closed) rather
   than inventing BLOCK/ALLOW from engagement alone.

Two operator-approved paths to complete the financial join later:
- **Option A**: explicit operator authorization of POST /v2/login in strict read-only mode (exchange + GET statistics
  only for the corpus window, never mutation).
- **Option B (recommended)**: operator exports ExoClick campaign/site/zone stats (window 2026-03-23 → 2026-04-16)
  into `/home/deploy/exoclick-imports/` — zero API access, doctrine-compliant.

## 2. Historical clickstream corpus facts

- Handoff: `/home/deploy/LAH_HISTORICAL_CLICKSTREAM_HERMES_HANDOFF_V2.json`
  (SHA-256 `f8349e81e6c80b338ca9ccbb1ff4a9a9d04d32b140e8b332674dd3351797bec8`).
  22,537 unique events · 8,414 campaign/site/domain-attributable · 3,057 with real zone IDs · 56 distinct zones ·
  662 real click IDs · 253 campaign/site/zone/domain aggregates.
- Corpus campaigns (8196420, 8198248, 8210662, 8210664, 8240920, 8240946) are **absent from all local data**
  (snapshots, campaign-ledger, decision.log, runtime logs). Offer and targeting for these campaigns are UNKNOWN —
  they predate the May 2026 microtest and are NOT the known local campaigns.
- The clickstream carries NO spend, NO billed impressions, NO revenue, NO paid conversions. CTA clicks and redirects
  are engagement signals only — never treat them as purchases.
- 56 zones each map to exactly one domain (no multi-domain zones observed in the aggregates).
- Lot 2 (6 files) added ~50 routing/offer fields (`routed_*`, `offer_*`, `domain_score`, `zone_id`, `click_id`,
  plus `tester`, `test_reason`, `is_test`) — possible test traffic, unconfirmed; flag as ENVIRONMENTAL_OR_TEST.

## 3. Engagement-only classification discipline

When financial data is unavailable:
- Engagement ratios (interaction rate per landing load, CTA rate, redirect rate) ARE computable.
- Spend-derived metrics (spend per CTA, spend per redirect, spend per load, CPM, CPC) MUST stay null.
- CPA / ROAS / profit / ROI are forbidden without real conversions + revenue.
- A zone with weak engagement cannot be BLOCK_INITIAL without cost (it might have been cheap); a zone with strong
  engagement cannot be ALLOW_INITIAL without cost (it might have been expensive). Keep both as
  `INSUFFICIENT_FINANCIAL_EVIDENCE` with engagement evidence documented for operator review.

## 4. NDJSON ingestion pipeline (lah-brain) for operator exports

Operator exports (NDJSON or CSV) land in `/home/deploy/exoclick-imports/`:
```bash
cd /home/deploy/lah-stack-repos/lah-brain
node src/cli.js import:dry-run /home/deploy/exoclick-imports/<file>.ndjson   # validate
node src/cli.js analyze /home/deploy/exoclick-imports/<file>.ndjson          # summary
node src/cli.js import /home/deploy/exoclick-imports/<file>.ndjson           # real import
```
Format: one JSON object per line (NDJSON). Parser: `src/ndjson.js` (`readNdjsonForImport`,
`parseNdjsonTextForImport`). HTTP routes also exist: `POST /imports/ndjson`, `/imports/exoclick`,
`/imports/exoclick-stats-json`. Reference sample: `data/sample-events.ndjson` (fields: timestamp, network,
campaign_id, zone_id, site_id, domain, ad_format, device, geo, event_type, cost …).

## 5. Evidence-directory artifact generation (SHA256SUMS pitfall)

When producing the mission evidence directory + SHA256SUMS (the `geo-opportunity-matrix/<mission>/` pattern):
- **SHA256SUMS self-inclusion**: `find . -type f \| xargs sha256sum > SHA256SUMS` fails on the SECOND
  regeneration — the redirection truncates SHA256SUMS while `find` is still scanning, so the checksum file ends up
  containing an entry for itself and `sha256sum -c` reports FAILED. First pass works only because the file didn't
  exist yet. Fix: write to a temp file then move:
  `find . -type f ! -name SHA256SUMS -print0 \| sort -z \| xargs -0 sha256sum > /tmp/sha256.tmp && mv /tmp/sha256.tmp SHA256SUMS`,
  then verify with `sha256sum -c SHA256SUMS` and count `: OK` lines.
- Regenerate SHA256SUMS AFTER any late edit (e.g. updating `evidence/VALIDATION_REPORT.json`), or the checksum goes stale.
- Artifacts live outside git repos (read-only missions never write to canonical checkouts); the mission directory
  under `~/.hermes/openclaw/opportunity-reports/geo-opportunity-matrix/<mission>/` is the durable home.

## 6. Zone prefilter facts (OurDream 10138)

- No previous site/zone prefilter ever existed — historical targeting was category + whole network
  (`zone_targeting.type=2`, empty sites/zones). Any V1 is a from-scratch build; declare the absence explicitly.
- Category 526 = "Adult - AI" (selectable) is the PRIMARY targeting category for the AI-companion offer; adjacent:
  492 Dating, 506 Erotic/Sexy, 110 Cartoons/Hentai, 524 Webcams, 515 VR, 517 Manga. Category 97 (Amateur) was used
  historically but is NOT the AI-companion target.
- Prefilter V1 (30 class-level entries, CERTIFIED_WITH_LIMITATIONS) stays the decision authority; V1.1 (56
  zone-level entries, all INSUFFICIENT_FINANCIAL_EVIDENCE) is the zone layer awaiting financial data. Never apply
  either remotely.
