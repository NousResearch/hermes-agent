# LAH Affiliate Postback Outcome Chain (lah-core → lah-brain)

Canonical map of the inbound affiliate postback chain, certified during
CLOE_AFFILIATE_OUTCOME_SYNTHETIC_POSTBACK_E2E_CERTIFICATION_V1 (2026-08-11).
Load before ANY mission touching affiliate outcomes, postbacks, payout
semantics, or the zone economics contract.

## Chain map (H1..H7) with exact files

- H1 provider→LAH: `CRAKREVENUE_POSTBACK_CONFIGURATION_UNVERIFIED` until a real
  callback is observed. Synthetic tests NEVER prove provider configuration.
- H2 normalize: `https://liveaccesshub.com/lah-postback/affiliate` →
  `wp-content/plugins/lah-core-router/includes/class-lah-conversion-endpoint.php`
  → `LAH_Utils::normalize_affiliate_conversion_request` (class-lah-utils.php).
  Auth: `X-LAH-SECRET` header preferred; legacy `?secret=` compat path; secrets
  in WP option `lah_affiliate_postback_secrets` — NEVER print.
- H3 persist/dedupe: `LAH_Conversion_Store::store_or_get`
  (class-lah-conversion-store.php). dedupe_key = `tx:<network>:<transaction_id>`
  (UNIQUE index). payout NULL survives (column nullable). Row-level
  campaign/zone/site columns DO NOT exist (dims live in
  normalized_payload_json + the S2S payload only).
- H4 S2S: `LAH_S2S_Push::push_conversion` → `build_sale_event` → `dispatch`
  (class-lah-s2s-push.php). Event persisted to `lah_retry_queue`
  (SHADOW_PENDING) BEFORE the fire-and-forget `wp_remote_post`
  (blocking=false, timeout=1). E-2.1 stable event_id.
- H5 LAHB ingestion: `POST /events` (src/lah-http.js, `x-ingest-secret`) →
  `adaptIncomingEvent` → `liveaccesshub-adapter.js adaptLiveAccessHubEvent`
  (**whitelist rebuild — drops unknown fields**) → `event-schema.js
  normalizeEvent` → `db.js insertEvent/runInsert` →
  `projectCanonicalSaleEvent` (src/money/conversion-importer.js) →
  `affiliate_conversions` table.
- H6 reconciliation: `src/money/reconciliation.js` — priority:
  direct_postback_dimensions (Mode B) > click_event > external_token_event;
  `WHERE status != 'rejected'` excludes rejected rows from revenue math.
- H7 readiness: `/business/runtime-context` zone contract
  (routes/business-runtime.js `buildZoneOutcomeBreakdown`) — paid_conversions
  counts ONLY payout>0 rows; downstream_events = all non-rejected rows.

## Outcome semantics contract (5 rules — encoded in the 2026-08-11 repair)

1. missing payout ≠ payout 0 — absent stays null at every hop
2. payout 0 ≠ proof of sale
3. sale requires explicit positive semantics: `event_semantics = PAID` when
   payout > 0, else `UNVERIFIED`; event_type is 'sale' ONLY when PAID,
   otherwise 'conversion' (never auto-promote the LAH 'conversion' constant)
4. null stays null until proof otherwise
5. synthetic events stay identifiable (is_test=1, source=synthetic_e2e, SYNTH_
   tx prefix) and excluded from KPI: canonical projection stores
   status='rejected' → reconciliation + zone contract exclude it; event
   listings exclude is_test by default (`isTestTrafficEvent`)

## Defect map (coercions that used to break this chain)

- s2s-push `build_sale_event`: `is_numeric($payout) ? (float)$payout : 0.0` +
  hardcoded 'sale' → null-preserving payout, conditional type/semantics, added
  is_test/source transport.
- `normalizeEventType` `"conversion" → "sale"` mapping — remove at BOTH sites:
  src/event-schema.js AND src/liveaccesshub-adapter.js
  (`normalizeCanonicalEventType`).
- event-schema payout: `toNumber(payout || revenue)` → nullable; an EXPLICIT
  null must NOT fall back to revenue (the adapter always sets revenue=0 → the
  fallback silently turns null into 0). Fallback to revenue only when the
  payout key is absent/"".
- liveaccesshub-adapter whitelist rebuild drops new fields: a new event field
  must be added in FOUR places — adapter event object, event-schema
  normalizeEvent, ALLOWED_KEYS (validateEventPayload REJECTS unknown keys),
  and EVENT_COLUMNS (+ DDL + ensureColumns + rowToEvent in db.js).
- conversion-importer `normalizeConversionRow`:
  `Number(obj.payout ?? revenue ?? amount ?? 0)` → missing payout now returns
  an error (fail closed), never coerced to 0.
- events table `payout REAL NOT NULL DEFAULT 0` blocks NULL persistence →
  SQLite table-rebuild migration (below).

## SQLite table-rebuild migration (drop NOT NULL on a column)

SQLite `ALTER TABLE` cannot remove NOT NULL. Pattern used in db.js
`migrateEventsPayoutNullable()`:
`PRAGMA table_info` check (notnull==1) → BEGIN →
`ALTER TABLE events RENAME TO events_old` → `CREATE TABLE events` (new DDL) →
`INSERT INTO events (<shared cols>) SELECT <shared cols> FROM events_old` →
`DROP TABLE events_old` → COMMIT.
Run AFTER `ensureColumns` (so new columns exist in the old table before the
copy — build `shared` dynamically from PRAGMA, don't hardcode) and BEFORE
`createIndexes` (indexes are dropped by the rebuild and recreated after).

## Pitfalls (each cost real cycles)

- JS `//` comment inside a SQL template literal → `near "/": syntax error`.
  Keep DDL strings comment-free.
- PHP WP-stub sim pattern (lah-core/test/*-simulation.php): define stubs
  (WP_Error, sanitize_key, wp_unslash, wp_remote_post capture,
  LAH_Retry_Queue, constants) BEFORE `require_once` of the real classes;
  assert on captured JSON bodies.
- lah-brain HTTP harness: reuse `loadHandlerForFixture`/`invokeHandler` from
  test/beacon-public.test.js (real `createLahRequestHandler` + temp SQLite via
  SQLITE_PATH env). A `.mjs` harness under tools/operator-validation needs
  `../../src/...` (one level deeper than the app dir) — MODULE_NOT_FOUND on the
  first `require('../src/...')` line is the signature of wrong relative depth.
- Secret topology: the postback secret exists ONLY in the WP DB option
  `lah_affiliate_postback_secrets` (remote server) — never on the VPS, never
  printed. LAHB read endpoints (/events/recent, /reconciliation,
  /business/runtime-context) are session-cookie auth-gated; /version is public.
  lah-mcp-bridge is strictly read-only (7 GET tools, no option reads) → real
  postback trials require the operator to provide the secret via a 0600
  file/env; otherwise certify with the deterministic harness + the endpoint
  response (H2/H3 + dedupe) only.
- Existing tests asserting the OLD contract (conversion→sale, payout-absent→0)
  must be updated WITH the repair, not left red: sale-financial-projection
  fixtures needed `event_semantics:'PAID'`; beacon-public conversion test now
  expects event_type stays 'conversion'.
- `status=conversion` is a LAH-side constant, never provider truth — never
  derive sale/non-paid from it alone.
