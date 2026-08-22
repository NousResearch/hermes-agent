# LAH Brain Route Test Harness — Pitfalls (tracking/analytics routes)

Session-proven from extending `test/tracking-ingestion-readonly-access.test.js` with
`tracking-identity.test.js` + `behavioral-summary.test.js` (CLOE_TRACKING_IDENTITY_AND_READONLY_BEHAVIORAL_SUMMARY_WIRING_V1).

## Harness shape (copy from the existing readonly-access test)

- Set env (`ADMIN_PASSWORD`, `SESSION_SECRET`, `INGEST_SECRET`, `OPENCLAW_ADMIN_API_KEY`,
  `SQLITE_PATH`, `PORT=0`, `LAH_TRACKING_BASE_URL`, `DEFAULT_OFFER_URL`) BEFORE any module load.
- mkdtemp fixture DB; `unloadModules()` cache-bust (config/db/server/routes) between fixtures.
- ephemeral-port express app; requests with `x-admin-api-key` header; assert 401 when absent.
- `seedEvent(db, overrides)` via `insertEvent`; close seed DB before `startApp`.

## Pitfall 1 — placeholder fixtures must set ALL attribution fields, not one

`seedEvent` fills unset fields with real defaults (`campaign-real-1`, `zone-real-1`...). A
fixture that only overrides `campaign_id: "unknown-campaign"` still has VALID zone/site/click
→ semantic counters count it → test expects 2 valid but gets 4. For a "placeholder event" set
EVERY attribution field to a placeholder:
```js
seedEvent(db, { event_id: "evt-ph", click_id: "unknown-campaign", campaign_id: "unknown-campaign",
  zone_id: "unknown-zone", site_id: "unknown-site", external_click_token: "unknown-token" });
```

## Pitfall 2 — `zone_name`/`site_name` defaults leak "unknown-zone" into the response

The composite row name is `"zone_id | zone_name"`. If fixtures don't set `zone_name`,
`cleanDimension`/import default fills `"zone-bs-1 | unknown-zone"` → assertion
`!text.includes("unknown-zone")` fails even though NO unattributed SEGMENT is ranked
(the ID part is valid). Fix the FIXTURE: pass real `zone_name`/`site_name`. `isAttributedRow`
checks the FIRST part (the stable ID) — that logic is correct.

## Pitfall 3 — `body.bytes` is the serialized SUMMARY only, not the HTTP response

The route spreads `...summary, provider_write:false, db_write:false, bytes` → `body.bytes`
equals the summary serialization, but the full response text includes those extra fields.
Assert `body.bytes <= 12*1024` AND `Buffer.byteLength(text) <= 12*1024` separately; do NOT
assert `body.bytes === responseBytes` (2218 !== 2271 in practice).

## Pitfall 4 — `process.env.X = process.env.X || default` lines in a test header

The write_file tool's secret-masking can mangle lines containing
`ADMIN_PASSWORD=`/`SESSION_SECRET=` literals (displayed as `proces...WORD` / `***`), and my
own patch attempts made it worse (duplicated `PORT` line). After writing such a header,
re-read the file and `node --check` it before running; fix any mangling with patch, and
remove duplicate lines if a blind patch duplicated them.

## Pitfall 5 — env-var delete semantics in withEnv helpers

`withEnv(url = LAHB_URL, key = LAHB_KEY)` with `withEnv(undefined, ...)` does NOT unset the
var (default applies). Use object args with a null sentinel:
```js
function withEnv({ url = LAHB_URL, key = LAHB_KEY } = {}) {
  const before = { url: process.env.LAHB_URL, key: process.env.LAHB_ADMIN_API_KEY };
  if (url === null) delete process.env.LAHB_URL; else process.env.LAHB_URL = url;
  if (key === null) delete process.env.LAHB_ADMIN_API_KEY; else process.env.LAHB_ADMIN_API_KEY = key;
  return () => { /* restore */ };
}
// usage: withEnv({ url: null }) → client must return LAHB_URL_REQUIRED
```

## Pitfall 6 — makeFetch must default `calls = []`

A mock fetch that does `calls.push(...)` with no default crashes (`Cannot read properties of
undefined (reading 'push')`) when the test doesn't pass `calls`. Default it:
```js
function makeFetch({ status = 200, body = makeSummaryResponse(), calls = [], ... } = {})
```

## Pitfall 7 — route uses the shared reportCache, not a fresh compute

`GET /admin/analytics/behavioral-summary` calls `reportCache.getReport({refresh:false})` —
first call computes via `analyzeEvents` (cache miss), so a seeded DB DOES produce a report.
No need to pre-warm the cache in tests. Fail-closed 503 can be simulated by monkey-patching
`serverMod.reportCache.getReport = () => { throw ... }` then restoring.

## OpenClaw side (client tests)

`getSecret('LAHB_ADMIN_API_KEY')` reads `process.env` directly — set/delete env around each
test with the null-sentinel helper. Assert: single fetch call (no retry storm), `redirect:
'manual'` in init, `x-admin-api-key` header equals the key, key NOT in URL. For
`cloe-native-tool-business-context-parity.test.js` when adding a NEW bridge: extend the
mock fetch with a branch for the new path AND extend the `providerCalls` filter to exclude
it — otherwise `provider payload captured` fails because the first "provider" call is now
the bridge GET.
