# ExoClick Device + SubID Wiring Authority (CLOE)

Established during `CLOE_EXOCLICK_OURDREAM_DEVICE_SUBID_WIRING_REPAIR_V1` (2026-07-31).
Use when a CLOE ExoClick campaign-preparation mission touches device targeting,
CrakRevenue SubID, destination-URL redaction, or paused enforcement.

## 1. Device → ExoClick payload

- **Authority:** lah-brain `DEVICE_ID` (src/providers/exoclick/builder.js): desktop=1, mobile=2, tablet=3 (tested).
- **Mirror:** openclaw-runtime cannot import lah-brain (separate repos). Mirror in
  `lah-openclaw-mvp/data/refs/device_name_to_id.json`, same format as
  `country_iso2_to_id.json` / `category_name_to_id.json`, each entry carrying an
  `authority` field pointing at lah-brain. This is the repo convention — do NOT
  hardcode the map in the normalizer.
- **Payload shape:** `devices: { type: 'targeted', elements: [id] }` — same shape as `countries`/`categories`.
- **Fail-closed rules (implemented in `resolveDeviceIds`):**
  - unknown device → explicit error `Unknown device "<v>". Allowed: desktop, mobile, tablet`
  - `all` → rejected (NO implicit all-devices fallback)
  - mobile+desktop in a mono-device contract → rejected (distinct ids > 1 = error)
  - device missing → BLOCKED_MISSING_FIELDS (required field)
  - case normalization: `Mobile` → `mobile`

## 2. SubID1 CrakRevenue → destination URL

- **CrakRevenue SubID1 param = `aff_sub1`** (NOT `subid1`). Authority: `OURDREAM_EXOCLICK_TRACKING_CONTRACT.json` (`[CRAKREVENUE_TRACKING_LINK_MASQUE]&aff_sub1=us_exo_pop_mob_t01`).
- `aff_sub5` = PROHIBITED_RESERVED_BY_NETWORK — never written, rejected if present in an input URL (code `CRAKREVENUE_PROHIBITED_PARAM`).
- Builder: `src/services/crakrevenue-tracking-url-builder.js` — `validateSubid1` (non-empty, no whitespace, ≤64 chars, charset `[A-Za-z0-9_.-]`), `buildCrakRevenueDestinationUrl` (inject exactly once, skip if already present, preserve existing query params + host, encode via URLSearchParams), `redactDestinationUrl`.
- SubID pipeline propagation: contract `tracking.subid1` → draft `campaign.subid1` (+ draft hash) → governed packet `target.subid1` → adapter `payload.target.subid1` → executor → `normalizeExoClickCampaign(target)` → destination_url with `aff_sub1`.

## 3. Redaction points (full URL must never leak)

1. Normalizer result: expose `destination_url_redacted` = `[CRAKREVENUE_TRACKING_LINK_MASQUE]?<query>` (host/path masked, query kept — subid is not a secret).
2. Executor receipts: `redactNormalizedForReceipt(normalized)` at ALL 3 `raw_provider_response` writes (NORMALIZATION_FAILED, PREFLIGHT_REJECTED, SUCCESS).
3. Server routes `/exoclick/normalize-payload` and `/exoclick/dry-run`: build `logOutput` with redacted destination_url BEFORE `logEvent` (the raw output is fine for the HTTP response, not for the log).

## 4. Null-budget path (microtest contracts)

Microtest budgets/bid are null by human decision ("aucun budget ni bid inventé").
- Normalizer `required` list and governed-adapter `createRequired` must NOT include `bid`/`daily_budget`/`total_budget` — otherwise preparation blocks before reaching the real-creation guard.
- The real-send guard stays in `runExoClickCampaignCreatePreflight` (popunder requires `max_daily_budget >= 2000` cents) + `validateExecutePayload` + `EXOCLICK_LIVE_ENABLED=false`.

## 5. Test notes

- `createExoClickCampaignPaused` with mocked fetch + `EXOCLICK_LIVE_ENABLED=true`: the mock MUST answer `/login` first with `{ token: 'mock-access-token', expires_in: 3600 }`, then the campaign POST — otherwise verdict `LOGIN_FAILED` before status forcing is exercised.
- To reach the LIVE gate in tests, the payload must pass the popunder preflight (`max_daily_budget >= 2000` cents, `pricing.model` in {2,4,8}) — otherwise it stops at `PREFLIGHT_REJECTED`.
- Numeric category id accepted directly: contract field `category: 2` should resolve via `Number(raw)` check before name lookup.

## 6. Full-suite test glob (lah-openclaw-mvp)

- Bare `node --test --test-concurrency=1` (no path) hangs: it executes `test/fixtures/runner-stdin-echo.mjs` (stdin-echo fixture that never exits) and scans `releases/` archives (2395 duplicate test files).
- Canonical full-suite command: `node --test --test-concurrency=1 test/*.test.js test/*.test.mjs` (501 active files, ~5 min).
- Baseline at HEAD 9958e41 (2026-07-31): 123 FAIL / 9211 PASS — far more than the 2 documented pre-existing failures. Classify full-suite failures against a baseline run of the SAME glob before claiming non-regression.
