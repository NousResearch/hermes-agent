# ExoClick campaign preparation — device & SubID wiring authority

Established during CLOE_EXOCLICK_OURDREAM_DEVICE_SUBID_WIRING_REPAIR_V1 (merged 2026-08-01, PR #636, 1eb22ed1). Reuse for any ExoClick campaign preparation mission (ourdream microtest, other offers).

## Device → ExoClick device_id

- Authority: lah-brain `DEVICE_ID` (src/providers/exoclick/builder.js): **desktop=1, mobile=2, tablet=3**.
- Local mirror in openclaw-runtime: `lah-openclaw-mvp/data/refs/device_name_to_id.json` (same format as country/category refs; each entry documents `authority: lah-brain DEVICE_ID.*`).
- Payload field: `devices: { type: 'targeted', elements: [id] }` (same shape as `countries` / `categories`).
- Fail-closed rules (in `resolveDeviceIds()` in exoclick-normalizer.js):
  - unknown device → explicit error (`Unknown device "<v>". Allowed: desktop, mobile, tablet`)
  - `all` → rejected (no implicit all-devices fallback)
  - multiple distinct devices → rejected (mono-device contract)
  - case normalized (`Mobile` → `mobile`)

## SubID1 CrakRevenue → destination URL

- CrakRevenue network parameter for SubID 1: **`aff_sub1`** (NOT `subid1`). Source: OURDREAM_EXOCLICK_TRACKING_CONTRACT.json — `destination_url: "[CRAKREVENUE_TRACKING_LINK_MASQUE]&aff_sub1=us_exo_pop_mob_t01"`.
- `aff_sub5` = PROHIBITED_RESERVED_BY_NETWORK — never written, rejected explicitly (`CRAKREVENUE_PROHIBITED_PARAM`).
- Builder: `lah-openclaw-mvp/src/services/crakrevenue-tracking-url-builder.js`:
  - `validateSubid1`: non-empty, no whitespace, ≤64 chars, charset `[A-Za-z0-9_.-]`
  - inject `aff_sub1` exactly once (skip if URL already carries it — no dup)
  - preserve existing query params + host; encode via URLSearchParams
  - reject malformed URL / non-http(s) protocol
- Redaction: `redactDestinationUrl` → `[CRAKREVENUE_TRACKING_LINK_MASQUE]` + query string. Apply to receipts (`redactNormalizedForReceipt` in exoclick-normalizer.js, used at 3 points in executor.js) and route logs (server.js `/exoclick/normalize-payload` + `/exoclick/dry-run`).

## Budget/bid null (microtest contracts)

- Microtest contracts carry `bid: null`, `daily_budget: null`, `total_budget: null` by HUMAN DECISION ("aucun budget ni bid inventé").
- Therefore `normalizeExoClickCampaign` and `buildExecutePayloadFromGovernedAction` do NOT require bid/budget for PREPARATION.
- The real-send guard stays in `runExoClickCampaignCreatePreflight`: popunder requires `max_daily_budget >= 2000` cents. So a null-budget payload normalizes fine but is blocked at creation time — intended.

## Testing client behavior (mock fetch)

`createExoClickCampaignPaused` calls `/v2/login` FIRST (exchange EXOCLICK_API_TOKEN → access token) before the POST /campaigns call. When mocking fetch for client tests with `EXOCLICK_LIVE_ENABLED=true`:
- for URLs containing `/login` return `{ token: 'mock-access-token', expires_in: 3600 }`
- capture `JSON.parse(opts.body)` only for the non-login call
- with `EXOCLICK_LIVE_ENABLED=false` + fetch mock that flags calls: assert `LIVE_DISABLED`, `live_sent: false`, and fetch NOT called (proves zero remote mutation).
- Preflight-valid payload needed to reach the live gate in tests: `{ type:'popunder', pricing:{model:2}, max_daily_budget:2000, total_budget_limit:2000 }`.

## Microtest ourdream reference values

- Offer ourdream.ai - PPS · CrakRevenue offer 10138 · payout $44 PPS · event: spending
- GEO US · format popunder (advertiser_ad_type 7, media_storage_template link) · category 2 (Adult)
- device mobile → device_id 2 · subid1 `us_exo_pop_mob_t01` (17 chars URL-safe)
- campaign name `OURDREAM_US_EXO_POP_MOBILE_PPS44_T01_202608` · status 0 (paused)
- ExoClick macros for tracking URLs: `{clickid}`, `{conversions_tracking}`, `{zone_id}`, `{site_id}`, `{src_hostname}`
