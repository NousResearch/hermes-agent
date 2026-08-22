# ExoClick Provider-Context → AEC Pattern

Established 2026-08-08 on HERMES_EXOCLICK_PROVIDER_CONTEXT_TO_AEC_REPAIR_V1
(deployed 68e3ffd, PR #734). Use for any mission that must ground Cloé's
business runtime in CURRENT provider readback facts rather than zone
historical statistics.

## Core semantic (the mission's root cause)

`ZONE_STATS_EMPTY_MISCLASSIFIED_AS_PROVIDER_TARGETING_UNAVAILABLE`: the LAH Brain
facade (`lah-brain routes/business-runtime.js`) builds `syncStatus` but does NOT
put it in the returned payload → `provider sync: n/a` is permanent; `zones:
{status:'EMPTY', rows:[]}` was the ONLY targeting signal Cloé saw. A
network-targeted campaign with `zones=[]` looked like "provider targeting
unavailable".

The fix encodes a two-authority vocabulary that MUST stay separate:
- **ZONE_STATS** (historical zone statistics): `OBSERVED | OBSERVED_EMPTY | MISSING`
- **PROVIDER_TARGETING** (current provider readback): `OBSERVED | PARTIAL | MISSING`
- `zones=[]` + valid `zone_targeting` (type 2 / network_selection 0 /
  partner_networks 1) = legitimately network-targeted, NEVER "targeting
  unavailable". The campaign normalizer already treats `zone_ids` as NOT
  required — do not re-introduce a zones-based block downstream.

## Real readback shapes (ExoClick v2, verified live 2026-08-08)

- `GET /campaigns` (list) → `{ result: { "<campaignId>": campaign, ... }, request_metadata: { limit, offset, count } }` — a MAP keyed by id, NOT an array. List view exposes: `id, name, advertiser_ad_type, advertiser_ad_type_label, pricing_model, pricing_model_name, price, calculated_status.{id,status}, variations_counts`.
- `GET /campaigns/:id` (detail) → unwrap via `result.campaign | campaign | data.campaign | result | data`. Detail adds capping/max_daily_budget/run_on_responsive_zones but for existing campaigns does NOT return countries/categories/devices/zones/zone_targeting (47 keys, no targeting arrays) → those surface as UNVERIFIED, never invented.
- IDs: advertiser_ad_type 7=Popunders, 8=Direct Link, 0=Banner-300x250; pricing_model 1=CPC, 2=CPM, 4=Smart CPM. `price` is in CENTS (price 50 = $0.50 bid).
- Campaign `status: 0` + `calculated_status.status === 'Paused'` is the readback pause signal.

## Adapter shape (`src/services/exoclick-provider-context.js`)

`getExoClickProviderContext({ env, fetchImpl, getToken, campaignId })`:
- injectable `getToken` (defaults to canonical `getExoClickAccessToken` from
  exoclick-login.js) + `fetchImpl` so tests stub the /login exchange instead of
  depending on global fetch; campaignId precedence: option > env
  `EXOCLICK_PROVIDER_CONTEXT_CAMPAIGN_ID` > first id from list readback.
- returns `{ ok, available, read_only, connectivity:{status:'AUTH_OK'}, read_timestamp, campaign_authority, zone_stats, provider_targeting:{status, format, advertiser_ad_type, pricing_model, price_bid, geo, devices, categories, zones, zone_targeting, sites, variation, campaign_status, observed}, safety }`.
- Classifier: OBSERVED when available && format_observed && pricing_observed;
  PARTIAL when exactly one of format/pricing; MISSING otherwise. Optional dims
  (geo/device/category/zones) absent → UNVERIFIED — never downgrade OBSERVED
  because of them, never invent a value.
- `zone_stats`: zones field absent → MISSING (unverified); zones:[] →
  OBSERVED_EMPTY (network-targeted, valid); zones non-empty → OBSERVED.

## Injection points

- `cloe-canonical-business-context.js` `buildCanonicalBusinessContext`: push a
  `provider_targeting_context` available_item SEPARATE from
  `business_runtime_context` (zone stats untouched); fail-closed unavailable
  item when read fails; accept `providerContextOptions` pass-through
  (getToken/campaignId) so tests stay deterministic.
- `affiliate-execution-context.js` `getAffiliateExecutionContext`: add
  `provider_targeting` to SECTION_ORDER + `buildProviderTargetingSection`;
  surface `zone_stats_status` inside it for disambiguation; provenance gains
  'exoclick-provider-context'.
- `chat-completions-service.js` `buildNativeChatCompletions`: thread
  `providerContextOptions` to the canonical context call (native-tool path).
- `exoclick-collections.js`: Bearer MUST be `getExoClickAccessToken()`, never
  raw `process.env.EXOCLICK_API_TOKEN` (that produced 401s on valid collection
  endpoints); 404 after correct auth → verdict `ENDPOINT_UNAVAILABLE` — never
  invent a replacement endpoint.

## Test shape (`test/exoclick-provider-context-to-aec.test.js`)

10 mission scenarios + 1 classifier unit. Mock fetch router must route:
`/login` (token), `/campaigns` (list map), `/campaigns/:id` (detail with
`zones:[]` + `zone_targeting` — the NOT-blocked fixture), `/business/runtime-context`
(bridge), `/collections/:name`. Assert: all methods GET-only (no writes),
collections auth header === exchanged token ≠ raw token, 404 → ENDPOINT_UNAVAILABLE,
device absent stays null/UNVERIFIED, normalizer zones=[] semantics unchanged.

When updating the pre-existing parity test (`cloe-native-tool-business-context-parity.test.js`),
its `makeFetch` must route /campaigns + /login and callers must pass
`providerContextOptions` — otherwise `available === true` assertions fail.

## Live certification (post-deploy, in-container probes)

Query the DEPLOYED container (read-only) rather than source:
```bash
docker exec lah-openclaw-mvp sh -c 'cd /app && node -e "import(\"./src/services/exoclick-auth.js\").then(async m => console.log(JSON.stringify(await m.checkExoClickAuth())))"'
```
PASS criteria: connectivity AUTH_OK (login ok + /campaigns 200); the provider
context item available with `provider_targeting.status` OBSERVED or PARTIAL;
zone stats stay OBSERVED_EMPTY/EMPTY in the AEC zones section while
provider_targeting is OBSERVED in the same pack; format/pricing surfaced;
geo/devices UNVERIFIED when provider detail omits them. In-container probe of
`buildCanonicalBusinessContext({})` shows the full pack — the compact_summary
line `zones:OBSERVED_EMPTY, provider_targeting:OBSERVED` cohabiting proves the
semantic distinction in production.

## Pitfalls

- **`GET /campaigns` result is a map keyed by id**, not an array — code that
  does `Array.isArray(body)` misses it; iterate `Object.keys(result)`.
- **Detail readback omits targeting arrays for real campaigns** — an adapter
  that defaults missing geo/devices/zones to `[]` would FALSELY claim
  OBSERVED/EMPTY; use `null` (UNVERIFIED) and classify on format+pricing only.
- **Secret masking in tool output**: test files containing fake tokens
  (`const GET_TOKEN = async () => '...'`, `EXOCLICK_API_TOKEN='raw...'`) may
  render as `***` in tool results — verify the on-disk file with read_file
  before assuming corruption (see secret-masking-file-corruption.md).
- **Continuity JSON closure**: docs/mcporter continuity records are committed
  DIRECTLY to main as a single-parent doc commit (`docs(cloe): ... continuity
  record (CERTIFIED)`), pushed via `git push origin-https <sha>:main` — no PR,
  no redeploy for doc-only closure. Verify `git merge-base --is-ancestor
  origin-https/main <sha>` before pushing.
