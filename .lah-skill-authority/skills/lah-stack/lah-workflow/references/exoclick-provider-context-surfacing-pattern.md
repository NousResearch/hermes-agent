# ExoClick Provider-Context Surfacing to Cloé (data path + serialization chokepoint)

Established 2026-08-08 (HERMES_EXOCLICK_PROVIDER_CONTEXT_TO_AEC_REPAIR_V1 → 68e3ffd PR#734,
HERMES_EXOCLICK_PROVIDER_DETAIL_SURFACING_TO_CLOE_V1 → bf12260 PR#735, both CERTIFIED).

## The data path (trace before assuming the adapter is the problem)

```
exoclick-provider-context.js   (getExoClickProviderContext — collects the detail)
  → cloe-canonical-business-context.js  (buildCanonicalBusinessContext — item.metadata.provider_targeting_detail)
  → cognitiveContextPack (available_items)
  → cognitive-context-formatters.js  (formatCognitiveContextPack → formatItem)   ← SERIALIZATION CHOKEPOINT
  → provider system context (system message)
```

Both Cloé paths consume the SAME formatter:
- non-tool path: readonly-operator-cli-client.js → openclaw-brain-context-builder.js (calls `formatCognitiveContextPack`)
- native-tool path: chat-completions-service.js (calls `formatCognitiveContextPack` on buildCanonicalBusinessContext result)

## The drop (root cause, verified)

The adapter collected the detail (geo/devices/categories/zones/zone_targeting/sites/variation)
into `provider_targeting` / `provider_targeting_detail` — but `formatItem()` serialized ONLY
`description` (truncated at 180 chars) + `preview_items` + facts/layers/history. The structured
`metadata.provider_targeting` NEVER reached the provider prompt → Cloé returned every detailed
field UNVERIFIED even though the adapter had the data.

Fix = extend `formatItem()` to render the structured detail. ONE fix covers BOTH paths because
they share the formatter. Do NOT patch chat-completions-service and openclaw-brain-context-builder
separately — the formatter is the chokepoint. When a "detail is collected but Cloé can't see it"
bug appears, check formatItem BEFORE the adapter.

## Semantics (OBSERVED_REFERENCE)

- Provider campaign values are REFERENCE AUTHORITY only — never the selected business values for a
  specific offer (crakrevenue:8780), never GROUNDED_FOR_NEW_CAMPAIGN, never promoted to business
  configuration automatically.
- Carry `authority_class: 'OBSERVED_REFERENCE'` + an explicit `authority_note` string in the detail
  object AND surface both in the rendered pack (provider reads the note, not just the value).
- ZONE_STATS stays a SEPARATE authority from PROVIDER_TARGETING; zones=[] with valid zone_targeting
  is network-targeted, never a blocker, never "targeting unavailable".

## Provider v2 readback quirks (verified live 2026-08-08)

- GET /campaigns returns `{ result: { [campaign_id]: campaign }, request_metadata: { limit, offset, count } }`
  — a MAP keyed by campaign id, not an array. 24 campaigns, AUTH_OK.
- GET /campaigns/:id exposes: advertiser_ad_type, advertiser_ad_type_label, pricing_model,
  pricing_model_name, price, status / calculated_status, variations_counts, max_daily_budget, capping…
- countries/geo, devices/device_types, categories, zones, zone_targeting, sites are GENUINELY
  ABSENT from the v2 detail payload for all sampled campaigns (8268574, 8268702, 8272276, 8308460,
  8293490 + full 24-campaign list scan). They must render UNVERIFIED live — never invent them. The
  adapter surfaces them only when the payload includes them (tests/fixtures or a future readback).
- Tracking readiness: variation_count / valid_variation_count derive from variations_counts;
  required_macros_present / missing_macros need variation URLs that the detail readback does NOT
  carry → UNVERIFIED unless the caller supplies `trackingMacros` via options.

## Live-certification proof technique

Inside the deployed container (node probe, no secrets printed):

1. `getExoClickProviderContext({ campaignId })` → assert authority_class + which fields are present.
2. `buildCanonicalBusinessContext({})` → find item kind `provider_targeting_context`.
3. `formatCognitiveContextPack({ ...available_items })` → extract the line containing
   `provider_targeting_context_v1` — the rendered ITEM_LINE is the PROOF the detail reaches the
   provider prompt. Look for `authority_class=OBSERVED_REFERENCE | ... | format=... | pricing=... |
   countries=UNVERIFIED ...`. Asserting on the metadata object alone is NOT sufficient — the point
   of the mission is what the provider actually receives.

## Test-harness pattern (deterministic, no provider)

- Injectable `getToken` (default `getExoClickAccessToken`) so tests never depend on global fetch for
  /login; injectable `fetchImpl` routing: /login, /business/runtime-context, /campaigns (list),
  /campaigns/:id (detail), /chat/completions (native-tool fallback).
- Capture ALL fetch methods and assert the set == ['GET'] (no-provider-write proof).
- After write_file with token-looking literals, verify on-disk integrity with search_files —
  `***` in the tool RESULT rendering is secret-masking, NOT file corruption (verified 2026-08-08).
- Mission test list to replicate: campaign authority detail; GEO visible; device/device_types;
  format; pricing/price; categories; zones=[] preserved; zone_targeting; tracking readiness;
  no secrets / no full signed click URLs; no provider writes; native-tool == non-tool detail.
