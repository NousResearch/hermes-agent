# ExoClick Provider Boundary — Device-ID Authority, Readback, Verification

Session-proven 2026-08-10 (campaign 8529830: mission "mobile" but provider UI showed
"Tablet" as the only Targeted Device). The root cause, the authoritative endpoints,
and the verification recipes are reusable for ANY ExoClick campaign work in the
LAH stack (openclaw-runtime + lah-brain).

## 1. TWO DISTINCT device ID spaces — never conflate them

- `GET /v2/collections/device-types` → **DEVICE TYPE ids**:
  1=Desktop, 2=Mobile, 3=Tablet, 4=Smart TV, 5=Wearable, 6=Console.
- `GET /v2/collections/devices?advertiser_ad_type=<id>` → **DEVICE ids**:
  0=Generic Desktop, 1=iPhone, 2=iPad (Tablet), 4=Kindle, 56=Generic Mobile,
  78=Generic Tablet, 79=Generic Smart TV, 80=Generic Console, ...
- `CreateCampaign` / `UpdateCampaign` `devices.elements` expects **DEVICE ids**.
  Proof: campaign 8529830 created with `elements:[2]` read back as
  `devices.targeted=[{id:2,name:"iPad",device_type:{id:3,name:"Tablet"}}]`.
  The old internal mapping (desktop=1, mobile=2, tablet=3) was the TYPE space —
  "mobile"→2 silently targeted iPad/Tablet.

Canonical name→DEVICE-ids mapping (live-verified 2026-08-10, popunder ad_type=7):

| name | device_type_id | device_ids |
|------|---------------|------------|
| desktop | 1 | [0] |
| mobile | 2 | [1,15,16,17,18,19,20,26,40,41,43,44,45,46,49,52,56,66,69,72,76,83,85,87,89,91,93,95,97,99,101,103,105,107,109] (35 ids) |
| tablet | 3 | [2,4,29,42,51,53,57,78,111] (9 ids) |

Canonical files:
- openclaw-runtime `lah-openclaw-mvp/data/refs/device_name_to_id.json` → per name:
  `{ device_type_id, device_ids[], name, short_name, authority }`.
- lah-brain `src/campaign-factory/exoclick-maps.js` → `DEVICE_TYPE_ID` +
  `DEVICE_IDS_PER_TYPE` (and `STATUS_ID`: 1 Running, 0 Paused, -1 Archived per swagger).

## 2. Campaign readback: `?detailed=true` is MANDATORY

- `GET /campaigns/{id}` WITHOUT `detailed` → omits ALL targeting collections.
  A "readback exact" that skips it CANNOT verify targeting — this is exactly how
  the 8529830 iPad/Tablet target passed the creation readback gate.
- `GET /campaigns/{id}?detailed=true` → response shape:
  `{ result: { campaign: {...}, devices: {targeted:[Device], blocked:[]},
  device_types: {...}, countries: {...}, categories: {...}, zones, variations_counts, ... } }`.
- Targeting collections sit at the **result level** (`result.devices`), NOT on
  `campaign.devices` — code reading `campaign.devices` gets null.
- Verify semantics from the raw `Device` objects
  (`{id, name, device_type:{id,name}, device_brand}`) — never from internal labels.

Swagger spec: `https://api.exoclick.com/v2/docs/swagger.json` (minified single line —
parse with a JSON parser, not grep). Docs corpus: `https://docs.exoclick.com/llms-full.txt`.

## 3. Traffic verification (read-only, zero spend)

`GET /v2/statistics/a/traffic?advertiser_ad_type=7&countries=USA&categories=2&devices=<comma-separated device ids>`
→ `result.impressions` = reachable daily traffic. Used to prove materiality:
devices=[2] (iPad) ≈ 336K vs all 35 mobile ids ≈ 20.07M (US/pop/Adult).
Note: device id 0 is rejected by this endpoint (400 "must be >= 1") although it is
valid in campaign targeting (desktop Generic).

## 4. Campaign device correction (surgical, stays PAUSED)

`PUT /campaigns/{id}/targeted/devices` with body = plain array of DEVICE ids
(swagger `body_collection_integers_array`). Replaces the targeted set only.
Keep the campaign PAUSED before/after; independent readback after the PUT must show
every `devices.targeted[].device_type` = the intended type.

## 5. Pitfalls

- Never trust internal labels for readback semantics — resolve raw ids against
  `/collections/device-types` + `/collections/devices`.
- In openclaw `exoclick-provider-context.js`, read targeting from the detailRoot
  (`body.result`) with a legacy fallback to `campaign.device_types`; expose
  `devices_detail` (raw provider objects) so a future audit can re-verify.
- Mono-DEVICE-TYPE contract: "mobile" expands to 35 ids — a payload validator must
  restrict TYPES, not count of ids.
- The provider detail endpoint omits device info without `detailed=true`; watchdog /
  monitor scripts that only read status+spend are fine, but targeting verification
  must always use the detailed variant.
