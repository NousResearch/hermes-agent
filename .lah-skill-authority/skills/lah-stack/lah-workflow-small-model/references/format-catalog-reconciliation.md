# Format Catalog / Runtime Reconciliation Pattern

Session: 2026-08-19 — Canary V2 failed at OpenClaw boundary because `format: "banner"` was catalog-selectable but missing from ExoClick normalizer `FORMAT_MAP`. This reference documents the reconciliation pattern and cross-format test matrix.

## Root Cause

The ExoClick normalizer (`exoclick-normalizer.js`) `FORMAT_MAP` only had `popunder` (advertiser_ad_type: 7). The catalog (`TEMPLATE_REGISTRY` in `exoclick-hybrid-builder.service.js`) and compiler (`campaign-creation-orchestrator.js`) both accepted `banner`, `native`, and `video` formats. This mismatch caused `NORMALIZATION_FAILED / BLOCKED_INVALID_FIELDS` at the OpenClaw boundary after operator approval.

## ExoClick Advertiser Ad Types (from `advertiser-ad-types.json`)

| Format | advertiser_ad_type | media_storage_template |
|--------|-------------------|----------------------|
| popunder | 7 | link |
| banner | 0 | img_banner |
| native | 22 | native_ad |
| video | 21 | link |

Reference: `openclaw-runtime/exoclick_refs/advertiser-ad-types.json`

## SELECTABLE_FORMAT_MUST_BE_RUNTIME_EXECUTABLE Invariant

Every catalog-selectable format must have a corresponding entry in `FORMAT_MAP`. If a catalog format has no runtime mapping, BLOCK at selection/compile time before reaching the operator approval gate.

**Validation function**: `validateFormatExecutable(selectableFormats, formatMap)` exported by `exoclick-normalizer.js`

## Cross-Format Test Matrix

| FORMAT | CATALOG_SUPPORTED | COMPILE_SUPPORTED | RUNTIME_NORMALIZES | RESULT |
|--------|-------------------|-------------------|--------------------|--------|
| popunder | YES | YES | YES | PASS |
| banner | YES | YES | YES | PASS (was BLOCKED, now fixed) |
| native | YES | YES | YES | PASS (was missing, now added) |
| video | YES | YES | YES | PASS (was missing, now added) |
| unknown_format | NO | NO | NO | BLOCK |
| img_banner | YES | YES | YES (alias→banner) | PASS |
| BANNER | YES | YES | YES | PASS (case-insensitive) |

## Repair Pattern

1. Identify catalog-selectable formats from `TEMPLATE_REGISTRY` and `campaign-creation-orchestrator.js` format extraction
2. Cross-reference against `FORMAT_MAP` in `exoclick-normalizer.js`
3. Any catalog format missing from `FORMAT_MAP` → add it using `advertiser-ad-types.json` for the correct `advertiser_ad_type` ID and `media_storage_template`
4. Run the cross-format test matrix: for each format, verify selection validation → compile → runtime normalization all pass
5. Add the `SELECTABLE_FORMAT_MUST_BE_RUNTIME_EXECUTABLE` invariant check

## Deployment Synchronization

The source code repo and the deployed runtime can diverge. After repairing FORMAT_MAP in the source, always verify the deployed runtime's copy matches:

```bash
# Check source
grep -A10 "FORMAT_MAP" /home/deploy/openclaw-runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js

# Check deployed runtime
sudo sed -n '50,65p' /opt/lah-goes/runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js
```

If they diverge, update the deployed runtime and restart:
```bash
sudo systemctl restart lah-governed-operator-executor.service
```

## Regression Tests

1. popunder known-valid → PASS
2. banner → PASS (canonically supported, now fixed)
3. every catalog-selectable format → runtime-normalizable → PASS
4. unknown format → BLOCK → PASS
5. semantic alias resolves deterministically → PASS
6. no provider mutation required → PASS