# FORMAT_MAP Deployment Drift Pattern

Session: 2026-08-20
Source: LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3_RESUME_WITH_CERTIFIED_LING

## Problem

The source code repo (`/home/deploy/openclaw-runtime/`) and the deployed runtime (`/opt/lah-goes/runtime/lah-openclaw-mvp/`) can diverge in their `FORMAT_MAP` entries. The source code may already contain a FORMAT_MAP fix while the deployed runtime still runs a stale version. This causes the canary to fail at execution even when the source code is correct.

## Concrete Instance (2026-08-20)

**Source code** (`lah-openclaw-mvp/src/services/exoclick-normalizer.js`):
```js
const FORMAT_MAP = {
  popunder: { advertiser_ad_type: 7, media_storage_template: 'link' },
  banner:   { advertiser_ad_type: 0, media_storage_template: 'img_banner' },
  native:   { advertiser_ad_type: 22, media_storage_template: 'native_ad' },
  video:    { advertiser_ad_type: 21, media_storage_template: 'link' }
};
```

**Deployed runtime** (`/opt/lah-goes/runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js`):
```js
const FORMAT_MAP = {
  popunder: { advertiser_ad_type: 7, media_storage_template: 'link' },
  banner:   { advertiser_ad_type: 0, media_storage_template: 'img_banner' }
};
// native and video are MISSING
```

**Diff**: Source has `native` and `video` entries; deployed runtime does not.

## Detection

Compare FORMAT_MAP keys between source and deployed runtime:
```bash
diff \
  <(sed -n '51,70p' /home/deploy/openclaw-runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js) \
  <(sudo cat /opt/lah-goes/runtime/lah-openclaw-mvp/src/services/exoclick-normalizer.js | sed -n '51,70p')
```

## Repair

1. Update the deployed runtime's `exoclick-normalizer.js` to match the source
2. Restart the governed operator executor:
   ```
   sudo systemctl restart lah-governed-operator-executor.service
   ```
3. Verify the runtime's FORMAT_MAP includes all catalog-selectable formats
4. Then resubmit the canary approval

## Impact

If a canary tries to use `native` or `video` formats with the stale deployed runtime, the normalizer rejects them with `NORMALIZATION_FAILED / BLOCKED_INVALID_FIELDS` at the OpenClaw boundary — after operator approval has already been granted.

## Related

- `references/format-catalog-reconciliation.md` — full reconciliation pattern and cross-format test matrix
- `references/canary-execution-flow.md` — canary execution flow including anti-archaeology constraints