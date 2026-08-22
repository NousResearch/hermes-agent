# LAH Brain tracking identity + bounded read-only behavioral summary pattern

Established during CLOE_TRACKING_IDENTITY_AND_READONLY_BEHAVIORAL_SUMMARY_WIRING_V1
(mission INTERRUPTED mid-implementation — code written in worktrees, not merged;
verify current state before reuse).

## 1. Canonical tracking data path (root-cause trace template)

ExoClick URL macros → landing URL → beacon → ingestion → persisted event → readiness

- **URL builder** (openclaw-runtime): `lah-openclaw-mvp/src/services/agents/implementations/agent-tracking-url-builder.js`
  — `REQUIRED_MACROS`: `{clickid}`, `{conversions_tracking}` (ct), `{zone_id}`, `{site_id}`,
  `{campaign_id}`, `{format}`, `{src_hostname}`, `{asset_id}`. Also
  `src/services/runtime-l1-operator-pack.js` `runTrackingUrl()` passes the literal macro set.
- **Redirect gateway** (lah-brain): `src/redirect-gateway.js` — preserves `{clickid}`→aff_sub,
  `{zoneid}`→aff_sub2, `{campaignid}`→aff_sub3 (network-side click identity survives).
- **Beacon** (lah-brain): `public/lah-beacon.js` — reads `click_id`/`clickid`/`exo_click_id`/
  `affiliate_click_id`, `ct`/`conversions_tracking`, `campaign_id`, `zone_id`, `site_id` from URL
  params; POSTs to `/events/public` (CORS: liveaccesshub.com only).
- **Ingestion**: `src/server.js` `/events/public` → `src/db.js` `insertEvents` →
  `src/event-schema.js` `normalizeEvent` (`cleanDimension` fallbacks).
- **Readiness**: `routes/api.js` `GET /admin/tracking/ingestion-readiness` (requireAdminAuth,
  `x-admin-api-key`; session auth also accepted via `isAuthenticated` first).

## 2. Placeholder origin (verified live 2026-08-09)

- `event-schema.js` `cleanDimension()` defaults MISSING identifiers to non-empty placeholders:
  `unknown-campaign`, `unknown-zone`, `unknown-site`, `unknown-domain`, `unknown-…`.
- `exoclick-importer.js` CSV rows default to the same `unknown-*` values when columns absent.
- **Conclusion: non-empty string ≠ usable identity.** Propagation itself was NOT broken — the URL
  builder injects all macros, the redirect gateway preserves them, the beacon reads them. The
  "defect" is (a) parser fallback for genuinely absent values and (b) readiness counting any
  non-empty string as valid. Do not hunt for a propagation bug when the trace shows placeholders
  come from fallbacks — verify each link, then fix the counting semantics.

## 3. Semantic-valid identity pattern

- Existing canonical detector: `src/segment-filter.js` `isUnattributedId()` (UNATTRIBUTED_SENTINELS
  `{"", "unknown", "n/a", "(unset)", "__unknown__"}` + any value starting with `unknown-`).
- **SQL mirror** (no JS row loop): `LOWER(TRIM(col)) IS NOT NULL AND != '' AND NOT IN
  ('unknown','n/a','(unset)','__unknown__') AND NOT LIKE 'unknown-%'` — callable via a helper
  `sqlSemanticValidExpr(column)`.
- **Counter split** (backward-compatible readiness):
  - raw presence `events_with_X_id` (non-empty string) — keep for compat
  - semantic-valid `events_with_valid_X_id` (placeholder excluded)
  - `identity_complete_rows` = ALL required semantic IDs valid (click + campaign + zone + external token)
- **Single source of truth**: put the counters in `src/tracking-readiness.js`
  `computeTrackingReadiness(db)` so the readiness route AND any analytics surface (behavioral
  summary) share identical semantics.

## 4. Bounded read-only behavioral summary surface (LOT B pattern)

- **Reuse** `reportCache.getReport()` — export `reportCache` from `src/server.js` (same pattern as
  `db` export) — NO new analytics engine, NO duplicated metric formulas.
- **Bounds** (mission contract): ≤12 KB serialized JSON; ≤5 entries per ranked dimension; 8 exposed
  dimensions (zone, site, geo, device, format, LP, CTA, creative); ≤5 opportunities, ≤5 mismatch
  segments, ≤3 verdicts; no raw events, no full URLs, no secrets.
- **Provenance explicit**: `source: "lah_brain_behavioral_analytics"` — a DISTINCT authority from
  ExoClick provider stats, CrakRevenue marketplace snapshot, campaign memory, Business Health.
- **Fail-closed**: unavailable/oversize → 503 `{ ok:false, read_only:true, provider_write:false,
  db_write:false }`; success → `read_only:true, provider_write:false, db_write:false, bytes`.
- **Auth**: `requireAdminAuth` + `x-admin-api-key` — the existing machine-to-machine pattern;
  never share dashboard session cookies/passwords.
- This is the 3rd instance of the read-only bridge family (predecessors:
  `business-runtime-bridge.js`, `marketplace-dataset-reader.js` in openclaw-runtime) — for the
  OpenClaw consumer, wire into `cloe-canonical-business-context.js` available_items with
  kind/name/source/read_only/safety_flags/metadata and `available:false` on failure.

## 5. Live baseline evidence (2026-08-09, authorized GET-only)

- `https://leanframeworklab.com/health` = 200, `service: "lah-brain-v6.2"`,
  `provider_evidence_guard_commit: 28aa9723…`.
- readiness: `total_events:41`, raw campaign/zone 41/41, external_click_token 11, identity_complete
  11; the 5 latest samples were ALL `unknown-campaign`/`unknown-zone`/`unknown-domain` → raw
  counters overstated usable identity (41/41 vs ~11 semantic).
- **LAH Brain deploys REMOTELY** (leanframeworklab.com → Hostinger IPv6 range); no deploy script
  exists in the lah-brain repo (runbook `docs/ops/TRACKING_INGESTION_ADMIN_AUTH_RUNBOOK.md`
  documents verification only) → the deploy path is operator-managed; do NOT guess it — surface as
  a blocker/operator question when the mission reaches deploy.

## 6. Dual-repo fast-bounded setup

- Canonical checkouts are often parked on feature branches and dirty. For a 2-repo mission use
  `git worktree add /home/deploy/lah-stack-worktrees/<mission>-<repo> origin/main` for BOTH repos
  (from the canonical checkout) — clean isolated workspaces at the exact main SHA.
- Fresh worktrees have NO `node_modules` → run `npm install` before executing tests; a pre-existing
  test file failing with `MODULE_NOT_FOUND express` in a fresh worktree is the install gap, not a
  regression.
