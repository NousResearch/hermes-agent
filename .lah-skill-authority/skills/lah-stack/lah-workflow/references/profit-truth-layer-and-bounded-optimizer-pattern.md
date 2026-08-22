# Profit Truth Layer + Bounded Auto-Optimizer — pattern and pitfalls

Established during CLOE_PROFIT_TRUTH_AND_BOUNDED_AUTO_OPTIMIZER_V1 (2026-08-09,
openclaw-runtime). Two-PR structure: PR A = canonical economic truth layer
(read-only), PR B = deterministic bounded optimizer (shadow-only) consuming
ONLY the truth layer.

## 1. Authority model (audit findings that shape the design)

- Cost authority (spend/clicks/conversions/impressions per zone) EXISTS:
  `exoclick-stats.js` (normalizeStatsRows, fetchAdvertiserZoneStats,
  zero-vs-missing, 93-day cap) + `zone-statistics-contract.js` (23 fields).
- Affiliate outcome authority: NO zone-level conversion+payout source in-repo.
  Only campaign-level revenue enters via LAH Brain (behavioral-summary client
  global_funnel; business-runtime-bridge money.summary). No CPA anywhere.
- Creative outcome store is caller-fed, defaults numerics to 0 (existing
  semantics — keep, do not change).
- Memory vocabulary ready: `campaign-memory-zone-events.js` has 13 event types
  (zone_shadow_decision, zone_monitor_cycle, zone_cut_*).
- Conclusion: zone-level revenue attribution is BLOCKED_REVENUE_AUTHORITY in
  V1. Implement the canonical PARTIAL contract; do NOT fake completeness;
  optimizer gates revenue-dependent decisions on BLOCKED_MISSING_EVIDENCE.

## 2. Profit Truth contract design (the reusable shape)

- Pure module: identity / observed / derived / provenance / completeness /
  verdict. Deterministic arithmetic — ONE authority (no two implementations
  computing different ROI).
- Derived semantics (undefined = null, never 0):
  - profit = revenue - spend
  - roi = spend > 0 ? ((revenue - spend) / spend) * 100 : null
  - epc = clicks > 0 ? revenue / clicks : null
  - cpa = conversions > 0 ? spend / conversions : null
- Zero-vs-missing strict: literal 0 preserved, absence → null
  (toNumberOrNull). Never coerce.
- Fail-closed: CURRENCY_MISMATCH and DATE_WINDOW_MISMATCH reject, never
  silently normalize.
- Verdict states: AVAILABLE / PARTIAL / INSUFFICIENT_IDENTITY /
  MISSING_COST_AUTHORITY / MISSING_REVENUE_AUTHORITY / UNATTRIBUTABLE_REVENUE /
  EMPTY.
- Deterministic aggregation: sums observed, recomputes derived from sums.

## 3. Attribution/join pitfalls (each cost a repair cycle)

1. **ExoClick-side revenue ≠ affiliate payout.** A zone row carrying
   `revenue`/`payout`/`earning` must NOT be treated as affiliate revenue.
   When revenue is unattributable, NEUTRALIZE observed.revenue → null and
   derived profit/roi/epc/cpa → null; keep the raw value as a provenance
   trace (`cost_authority_revenue_observed`). Leaving the value in the
   observation made a behavioral scenario fail with a fabricated profit claim.
2. **Keep cost observations when the revenue authority is absent.** The
   compose step must push the cost observation (verdict
   MISSING_REVENUE_AUTHORITY / PARTIAL) instead of dropping the zone into
   `blocked` when the join returns MISSING_REVENUE_AUTHORITY. Dropping it
   turned a PARTIAL pack into EMPTY.
3. **Pack-level verdict:** any row that is not fully AVAILABLE
   (UNATTRIBUTABLE_REVENUE, MISSING_REVENUE_AUTHORITY, PARTIAL) makes the pack
   PARTIAL — never EMPTY. Counting only AVAILABLE/PARTIAL rows missed
   UNATTRIBUTABLE and returned EMPTY for a populated pack.
4. **revenue_attributable defaults true at cost-row build.** Attributability
   is decided at JOIN time (campaign revenue cannot project onto zones). A
   standalone cost row without revenue is MISSING_REVENUE_AUTHORITY, NOT
   UNATTRIBUTABLE. Forcing revenue_attributable:false in the cost builder
   made every no-revenue row UNATTRIBUTABLE.

## 4. Bounded optimizer (deterministic, shadow-only)

- Consumes ONLY the Profit Truth layer output (never queries providers
  itself). One truth layer → many consumers.
- Decision states: OBSERVE / HOLD / CUT_CANDIDATE / PROMOTE_CANDIDATE /
  SCALE_CANDIDATE / INSUFFICIENT_DATA / BLOCKED_MISSING_EVIDENCE /
  BLOCKED_POLICY / BLOCKED_STALE_DATA.
- Policy: versioned OPTIMIZER-POLICY-v1, env-driven, deep-frozen, labeled
  TEST/SHADOW defaults (not proven business truth). OPTIMIZER_MODE accepts
  ONLY 'shadow' — 'auto'/unknown fails closed to shadow. No env default
  enables autonomous execution.
- Loser safety: spend < min_spend_before_cut + zero conversions → OBSERVE
  (SPARSE_SAMPLE), never CUT.
- Winner safety: conversions < min_conversions_before_promote → HOLD
  (TINY_SAMPLE), never SCALE. BLOCKED_MISSING_EVIDENCE when revenue not
  attributable (NO_FAKE_PROFIT_CLAIM).
- decision_id = sha256(identity | window | policy_version | **effective
  thresholds** | metrics). Pitfall: hashing only policy_version makes two
  env-different policies collide (same version string). Include
  `JSON.stringify(Object.entries(thresholds).sort())`.
- Shadow cycle returns a permanent safety envelope:
  provider_mutation_attempted:false, live_sent:false, writes_performed:false,
  traffic_activation:false, campaign_mutation:false.

## 5. Test/harness pitfalls

1. **`decision.extra` spread gotcha:** `{...extra}` spreads extra's KEYS into
   the decision object; it does NOT create an `extra` property. Tests reading
   `decision.extra.shadow_only` get undefined. Fix: `extra: Object.freeze(extra)`
   BEFORE `...extra` in the returned object.
2. **Memory event sanitization masks long tokens:** writeZoneMemoryEvent
   sanitizes tokens ≥ 20 chars in write mode — including schema strings like
   `cloe_zone_monitoring_memory_v1` → '***'. Assert on the returned
   `event.record` (non-sanitized), not on the persisted content.
3. **ExoClick + LAH Brain composition mock:** patch global.fetch + set
   EXOCLICK_API_TOKEN, LAHB_URL, LAHB_ADMIN_API_KEY +
   clearExoClickAccessTokenCache (pattern from
   exoclick-stats-auth-flow.test.js). The mock must handle POST {BASE}/login
   (returns token), GET {BASE}/statistics/a/zone, GET
   {LAHB}/admin/analytics/behavioral-summary and GET
   {LAHB}/business/runtime-context.
4. **fetchImpl must be threaded through the whole chain.** When composing
   existing authority clients, add injectable fetchImpl (default
   globalThis.fetch) at EVERY layer — otherwise the mock is silently ignored
   and tests hit the real network. Rétro-compatible default keeps callers
   stable.

## 6. Behavioral simulation receipt

The receipt writer's REPO_ROOT must be the git worktree root. Scripts nested
in lah-openclaw-mvp/tools/operator-validation/ are THREE levels above the repo
root; two levels points at lah-openclaw-mvp/ → empty diff → diff_hash
e3b0c442... (sha256 of empty string). Use
`git rev-parse --show-toplevel` programmatically; never count `..` levels.
See behavioral-operator-simulation skill pitfall 15.
