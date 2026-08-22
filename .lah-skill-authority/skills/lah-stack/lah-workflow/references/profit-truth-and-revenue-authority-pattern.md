# Profit Truth Layer + Cross-Repo Revenue Authority Reconciliation

Established during CLOE_PROFIT_TRUTH_AND_BOUNDED_AUTO_OPTIMIZER_V1 (2026-08-09,
openclaw-runtime PR #745 + #746). Two reusable halves:

1. The Profit Truth Layer + Bounded Optimizer architecture (canonical economic
   truth, one arithmetic authority, shadow-only decisions).
2. The cross-repo revenue-authority reconciliation that MUST precede any final
   verdict claiming "revenue authority missing" / BLOCKED_REVENUE_AUTHORITY.

---

## Part 1 — Profit Truth Layer + Bounded Optimizer (implementation shape)

### Non-negotiables (mission contract, reusable verbatim)
- ONE canonical economic truth authority. Never two modules computing
  materially different ROI/profit semantics.
- observed zero != missing: literal `0` is a real observed zero; absence is
  `null`. Never coerce.
- A mathematically undefined derived metric is `null`, never `0`.
- NEVER invent missing conversions / payout / spend / revenue / ROI / EPC /
  CPA / site / zone / creative / offer / attribution.
- ExoClick provider row carrying `revenue`/`payout`/`earning` is NOT affiliate
  payout unless its semantics are proven.
- CrakRevenue advertised payout (public offer) is informational, never
  observed earned revenue.
- Do NOT build a second tracker / campaign-memory authority / creative engine.
  Adapters/composition over replacement.

### Module shape (3 files per layer, pure core + adapter + read-only surface)
```
src/services/profit-truth/
  profit-truth-contract.js   # pure: buildProfitTruthObservation, deriveEconomicMetrics,
                             #   deriveObservationVerdict, aggregateProfitTruth, dedupeProfitTruth
  profit-truth-join.js       # provenance-first join: buildCostObservationFromZoneRow,
                             #   normalizeAffiliateRevenueEvidence, joinZoneCostWithAffiliateRevenue,
                             #   composeProfitTruth
  profit-truth-context.js    # read-only surface: buildProfitTruthPack, answerProfitTruthQuestion,
                             #   buildProfitTruthAvailableItem, collectTrafficCostAuthority,
                             #   collectAffiliateRevenueAuthority
src/services/bounded-optimizer/
  optimizer-policy.js        # buildOptimizerPolicy -> OPTIMIZER-POLICY-v1, frozen, env-driven
  optimizer-engine.js        # evaluateOptimizerDecision (9 states), deriveOptimizerDecisionId
  optimizer-shadow.js        # runShadowOptimizerCycle (zero mutation, zone_shadow_decision events)
```

### Deterministic metric formulas (shared with upstream roi-engine)
```
profit = revenue - spend
roi    = spend  > 0 ? ((revenue - spend) / spend) * 100 : null
epc    = clicks > 0 ? revenue / clicks                  : null
cpa    = conversions > 0 ? spend / conversions          : null
```

### Attribution / join states
AVAILABLE | PARTIAL | INSUFFICIENT_IDENTITY | MISSING_COST_AUTHORITY |
MISSING_REVENUE_AUTHORITY | UNATTRIBUTABLE_REVENUE | EMPTY

Join rules (fail closed):
- revenue evidence grain `zone` + zone_id matches → attach (zone_direct)
- revenue evidence grain `campaign` + observation has NO zone → attach at
  campaign grain
- revenue evidence grain `campaign` + observation HAS zone →
  UNATTRIBUTABLE_REVENUE — never project campaign totals onto zones
- revenue authority absent → KEEP the cost observation with honest verdict
  (MISSING_REVENUE_AUTHORITY), do NOT drop the zone (this was a real bug:
  composeProfitTruth initially blocked the whole row into `blocked[]` instead
  of preserving the cost observation)
- ExoClick-side revenue value is neutralized (kept in provenance as
  `cost_authority_revenue_observed`) when it cannot be proven affiliate payout

### Optimizer decision states
OBSERVE | HOLD | CUT_CANDIDATE | PROMOTE_CANDIDATE | SCALE_CANDIDATE |
INSUFFICIENT_DATA | BLOCKED_MISSING_EVIDENCE | BLOCKED_POLICY |
BLOCKED_STALE_DATA

Safety semantics:
- LOSER SAFETY: zero conversions + trivial spend (< min_spend_before_cut) →
  OBSERVE, never CUT (sparse-sample protection)
- WINNER SAFETY: tiny sample (conversions < min_conversions_before_promote) →
  HOLD, never SCALE; SCALE requires stronger evidence than PROMOTE
- revenue not attributable → BLOCKED_MISSING_EVIDENCE with reason
  NO_FAKE_PROFIT_CLAIM (never a winner claim from unproven revenue)
- protected campaigns → BLOCKED_POLICY
- idempotency: decision_id = sha256(identity | window | policy_version |
  EFFECTIVE THRESHOLDS | metrics). Include the effective thresholds — same
  policy version with different env thresholds must produce different ids
  (real bug: policy_version alone collided)
- shadow default fail-closed: OPTIMIZER_MODE accepts only 'shadow'; 'auto' /
  unset / unknown collapses to shadow. No env default enables autonomous
  execution.
- `decision.extra` must be stored as a real property (`extra: Object.freeze(extra), ...extra`)
  or consumers reading `decision.extra.shadow_only` get undefined (real bug:
  spread-only created the keys but no `extra` property).

### Read-only exposure (no dashboard)
- canonical business context item pattern: `kind: 'profit_truth'` /
  `kind: 'bounded_optimizer_shadow'`, `read_only: true`, safety_flags
  `['read_only','no_write','no_execute','no_provider_call','no_traffic','no_spend']`,
  injected ONLY on explicit query (ordinary prompts never trigger a fetch).
- `answerProfitTruthQuestion({pack, question})` → deterministic evidence-only
  answers ("Which zones are profitable?", "Which zones spent without
  conversions?", "EPC overview", "partial due to revenue attribution").
- ExoClick stats `fetchImpl` injectable (rétro-compatible) so profit-truth can
  be tested without the global fetch patch.

### Test / verification bars that passed
- 25 contract+join tests (mission 1F cases 1-20), 24+1 optimizer tests (2K),
  6+8 behavioral simulation scenarios, receipts VALID via
  behavioral-operator-simulation validator.
- Regressions: exoclick-stats-auth-flow, zone-statistics-contract,
  canonical-tracking-context, creative-outcome-store/winner/ranking,
  lahb-behavioral-summary-client, cloe-canonical-business-context.

---

## Part 2 — Cross-Repo Revenue Authority Reconciliation (MANDATORY before verdict)

### Trigger
Any mission whose final verdict would claim "no X authority exists" or return
BLOCKED_REVENUE_AUTHORITY based on an in-repo audit. The in-repo surface may
lack the authority while the authority exists UPSTREAM with a missing WIRING.

### Bounded read-only method (verified working)
1. **Check the upstream push PR**: `gh pr view 1 --repo leanframeworklab/lah-core`
   → verify MERGED, note payload fields (sale event: event_type, click_id,
   affiliate_click_id, external_click_token, transaction_id, offer_id, goal_id,
   payout, revenue=payout, currency, network, reconciliation_state).
2. **Event schema persists the fields**: `src/event-schema.js` — transaction_id,
   offer_id, goal_id, currency, reconciliation_state, payout, revenue,
   click_id/affiliate_click_id/external_click_token, network, import_source.
3. **Ledger + engines in upstream MAIN**: `git merge-base --is-ancestor <sha> main`
   for src/money/reconciliation.js, src/money/roi-engine.js,
   src/money/conversion-importer.js, routes/money.js. by_zone rows:
   campaign_id, zone_id, impressions, clicks, spend, conversions (matched),
   revenue (matched_revenue), profit, roi, epc, cpa, ctr.
4. **Probe the LIVE endpoint** (deployed ≠ branch): `fetch(LAHB_URL +
   '/business/runtime-context', {headers:{'x-admin-api-key':...}})` → 200 with
   zones.rows structure even when the route file lives on a non-merged branch
   (feat/cloe-affiliate-e2e-lot1-business-runtime). Check zones.status:
   EMPTY is a data-window state, not proof the structure is absent.
5. **Trace identifier loss per boundary**: sale payload (present) → events
   (present) → affiliate_conversions ledger (ABSENT — schema has only `offer`,
   no transaction_id/offer_id columns) → money report → business-runtime-context
   → Profit Truth.

### Classification (use in final report)
- authority exists upstream: YES (lah-core PR #1 merged, events table persists)
- financial-ledger ingestion wiring: MISSING —
  `LAH_CORE_POSTBACK_TO_LAHB_FINANCIAL_LEDGER_WIRING_MISSING` (events `sale` are
  NOT auto-projected into affiliate_conversions; only POST /imports/conversions
  writes the ledger)
- runtime projection grain: by_zone rows carry campaign_id+zone_id (sufficient)
- Profit Truth consumption wiring: money.summary only (campaign grain);
  zones.rows consumption is the smallest follow-up — do NOT build another
  postback receiver

### Verdict discipline
- Do NOT use BLOCKED_REVENUE_AUTHORITY merely because Profit Truth currently
  sees campaign-level revenue. Distinguish the four layers above.
- A bounded reconciliation report + the four-layer taxonomy satisfies the
  operator's authority correction and keeps the optimizer semantics unchanged
  (shadow, fail-closed revenue gating, zero provider mutation).

### Minimal follow-up seams (future mission)
1. Merge the /business/runtime-context branch into lah-brain main.
2. Idempotent projection events `sale` → affiliate_conversions (dedup_key
   click_id|goal|payout / token|goal|payout, INSERT OR IGNORE) + add
   transaction_id/offer_id columns to the ledger.
3. Profit Truth collectAffiliateRevenueAuthority reads zones.rows at
   campaign_id+zone_id grain via the existing business-runtime-bridge.
