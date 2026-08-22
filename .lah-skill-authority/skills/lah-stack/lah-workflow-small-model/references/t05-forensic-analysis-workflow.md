# T05 Forensic Analysis Workflow

Session: HERMES_T05_LIVE_BUSINESS_FORENSIC_AND_STOP_GATE_V1 (2026-08-19)
Campaign: 8552896 (T05_Fresh_Three_Arm_Math_Observation)
Verdict: T05_TERMINATE

## Data Unit Normalization (CRITICAL)

The ExoClick API returns values in **different units** depending on the endpoint.

| Endpoint | Field | Unit | Normalization |
|---|---|---|---|
| `/campaigns/{id}?detailed=true` | `total_budget_spent`, `total_budget_limit`, `max_daily_budget`, `price` | **cents** | Divide by 100 for USD |
| `/statistics/a/global` | `cost`, `cpc`, `cpm` | **USD** | Already in dollars |
| `/statistics/a/global` | `impressions`, `clicks`, `conversions` | raw counts | No conversion |

**Pitfall**: Diagnostic reports have historically displayed raw cent values as dollar values (e.g., showing `$1,375.60` for 1375.60 cents = $13.756). Always verify the unit before reporting. The `cost` field in statistics responses is in USD; the `total_budget_spent` field in campaign readback is in cents.

**Verification**: At $0.50 CPM, 28,861 impressions should cost ~$14.43. If a campaign readback shows `total_budget_spent=1451.3`, that is 1451.3 cents = $14.513, which is consistent with the stats API cost of $14.43 (rounding difference).

## Popunder CPM Tracking Semantics

For Popunder CPM traffic, **ExoClick clicks = 0 is normal and expected**. Do NOT treat zero ExoClick clicks as evidence of broken tracking.

Zero CR clicks and zero conversions indicate a **business-level failure** (the offer does not convert this traffic), not a tracking failure. The tracking chain is intact if macros resolve correctly and the redirect gateway tests pass.

**Do not confuse traffic quality with tracking health.**

## 12-Section Forensic Structure

1. **Campaign Identity** — Lock to campaign_id, verify from provider + LAH records
2. **Live Authoritative Funnel** — impressions, CR clicks, events, conversions, revenue, spend (normalize cents→USD)
3. **Tracking Health** — Verify tracking chain, classify TRACKING_HEALTHY or TRACKING_DEGRADED
4. **Comparison vs Prior Campaigns** — Normalize by format, GEO, device, offer, spend
5. **Zone/Domain Forensics** — Decompose by zone_id × domain, rank by spend and signal
6. **Arm/Creative/Experience Analysis** — Decompose by offer/arm/creative
7. **Time-Series** — Launch, middle, latest windows
8. **Statistical Negative Evidence** — P(0 conversions | p, n) for CVR hypotheses
9. **Economic Value of Continuing** — Information value per dollar, already-resolved hypothesis check
10. **Action Gate** — KEEP / PRUNE / PAUSE / TERMINATE
11. **Mutation Rules** — Allowed vs forbidden mutations
12. **Final Verdict** — One of the T05_* verdict codes

## Key Findings for T05

- 28,861 impressions, 0 CR clicks, 0 conversions, $14.43 spent, $0 revenue
- Tracking chain: TRACKING_HEALTHY (zero CR clicks is business failure, not tracking)
- Failure is GLOBAL across all 276 zones and all 3 arms
- No positive signal in any segment
- Information value per dollar: ZERO — hypothesis already resolved
- Verdict: T05_TERMINATE