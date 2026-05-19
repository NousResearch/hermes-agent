# Idea 1 — RFQ Verification Board v0

**INTERNAL RESEARCH DRAFT ONLY · NOT FOR EXTERNAL DISTRIBUTION · NO RFQ SUBMISSION · NO EXECUTION CAPACITY**

**Purpose:** Internal board for converting screen-only BTC treasury/miner/cross-venue structures into quote-verified evidence.

**Status:** Internal planning only. Do not send client-specific RFQs, solicit transactions, introduce orders, or accept transaction compensation until counsel approves the legal wrapper/partner model.

**Confidence ladder:** hypothesis → screen-only → quote-verified → trade-verified.

---

## 1. Board rules

1. Every row starts as **screen-only** unless a real quote is attached.
2. Every quote must include timestamp, good-until, quote type, counterparty, size, margin/collateral, fees, and legal capacity.
3. Two independent quotes are preferred before using a structure as investor evidence.
4. No return claim is allowed from screen-only data.
5. CME rows require licensed/vendor/broker source before production use.
6. Deribit rows require jurisdiction/legal review before client-use assumption.
7. ETF rows require OPRA/vendor/broker data before production use; Nasdaq public JSON is research-only.

---

## 2. Quote status legend

- **Not prepared:** structure not specified enough for RFQ.
- **Template ready:** exact RFQ terms drafted; no outreach.
- **Legal pending:** ready technically, but counsel/partner approval required before sending.
- **Sent:** RFQ sent under approved legal capacity.
- **Indicative quote received:** non-firm market color.
- **Firm quote received:** executable or time-limited firm quote.
- **Trade executed:** executed/confirmed trade.
- **Rejected:** quote did not survive all-in economics or legal/operational filter.

Current board starts at **Template ready / Legal pending**.

---

## 3. First 10 RFQ board

| ID | Product | Client use case | Notional | Tenor | Structure | Target venues | Status | Evidence target |
|---|---|---|---:|---:|---|---|---|---|
| RFQ-001 | Treasury covered call | Income sleeve | 500 BTC | 45D | Sell 15% OTM call | ETF options, OTC, Deribit reference | Template ready / Legal pending | premium/yield quote |
| RFQ-002 | Treasury covered call | Upside-preserving income | 500 BTC | 45D | Sell 25% OTM call | ETF options, OTC, Deribit reference | Template ready / Legal pending | premium vs cap tradeoff |
| RFQ-003 | Treasury covered call | Quarterly income | 1,000 BTC | 135D | Sell 20% OTM call | ETF options, OTC, CME/Deribit reference | Template ready / Legal pending | longer-tenor premium |
| RFQ-004 | Treasury put-spread collar | Board downside policy | 500 BTC | 135D | Buy 90P / sell 80P / sell 115C / buy 130C | OTC, ETF options, Deribit reference | Template ready / Legal pending | net cost + floor/cap |
| RFQ-005 | Treasury put-spread collar | Larger treasury sleeve | 1,000 BTC | 135D | Buy 90P / sell 80P / sell 115C / buy 130C | OTC, ETF options, Deribit reference | Template ready / Legal pending | capacity/size discount |
| RFQ-006 | Miner production collar | Runway hedge | 225 BTC | 90D | Buy 85P / sell 120C | OTC, CME, Deribit reference | Template ready / Legal pending | runway protection cost |
| RFQ-007 | Miner disaster put | Severe downside hedge | 225 BTC | 90D | Buy 80–85P | OTC, CME, Deribit reference | Template ready / Legal pending | premium vs crash payoff |
| RFQ-008 | Miner put-spread collar | Quarterly production hedge | 225 BTC | 135D | Buy 90P / sell 75P / sell 115C / buy 130C | OTC, CME, Deribit reference | Template ready / Legal pending | low-cost protected band |
| RFQ-009 | Cross-venue straddle | RV / hedge recycling | BTC-equivalent TBD | 30D | IBIT ATM straddle vs Deribit ATM straddle | ETF options, Deribit | Template ready / Legal pending | net IV spread after costs |
| RFQ-010 | Cross-venue risk reversal | Skew comparison | BTC-equivalent TBD | 30D | IBIT 25D RR vs Deribit 25D RR | ETF options, Deribit | Template ready / Legal pending | skew spread after costs |

---

## 4. Detailed RFQ cards

## RFQ-001 — 500 BTC, 45D, 15% OTM treasury covered call

**Objective:** Verify premium available for a board-approved income sleeve with moderate upside cap.

**Client archetype:** BTC treasury company.

**Terms:**
- Exposure: 500 BTC-equivalent.
- Tenor: approximately 45 days.
- Strike: 115% of spot.
- Structure: sell call, fully covered by BTC/ETF exposure.
- Settlement: request alternatives — cash-settled BTC/USD, ETF-share-settled, or OTC cash-settled.
- Quote type: indicative first; firm quote later.

**Fields required:**
- premium bid/offer;
- IV;
- delta/vega;
- size available;
- collateral/margin;
- fees;
- assignment/exercise mechanics;
- quote good-until;
- documentation/onboarding.

**Decision rule:**
Advance if annualized premium remains attractive after all costs and cap is acceptable under policy.

---

## RFQ-002 — 500 BTC, 45D, 25% OTM treasury covered call

**Objective:** Verify upside-preserving income alternative.

**Client archetype:** BTC treasury company with high upside sensitivity.

**Terms:**
- Exposure: 500 BTC-equivalent.
- Tenor: approximately 45 days.
- Strike: 125% of spot.
- Structure: sell covered call.

**Fields required:** same as RFQ-001.

**Decision rule:**
Advance if premium is still meaningful while upside cap is board-friendly.

---

## RFQ-003 — 1,000 BTC, 135D, 20% OTM treasury covered call

**Objective:** Verify longer-tenor premium and capacity for larger BTC treasury sleeve.

**Client archetype:** BTC treasury company with quarterly/semiannual liquidity objective.

**Terms:**
- Exposure: 1,000 BTC-equivalent.
- Tenor: approximately 135 days.
- Strike: 120% of spot.
- Structure: sell covered call.

**Fields required:**
- package price for full size;
- partial-fill/layered execution price;
- collateral/margin;
- OTC vs listed economics;
- capacity discount vs 500 BTC;
- quote good-until.

**Decision rule:**
Advance if size does not degrade economics enough to make smaller staggered clips clearly superior.

---

## RFQ-004 — 500 BTC, 135D treasury put-spread collar

**Objective:** Verify board-friendly downside band with bounded upside give-up.

**Client archetype:** BTC treasury company needing downside floor but reluctant to fully cap upside.

**Terms:**
- Exposure: 500 BTC-equivalent.
- Tenor: approximately 135 days.
- Buy put: 90% of spot.
- Sell put: 80% of spot.
- Sell call: 115% of spot.
- Buy call: 130% of spot.
- Target: low debit, zero-cost, or modest credit.

**Fields required:**
- net package premium;
- leg-by-leg prices;
- package Greeks;
- max protected band;
- max upside band sold;
- collateral/margin;
- execution as package vs legs;
- quote good-until.

**Decision rule:**
Advance if the structure creates meaningful protection with acceptable upside give-up and collateral burden.

---

## RFQ-005 — 1,000 BTC, 135D treasury put-spread collar

**Objective:** Verify whether larger notional can be priced without excessive slippage.

**Client archetype:** larger BTC treasury company.

**Terms:** same as RFQ-004 but 1,000 BTC-equivalent.

**Additional required fields:**
- size availability;
- market impact estimate;
- staged execution alternative;
- block/RFQ minimums;
- OTC vs listed package comparison.

**Decision rule:**
Advance if size economics remain close to 500 BTC quote or if staged execution solves capacity.

---

## RFQ-006 — Miner 225 BTC, 90D, 85% floor / 120% cap collar

**Objective:** Verify cost of a practical quarterly production hedge.

**Client archetype:** miner with 450 BTC conservative 3-month production and 50% hedge ratio.

**Terms:**
- Hedge notional: 225 BTC.
- Tenor: approximately 90 days.
- Buy put: 85% of spot.
- Sell call: 120% of spot.
- Structure: cash-settled collar preferred.

**Fields required:**
- net premium;
- margin/collateral;
- settlement/index;
- size available;
- documentation;
- lender/collateral implications;
- stress payoff at BTC -20%, -30%, -40%.

**Decision rule:**
Advance if cost is lower than credible dilution/forced-sale risk and collateral is manageable.

---

## RFQ-007 — Miner 225 BTC, 90D, 80–85% disaster put

**Objective:** Verify standalone downside insurance cost.

**Client archetype:** miner wanting no upside cap.

**Terms:**
- Hedge notional: 225 BTC.
- Tenor: approximately 90 days.
- Buy put: request both 80% and 85% strikes.
- Settlement: cash-settled preferred.

**Fields required:**
- premium for 80% and 85% puts;
- delta/vega;
- liquidity/size;
- margin/collateral;
- stress payoff;
- quote good-until.

**Decision rule:**
Advance if premium is affordable relative to runway protected and no collateral issue emerges.

---

## RFQ-008 — Miner 225 BTC, 135D, put-spread collar

**Objective:** Verify longer-dated runway hedge with lower upfront cost.

**Client archetype:** miner with debt/capex/refi window.

**Terms:**
- Hedge notional: 225 BTC.
- Tenor: approximately 135 days.
- Buy put: 90% of spot.
- Sell put: 75% of spot.
- Sell call: 115% of spot.
- Buy call: 130% of spot.

**Fields required:** same as RFQ-006 plus leg-by-leg package price.

**Decision rule:**
Advance if protected band is meaningful and upside give-up is acceptable for a specific liability window.

---

## RFQ-009 — IBIT vs Deribit 30D ATM straddle package

**Objective:** Test whether screen-level ETF/Deribit ATM IV differences survive all-in costs.

**Client archetype:** internal RV / hedge-recycling engine, not first client product.

**Terms:**
- Tenor: closest common 30D expiry.
- Long leg: cheaper ATM straddle.
- Short leg: richer ATM straddle.
- Notional: BTC-equivalent matched.
- Delta hedge: define method and cost.
- ETF normalization: use official BTC/share.

**Fields required:**
- IBIT ATM call/put bid/ask;
- Deribit ATM call/put bid/ask;
- IV by leg;
- size available;
- fees;
- ETF share/BTC conversion;
- delta hedge cost;
- margin/collateral;
- basis/settlement mismatch;
- borrow/assignment issues.

**Decision rule:**
Advance only if net vol/premium spread survives all-in cost and wrapper basis risk is acceptable.

---

## RFQ-010 — IBIT vs Deribit 30D 25-delta risk reversal

**Objective:** Test whether skew differences are actionable after normalization.

**Client archetype:** internal RV / hedge-recycling engine; possibly useful for pricing collars.

**Terms:**
- Tenor: closest common 30D expiry.
- Structure: 25-delta put vs 25-delta call risk reversal.
- Compare IBIT and Deribit.
- Notional: BTC-equivalent matched.

**Fields required:**
- 25D put/call strikes by venue;
- bid/ask;
- IV;
- deltas;
- package price;
- size;
- fees;
- margin/collateral;
- settlement/basis risks;
- quote timestamp.

**Decision rule:**
Advance only if skew differential survives all-in cost and is repeatable across observations.

---

## 5. Quote capture fields

Every quote response should be entered using this structure:

```yaml
rfq_id:
quote_id:
created_at_cst:
counterparty:
contact:
legal_capacity: internal_research_only | counsel_review_required | external_partner_possible_later
quote_type: indicative | firm | executable_screen | trade
quote_status:
product:
client_archetype:
underlying_reference:
notional_btc_equivalent:
tenor_days:
expiry:
spot_reference:
legs:
  - side:
    option_type:
    strike_percent_of_spot:
    strike_absolute:
    quantity:
    bid:
    ask:
    mid:
    iv:
    delta:
    gamma:
    vega:
    theta:
net_premium:
premium_currency:
fees:
margin_initial:
margin_maintenance:
collateral_currency:
eligible_collateral:
settlement_method:
quote_timestamp_cst:
quote_good_until_cst:
size_available:
minimum_ticket:
documentation_required:
model_value:
difference_vs_screen:
all_in_cost:
scenario_pnl:
source_confidence:
execution_confidence:
legal_notes:
operational_notes:
decision:
follow_up:
```

---

## 6. Current missing inputs before live quote collection

### Legal / role
- Counsel-approved role for RFQ outreach.
- Compensation model decision.
- Whether outreach is internal market color, client-facing, principal, or partner-led.

### Counterparties
- Approved ETF options broker route.
- Approved FCM/CME route.
- Approved OTC options desk route.
- Approved crypto-native/Deribit route.
- Approved structured-product issuer route if relevant.

### Data
- Current spot reference source.
- Production ETF options source, ideally OPRA/vendor/broker.
- CME source.
- Deribit account/RFQ/block access status if used.
- BTC/share normalization feed.

### Operations
- Quote repository location.
- Versioning/audit trail.
- Person responsible for quote capture.
- Standard timestamp/timezone.
- Archive of raw counterparty responses.

---

## 7. Decision matrix

| Decision | Condition |
|---|---|
| Reject | quote worse than screen after costs, poor liquidity, legal issue, or unacceptable collateral |
| Watch | screen interest remains but quote weak/stale/incomplete |
| Requote | quote missing fields, stale, or inconsistent with market |
| Quote-verified | quote complete enough to evidence market economics |
| Client prototype | legal wrapper approved and product fits client policy |
| Trade candidate | risk/legal/counterparty approval plus executable quote |

---

## 8. First milestone

The first quote-verification milestone is **not execution**.

It is:

> Complete indicative/firm quote records for RFQ-001 through RFQ-008 from at least two legally approved quote sources, plus RFQ-009 and RFQ-010 from at least one ETF-options source and one Deribit/crypto-native source.

After that, update:
- treasury case study from screen-only to quote-verified;
- miner case study from screen-only to quote-verified;
- investor deck evidence slide;
- business-model economics;
- capacity estimates.

---

## 9. Bottom line

This board keeps the business honest.

The next proof threshold is not another memo. It is quote evidence. Until quotes are collected under an approved legal role, the current economics remain **screen-only illustrative research**.
