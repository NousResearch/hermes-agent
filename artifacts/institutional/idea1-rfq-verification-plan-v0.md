# Idea 1 — RFQ Verification Plan v0

**INTERNAL RESEARCH DRAFT ONLY · NOT FOR EXTERNAL DISTRIBUTION · NO RFQ SUBMISSION · NO EXECUTION CAPACITY**

**Concept:** BTC volatility intermediation desk — ETF / Deribit / CME / OTC pricing verification for treasury collars, miner production hedges, and cross-venue BTC vol spreads.

**Prepared:** 2026-05-14 CDT  
**Status:** Internal verification design. Not client execution.  
**Legal status:** Research workflow only until counsel approves RFQ/execution role. Do not send client RFQs, introduce client orders, solicit transactions, or accept transaction compensation without legal wrapper.

---

## 1. Purpose

The RFQ plan converts the current workstream from **screen-only thesis** into **quote-verified evidence**.

The goal is not to execute trades immediately. The goal is to determine whether the economics shown by the BTC Vol Desk Monitor survive real-world quote frictions:

- bid/ask width;
- size;
- counterparty willingness;
- margin/collateral terms;
- execution venue restrictions;
- legal onboarding;
- slippage;
- quote freshness;
- basis/wrapper mismatch;
- settlement and custody constraints.

The rule:

> No screen-only dislocation becomes an investor-facing return claim until at least two executable or indicative quotes confirm the structure after fees, collateral, and basis adjustments.

---

## 2. Confidence ladder

Every opportunity and product structure gets one of four execution labels.

### Level 0 — Hypothesis

**Definition:** Economic idea only. No live data.

Example:
- “ETF puts may be richer than Deribit puts because ETF holders demand protection.”

Use:
- brainstorming;
- market-structure memo;
- target research.

Do not use:
- investor return claims;
- client term sheets;
- revenue forecasts.

---

### Level 1 — Screen-only

**Definition:** Visible from public/vendor screens but not confirmed executable.

Example:
- Deribit mark IV vs Nasdaq IBIT mid-derived IV shows a 2–5 vol point difference.

Requirements:
- timestamp;
- source;
- bid/ask/mid/mark;
- size/OI/volume if available;
- calculation method;
- source-confidence label.

Use:
- research appendix;
- internal dislocation board;
- illustrative case studies.

Do not use:
- “tradable edge” language;
- expected-return claims;
- client execution recommendation.

---

### Level 2 — Quote-verified

**Definition:** Counterparty or venue quote received for a defined structure, with quote timestamp, size, expiry, strike, settlement, collateral/margin terms, and quote status.

Quote types:
- indicative quote;
- firm quote with expiry window;
- broker market color;
- executable screen quote from approved account;
- dealer RFQ response.

Requirements:
- at least one quote for a defined structure;
- ideally two independent quotes;
- exact quote fields stored;
- quote age tracked;
- legal role clarified before any client-facing use.

Use:
- quote-verification appendix;
- investor evidence, carefully labeled;
- pricing sanity checks.

Do not use:
- “trade-verified” or realized economics language.

---

### Level 3 — Trade-verified

**Definition:** Executed trade or firm executable quote in size within quoted window.

Requirements:
- execution confirmation or firm quote proof;
- size;
- all-in fees;
- margin/collateral impact;
- post-trade mark;
- hedge/recycling result;
- realized slippage vs model;
- compliance/approval record.

Use:
- investor performance evidence;
- operating model calibration;
- capacity estimates.

Do not use:
- broad scaling claims unless repeated across conditions.

---

## 3. What must be quote-verified first

### Priority 1 — BTC treasury covered-call package

**Why first:** Most commercially intuitive product. Quotes should be easy to compare.

**Standard RFQ package:**
- Underlying: BTC or IBIT-equivalent BTC exposure.
- Notional: 500 BTC and 1,000 BTC scenarios.
- Tenors: 30D, 45D, 90D, 135D.
- Strikes: 15%, 20%, 25% OTM calls.
- Settlement: cash-settled BTC/USD or ETF shares depending wrapper.
- Quote: premium received, bid/ask, collateral/margin, fees.

**Verification objective:**
- Can a treasury monetize a 5–10% sleeve at attractive premium without unacceptable collateral or upside cap?

---

### Priority 2 — BTC treasury put-spread collar

**Why second:** Better governance than tight full collars.

**Standard RFQ package:**
- Notional: 500 BTC and 1,000 BTC.
- Tenors: 90D and 135D.
- Put spread: buy 90% put / sell 75–80% put.
- Call-spread financing: sell 115–120% call / buy 125–140% call.
- Goal: define drawdown band, avoid unlimited upside give-up.

**Verification objective:**
- Can the desk create a board-friendly floor at modest cost while retaining upside above the call spread?

---

### Priority 3 — Miner production collar

**Why third:** Natural recurring product with clear pain.

**Standard RFQ package:**
- Client exposure: 3-month conservative production.
- Notional: 100 BTC, 225 BTC, 500 BTC.
- Tenors: 45D, 90D, 135D.
- Structure: buy 80–90% put, sell 115–125% call.
- Variants: put-spread collar and outright disaster put.

**Verification objective:**
- Can miners protect runway at a cost that is cheaper than emergency equity/dilution or forced spot liquidation?

---

### Priority 4 — ETF vs Deribit / CME vol spread package

**Why fourth:** This is the intellectual alpha, but it needs stronger proof.

**Standard RFQ package:**
- Compare equivalent expiry/strike/moneyness across:
  - IBIT options;
  - Deribit BTC options;
  - CME BTC options when data/access available;
  - OTC dealer quote.
- Structures:
  - ATM straddle spread;
  - 25-delta risk reversal;
  - 90/110 collar package;
  - calendar spread.

**Verification objective:**
- Does the screen-only IV gap survive all-in execution, hedging, margin, collateral, fees, and wrapper/basis adjustments?

---

## 4. Counterparty categories

### A. Listed ETF options route

**Use for:**
- IBIT/spot BTC ETF covered calls;
- ETF collars;
- ETF-option wealth overlays;
- comparison against Deribit/CME.

**Possible sources:**
- listed options broker;
- prime broker;
- institutional options desk;
- OPRA/vendor data;
- OCC series validation.

**Required fields:**
- ETF ticker;
- expiry;
- strike;
- bid/ask;
- size at bid/ask;
- implied vol;
- delta/gamma/vega/theta;
- open interest/volume;
- position limits;
- assignment/exercise mechanics;
- margin requirement;
- commission/fees;
- timestamp.

**Production data caveat:**
Nasdaq public JSON is research-use only. Production should use OPRA/vendor/broker data.

---

### B. CME route

**Use for:**
- regulated futures/options overlay;
- institutional funds;
- miner/treasury clients preferring FCM route;
- basis/vol package comparison.

**Possible sources:**
- FCM;
- broker desk;
- CME DataMine/API/vendor;
- terminal export.

**Required fields:**
- futures contract reference;
- option contract;
- expiry;
- strike;
- bid/ask;
- futures price/basis;
- implied vol;
- contract multiplier;
- initial/maintenance margin;
- block liquidity;
- fees;
- position/accountability limits;
- settlement method;
- timestamp.

**Caveat:**
CME public pages are not enough for production quotes.

---

### C. Deribit / crypto-native route

**Use for:**
- BTC options surface reference;
- crypto-native hedge/recycling;
- non-US sophisticated clients where legally permitted;
- internal screen monitor.

**Possible sources:**
- Deribit public API;
- Deribit RFQ/block if account eligible;
- crypto options market makers;
- prime/custody-integrated access provider.

**Required fields:**
- instrument name;
- expiry;
- strike;
- call/put;
- bid/ask/mark;
- mark IV;
- Greeks;
- open interest/volume;
- underlying/index price;
- settlement currency;
- margin requirement;
- collateral currency;
- liquidation/margin model;
- counterparty/venue restrictions;
- timestamp.

**Caveat:**
Deribit may not be usable for US clients. Treat as pricing/hedge venue only after legal review.

---

### D. OTC/RFQ route

**Use for:**
- bespoke treasury/miner collars;
- block size;
- custom maturities;
- public-company friendly term sheets;
- structured-product hedges.

**Possible sources:**
- crypto OTC options desks;
- bank derivatives desks if available;
- prime brokers;
- market makers;
- structured-products issuers.

**Required fields:**
- counterparty;
- quote type: indicative vs firm;
- quote expiry time;
- structure legs;
- notional;
- premium/price;
- bid/offer spread;
- collateral/CSA terms;
- eligible collateral;
- haircuts;
- margin call timing;
- settlement method;
- documentation required;
- onboarding timeline;
- minimum ticket size;
- legal eligibility requirements;
- valuation agent;
- closeout/disruption terms.

**Caveat:**
OTC economics are hypothesis until quotes are received.

---

## 5. Standard RFQ template — treasury covered call

### RFQ header

- RFQ ID:
- Date/time/timezone:
- Requestor:
- Counterparty:
- Quote type requested: indicative / firm
- Quote good-until:
- Legal capacity: internal research only / client-facing approved / principal
- Confidentiality level:

### Client/exposure assumptions

- Exposure type: direct BTC / ETF shares / futures / mixed
- Total exposure:
- Hedge sleeve:
- Custody/venue restrictions:
- Settlement preference: cash / physical / ETF shares
- Collateral currency: USD / BTC / T-bills / stablecoin / other

### Structure

- Product: covered call
- Underlying/reference:
- Notional:
- Expiry:
- Strike:
- Moneyness:
- American/European:
- Settlement index/source:
- Premium currency:

### Quote response fields

- Bid premium:
- Ask premium:
- Mid/indicative:
- Implied vol:
- Delta:
- Gamma:
- Vega:
- Theta:
- Upfront premium:
- Fees/commission:
- Margin/collateral requirement:
- Minimum ticket:
- Size available:
- Quote timestamp:
- Quote expiry:
- Notes:

### Verification notes

- Screen model value:
- Difference vs screen:
- All-in cost/spread:
- Execution confidence:
- Source confidence:
- Follow-up required:

---

## 6. Standard RFQ template — treasury put-spread collar

### Structure

- Product: put-spread collar
- Underlying/reference:
- Notional:
- Expiry:
- Buy put strike:
- Sell put strike:
- Sell call strike:
- Buy call strike:
- Target net premium: zero-cost / debit / credit
- Settlement:
- Premium currency:

### Quote response fields

- Net premium/debit/credit:
- Each leg bid/ask:
- Net implied vol by leg:
- Package delta/gamma/vega/theta:
- Max protection:
- Max upside give-up:
- Collateral/margin:
- Fees:
- Size available:
- Quote good-until:
- Valuation source:
- Disruption/fallback notes if OTC:

### Verification notes

- Cost as % of protected notional:
- Floor level:
- Cap/sold band:
- Scenario P&L at BTC -20%, -30%, +20%, +50%:
- Fit for board package: yes/no:
- Main legal/operational blockers:

---

## 7. Standard RFQ template — miner production hedge

### Client/exposure assumptions

- Miner:
- Monthly conservative production:
- Hedge horizon:
- Total forecast production:
- Hedge ratio:
- Hedge BTC:
- Existing BTC inventory:
- Pledged/restricted BTC:
- Power/capex/debt dates:
- Settlement preference:
- Approved venues/counterparties:

### Structure options requested

Request quotes for three alternatives:

1. **Outright disaster put**
   - Notional:
   - Expiry:
   - Strike: 80–85% of spot

2. **Production collar**
   - Buy put: 80–90% of spot
   - Sell call: 115–125% of spot

3. **Put-spread collar**
   - Buy put: 90% of spot
   - Sell put: 75–80% of spot
   - Sell call: 115–120% of spot
   - Buy call: 125–140% of spot

### Quote response fields

- Net premium by structure:
- Cost as % of hedged notional:
- Initial/variation margin:
- Eligible collateral:
- Size available:
- Minimum ticket:
- Settlement/index:
- Quote validity:
- Documentation/onboarding required:
- Operational constraints:

### Verification notes

- Runway protection at BTC -20%, -30%, -40%:
- Premium vs emergency equity/dilution cost:
- Lender/covenant compatibility:
- Overhedging risk:
- Board fit:

---

## 8. Standard RFQ template — cross-venue vol spread

### Structure

- Product: cross-venue vol spread
- Dislocation type: wrapper / skew / calendar / basis / margin / flow
- Long leg:
- Short leg:
- Underlying conversion:
- Expiry matching:
- Strike/moneyness matching:
- Delta hedge:
- Basis hedge:
- Notional:

### Quote response fields

For each leg:
- venue;
- instrument;
- bid/ask;
- IV;
- Greeks;
- size;
- fees;
- margin/collateral;
- settlement;
- execution timestamp.

Package fields:
- gross IV spread;
- premium spread;
- delta-neutral cost;
- basis adjustment;
- financing/collateral adjustment;
- expected slippage;
- net spread;
- capacity;
- stress/basis risk.

### Verification notes

- Does the screen IV gap survive all-in costs?
- Which leg is hard to execute?
- Which venue creates margin/collateral drag?
- Does basis/wrapper risk dominate?
- Is the spread client-useful or only prop-tradable?

---

## 9. Quote storage schema

Every quote should be stored with this schema.

```yaml
quote_id:
created_at:
requested_by:
counterparty:
quote_type: indicative | firm | executable_screen | trade
source_confidence:
execution_confidence:
legal_capacity: internal_research_only | counsel_review_required | external_partner_possible_later
client_archetype: treasury | miner | fund | family_office | structured_product | internal
product_type: covered_call | collar | put_spread_collar | miner_hedge | vol_spread | structured_note
underlying:
venue:
notional:
expiry:
legs:
  - side:
    type:
    strike:
    quantity:
    bid:
    ask:
    mid:
    iv:
    delta:
    gamma:
    vega:
    theta:
premium_currency:
net_premium:
fees:
margin_initial:
margin_maintenance:
collateral_currency:
eligible_collateral:
settlement_method:
quote_timestamp:
quote_good_until:
size_available:
minimum_ticket:
documentation_required:
legal_notes:
operational_notes:
model_value:
difference_vs_model:
all_in_cost:
scenario_pnl:
verification_status:
follow_up:
```

---

## 10. Quote comparison rules

### Rule 1 — never compare IV alone

Compare:
- premium;
- IV;
- delta/vega;
- expiry;
- forward/basis;
- collateral cost;
- fees;
- settlement;
- size;
- legal usability.

### Rule 2 — normalize to BTC-equivalent exposure

For ETF options:
- convert ETF shares to BTC equivalent using official ETF holdings/NAV data.
- account for ETF expense drag, tracking, and share settlement.

For CME:
- account for futures basis and contract multiplier.

For Deribit:
- account for index, settlement currency, margin/collateral, and jurisdiction.

For OTC:
- account for CSA, credit, valuation, disruption clauses, and documentation cost.

### Rule 3 — all-in economics beat screen marks

A trade is attractive only if it survives:
- bid/ask;
- slippage;
- fees;
- financing;
- collateral drag;
- hedging cost;
- tax/accounting friction if relevant;
- legal/operational feasibility.

### Rule 4 — quote age matters

For BTC options:
- <5 minutes: fresh;
- 5–30 minutes: stale risk;
- >30 minutes: market color only unless quiet market and counterparty confirms.

### Rule 5 — separate client product from hedge venue

A treasury client may need OTC or CME governance, while the desk may observe cheaper hedge risk on Deribit. Do not assume client can use the hedge venue directly.

---

## 11. Verification workflow

### Step 1 — screen candidate

From BTC Vol Desk Monitor:
- identify structure;
- record screen prices;
- calculate model economics;
- label screen-only.

### Step 2 — quote request package

Prepare RFQ with:
- exact legs;
- notional;
- expiry;
- settlement;
- collateral preference;
- quote type requested;
- no client name unless legally approved.

### Step 3 — receive quote

Record:
- price;
- size;
- quote age;
- margin/collateral;
- fees;
- constraints;
- whether firm or indicative.

### Step 4 — compare to model

Calculate:
- gross edge;
- all-in edge;
- collateral impact;
- basis/wrapper adjustment;
- execution confidence.

### Step 5 — decision label

Assign:
- reject — no edge;
- research — keep watching;
- quote-verified — worth investor evidence;
- client prototype — if legal and product fit;
- trade candidate — only if legal/risk approval.

### Step 6 — archive

Store quote and reasoning. Do not rely on memory.

---

## 12. Minimum quote set before investor claims

Before any investor memo claims “there is executable edge,” collect at least:

### Treasury products
- 2 quotes for 30–45D covered calls.
- 2 quotes for 90–135D covered calls.
- 2 quotes for 90–135D put-spread collars.
- At least one ETF/options route and one OTC/crypto-native route if possible.

### Miner products
- 2 quotes for 3-month production collars.
- 2 quotes for disaster puts.
- 2 quotes for put-spread collars.

### Cross-venue RV
- 5–10 repeated observations where screen spread is followed by quote verification.
- At least 2 venues per package.
- Evidence that net edge survives all-in costs.
- Capacity estimate.

### CME integration
- At least one licensed/vendor/broker source for CME chain and margin.
- Do not use CME in return claims from public-page data alone.

---

## 13. Counterparty due diligence checklist

For each potential counterparty:

- legal entity;
- regulatory status;
- jurisdiction;
- approved products;
- minimum ticket size;
- supported collateral;
- custody/settlement process;
- ISDA/CSA availability;
- FCM/broker relationship;
- credit support/parent guarantee;
- valuation agent process;
- dispute process;
- fees and markups;
- quote response speed;
- stress-market performance;
- conflicts;
- sanctions/AML/KYC process.

---

## 14. What not to do

Do not:
- send client-specific RFQs before legal approval;
- accept transaction compensation before legal approval;
- call screen-only data executable;
- use stale quotes as live pricing;
- compare Deribit IV to IBIT IV without wrapper/basis/margin adjustment;
- imply Deribit is available to US clients without legal review;
- quote a public company structure without board/accounting/covenant diligence;
- build investor return claims from one-day screen snapshots.

---

## 15. First 10 RFQs to design internally

These are internal templates only until counsel approves outreach.

1. 500 BTC, 45D, 15% OTM covered call.
2. 500 BTC, 45D, 25% OTM covered call.
3. 1,000 BTC, 135D, 20% OTM covered call.
4. 500 BTC, 135D, 90/80 put spread funded by 115/130 call spread.
5. 1,000 BTC, 135D, 90/80 put spread funded by 115/130 call spread.
6. Miner 225 BTC, 90D, 85% floor / 120% cap collar.
7. Miner 225 BTC, 90D, 80–85% outright disaster put.
8. Miner 225 BTC, 135D, 90/75 put spread funded by 115/130 call spread.
9. IBIT vs Deribit matched 30D ATM straddle package.
10. IBIT vs Deribit matched 25-delta risk reversal package.

---

## 16. Bottom line

The RFQ workflow is the bridge from story to proof.

Current status:
- Product thesis: developed.
- Data map: developed.
- Screen monitor: developed.
- Treasury/miner case studies: developed.
- Legal perimeter: developed.
- RFQ workflow: now defined.

Next proof milestone:

> Convert the first 10 internal RFQ templates into a quote-verification board, then collect quotes only after legal role/partner path is approved.
