# Idea 1 — 30-Day Execution Sprint v0

**Concept:** BTC Volatility Intermediation Desk  
**Goal:** Convert current evidence package into a counsel-ready, investor-ready, and quote-verification-ready institutional concept.

**Status:** Internal execution plan. No client RFQs or client-specific advice until legal wrapper is approved.

---

## Sprint objective

By day 30, produce a complete launch decision package answering:

1. Is the data pipeline credible enough to run a daily BTC Vol Desk Monitor?
2. Are treasury/miner structures commercially compelling under screen-only assumptions?
3. What quotes must be collected to prove economics?
4. What legal wrapper or partner model is required?
5. Is this best launched as analytics/advisory, fund/prop, execution partnership, or hybrid?

---

## Non-negotiable rules

- Every market claim gets a confidence label: hypothesis, screen-only, quote-verified, trade-verified.
- No screen-only dislocation becomes a return claim.
- No client-specific RFQ is sent before legal approval.
- No transaction compensation before counsel-approved model.
- No Deribit use is assumed for US clients without legal review.
- Public-company programs require board/accounting/covenant framing.

---

## Week 1 — Data spine and monitor hardening

### Deliverables

1. **Daily BTC Vol Desk Monitor v1**
   - Deribit BTC options ingestion.
   - IBIT option-chain ingestion from research source.
   - iShares BTC/share normalization parser hardened.
   - OCC series/OI validation path.
   - CME source decision: broker/vendor/API/manual terminal export.

2. **Normalized schema implementation**
   - venue;
   - wrapper;
   - expiry/DTE;
   - strike/moneyness;
   - bid/ask/mid/mark;
   - IV;
   - Greeks where available;
   - OI/volume;
   - basis/forward assumption;
   - margin/collateral notes;
   - source confidence;
   - execution confidence.

3. **First historical store**
   - save daily snapshots;
   - preserve raw API/source output;
   - preserve computed monitor output;
   - timestamp in CST.

### Success criteria

- Monitor runs daily without manual scraping confusion.
- Deribit and IBIT surfaces are comparable by expiry/moneyness.
- iShares BTC/share conversion is reliable.
- Outputs distinguish screen-only from quote-verified.
- CME plan is explicit even if not integrated.

### Failure criteria

- Public data is too unstable to produce consistent monitor.
- IV calculations cannot be reconciled to vendor/broker values.
- No viable CME data path is identified.

---

## Week 2 — Product calculators and board packages

### Deliverables

1. **Treasury calculator v0**
   - Input: BTC holdings, sleeve %, tenor, floor, cap, call OTM target.
   - Output: premium/yield, downside floor, upside cap, BTC encumbrance, scenario table.
   - Structures:
     - covered call;
     - full collar;
     - put-spread collar;
     - call-spread-financed hedge.

2. **Miner hedge calculator v0**
   - Input: monthly production, conservative forecast, hedge ratio, power/capex/debt dates.
   - Output: hedge notional, premium, runway protection, BTC -20/-30/-40 scenarios.
   - Structures:
     - outright disaster put;
     - production collar;
     - put-spread collar.

3. **BTC treasury board policy template**
   - objectives;
   - permitted instruments;
   - max sleeve;
   - tenor/strike/floor limits;
   - collateral/custody policy;
   - counterparty limits;
   - valuation/reporting;
   - approval authority;
   - disclosure triggers.

4. **Miner board/lender memo template**
   - production forecast methodology;
   - hedge ratio policy;
   - runway/capex linkage;
   - lender/covenant review;
   - collateral plan;
   - scenario stress table.

### Success criteria

- Treasury and miner case studies become reusable calculators/templates.
- Board language is disciplined and not promotional.
- Outputs explicitly label all prices as screen-only unless quote-verified.

---

## Week 3 — Legal wrapper and RFQ verification board

### Deliverables

1. **Counsel-ready activity map**
   - research;
   - generic education;
   - bespoke advisory;
   - RFQ/execution;
   - principal risk;
   - structured products;
   - compensation models.

2. **Legal questions memo**
   - CTA/CPO;
   - RIA;
   - broker-dealer/FINRA;
   - IB/FCM;
   - swap dealer;
   - public-company treasury;
   - structured products;
   - Deribit/offshore access.

3. **RFQ verification board v0**
   - first 10 internal RFQ templates;
   - quote fields;
   - required counterparties;
   - legal capacity status;
   - quote priority;
   - no client names.

4. **Counterparty map**
   - ETF/options broker route;
   - FCM/CME route;
   - crypto-native/Deribit route;
   - OTC options desk route;
   - structured-product issuer route.

### Success criteria

- Counsel can answer concrete questions, not vague “can we do this?”
- RFQ board is ready but not sent until legal role is approved.
- Compensation models are separated: subscription, advisory fee, execution fee, spread, principal P&L.

---

## Week 4 — Investor package and launch decision

### Deliverables

1. **Investor deck outline**
   - problem;
   - why now;
   - market structure;
   - product menu;
   - data evidence;
   - screen monitor;
   - treasury case;
   - miner case;
   - legal path;
   - RFQ plan;
   - 90-day roadmap;
   - capital/team needs.

2. **Investor memo v1 refresh**
   - integrate one-page summary;
   - tighten legal caveats;
   - add sprint outputs;
   - clarify quote-verification milestones.

3. **Launch model decision matrix**
   - analytics/research company;
   - advisory/structuring with counsel-approved registration/partner;
   - fund/prop vehicle;
   - execution/RFQ partner model;
   - structured-product partnership;
   - hybrid.

4. **Go/no-go decision package**
   - data readiness;
   - quote readiness;
   - legal readiness;
   - commercial readiness;
   - capital/team readiness.

### Success criteria

- The concept can be explained in one page and defended in appendix depth.
- Investor materials do not overclaim screen-only economics.
- Next 90 days are concrete: data, counsel, quotes, counterparties, pilot clients.

---

## Workstream owners / roles needed

### Quant/data
- Build monitor and data store.
- Normalize option surfaces.
- Validate IV calculations.

### Structuring/product
- Treasury calculator.
- Miner hedge calculator.
- Board templates.
- RFQ templates.

### Legal/compliance
- Activity map.
- Registration/wrapper analysis.
- Communications policy.
- Client eligibility.

### Sales/capital markets
- Target account list.
- Persona map.
- Counterparty map.
- Outreach language after legal approval.

### Risk/operations
- Margin/collateral model.
- Counterparty diligence.
- Valuation/marks process.
- Scenario/stress reporting.

---

## 30-day artifacts checklist

Already created:

- `idea1-package-index-v0.md`
- `idea1-btc-vol-desk-plan.md`
- `idea1-data-evidence-appendix.md`
- `idea1-btc-vol-monitor-v0.md`
- `idea1-client-map-commercial-wedge.md`
- `idea1-investor-deep-dive-v0.md`
- `idea1-treasury-case-study-10000btc-v0.md`
- `idea1-miner-production-hedge-case-study-v0.md`
- `idea1-legal-perimeter-memo-v0.md`
- `idea1-rfq-verification-plan-v0.md`
- `idea1-one-page-executive-summary-v0.md`

To create next:

- `idea1-investor-deck-outline-v0.md`
- `idea1-treasury-board-policy-template-v0.md`
- `idea1-miner-hedge-calculator-spec-v0.md`
- `idea1-rfq-verification-board-v0.md`
- `idea1-launch-model-decision-matrix-v0.md`

---

## Decision gates

### Gate 1 — Data gate
Proceed only if daily monitor can reliably compute normalized surfaces and store snapshots.

### Gate 2 — Legal gate
Proceed to client-facing work only if counsel approves the activity, compensation, and wrapper.

### Gate 3 — Quote gate
Proceed to investor return/economics claims only after quote verification.

### Gate 4 — Commercial gate
Proceed to outreach only if treasury/miner packages are board-ready, simple, and legally reviewed.

### Gate 5 — Capital gate
Proceed to principal risk only with risk capital, limits, counterparties, and compliance.

---

## Bottom line

The next 30 days should not be “raise money on a deck.”

The next 30 days should produce a defensible launch package:

- daily data evidence;
- board-ready treasury/miner products;
- legal wrapper clarity;
- RFQ verification readiness;
- launch model decision.

That is how this becomes a serious institutional business rather than a clever trade idea.
