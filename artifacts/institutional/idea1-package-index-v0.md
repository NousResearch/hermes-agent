# Idea 1 Package Index v0 — BTC Volatility Intermediation Desk

**Concept:** BTC ETF / Deribit / CME / OTC volatility intermediation + BTC treasury/miner hedging platform.

**Purpose:** Consolidated artifact map for investor/client package development. This is the source-of-truth index for the institutional workstream.

---

## 1. Core positioning

Do not lead with:

> BTC options arbitrage fund.

Lead with:

> An institutional BTC risk-intermediation platform that helps treasuries, miners, funds, and allocators transform BTC volatility into income, protection, and structured risk — powered by cross-venue ETF/CME/Deribit/OTC volatility analytics.

The strongest wedge is **board-ready BTC treasury and miner hedging programs**. The cross-venue vol monitor is the pricing and hedge-recycling engine behind the business.

---

## 2. Current artifact stack

### 2.1 Master plan

**File:** `idea1-btc-vol-desk-plan.md`

**Use:** Internal roadmap and project control.

**Contains:**
- build sequence;
- proof-before-narrative discipline;
- product/data/client/legal/RFQ workstream order;
- immediate next steps.

---

### 2.2 Data evidence appendix

**File:** `idea1-data-evidence-appendix.md`

**Use:** Evidence spine for data availability and source confidence.

**Contains:**
- Deribit public API evidence;
- IBIT/Nasdaq semi-official option-chain path;
- BlackRock/iShares BTC-per-share normalization;
- OCC series/OI validation role;
- CME licensed/vendor/API path;
- OTC/RFQ quote-verification gap;
- normalized data schema;
- source-confidence taxonomy.

**Key conclusion:**
- Deribit is immediately usable for pilot research.
- IBIT is usable for pilot, but production needs OPRA/vendor/broker data.
- CME is strategically required but needs licensed/vendor/broker access.
- OTC economics are hypothesis until quote-verified.

---

### 2.3 BTC Vol Desk Monitor v0

**File:** `idea1-btc-vol-monitor-v0.md`

**Use:** First quantified screen-only monitor and dislocation board.

**Contains:**
- Deribit ATM IV term structure;
- IBIT ATM IV estimates;
- IBIT vs Deribit screen-only comparisons;
- dislocation labels;
- confidence labels;
- parser issue identified for iShares holdings.

**Key conclusion:**
- Screen-level differences exist, especially short-dated IBIT richness vs Deribit in the sample.
- This is **not executable arbitrage proof**.
- It is enough to justify quote verification.

---

### 2.4 Client map / commercial wedge

**File:** `idea1-client-map-commercial-wedge.md`

**Use:** Go-to-market and client-priority map.

**Contains:**
- BTC treasury companies;
- miners;
- RIAs/family offices/wealth platforms;
- hedge funds/RV funds;
- structured-product desks/private banks;
- venue fit by client;
- product-client matrix;
- outreach hooks;
- qualification checklist.

**Key conclusion:**
- First wedge: BTC treasury companies.
- Second wedge: Bitcoin miners.
- Do not start with hedge funds/RV funds unless edge is quote-verified.

---

### 2.5 Investor deep-dive v0

**File:** `idea1-investor-deep-dive-v0.md`

**Use:** Full investor-grade memo draft.

**Contains:**
- executive thesis;
- why now;
- product menu;
- data/analytics engine;
- natural clients;
- operating model;
- economics model;
- legal perimeter;
- risk controls;
- 30-day pilot;
- 90-day roadmap;
- team/infrastructure;
- investment case.

**Key conclusion:**
- The strongest launch product is not a prop-vol fund. It is a board-ready BTC treasury and miner hedging platform with a proprietary cross-venue volatility engine behind it.

---

### 2.6 Treasury case study — 10,000 BTC holder

**File:** `idea1-treasury-case-study-10000btc-v0.md`

**Use:** First board/client proof artifact.

**Contains:**
- hypothetical 10,000 BTC public treasury;
- 1,000 BTC / 10% program sleeve;
- 43-day and 134-day covered calls;
- collars;
- put-spread collars;
- recommended policy limits;
- board risk language.

**Key conclusion:**
- A 10% BTC sleeve can generate meaningful premium or defined protection without compromising the remaining 90% strategic BTC position.
- Best first structures: staggered 15–25% OTM covered calls and put-spread collars.

---

### 2.7 Miner production hedge case study

**File:** `idea1-miner-production-hedge-case-study-v0.md`

**Use:** Second board/client proof artifact.

**Contains:**
- hypothetical miner producing 150 BTC/month;
- 450 BTC conservative 3-month forecast;
- 225 BTC hedge / 50% hedge ratio;
- protective puts;
- production collars;
- put-spread collars;
- downside scenario impact;
- board/lender package language.

**Key conclusion:**
- The miner product is runway protection, not bearish speculation.
- Best first product: 85% floor / 120% cap quarterly collar or put-spread collar on conservative production.

---

### 2.8 Legal perimeter memo

**File:** `idea1-legal-perimeter-memo-v0.md`

**Use:** Counsel-ready legal gating map.

**Contains:**
- phase map: research, education, advisory, RFQ/execution, principal, structured products;
- instrument classification spine;
- SEC/CFTC/NFA/FINRA/RIA/BD/CTA/CPO/IB/FCM/swap-dealer gates;
- public-company treasury checklist;
- miner checklist;
- wealth/RIA/structured-product checklist;
- red/yellow/green activity matrix;
- counsel workstream.

**Key conclusion:**
- Clean MVP is research/analytics/illustrative case studies/legal design/RFQ planning.
- Client-specific advice, live RFQs, transaction compensation, and structured products require counsel-approved wrapper or partner model.

---

### 2.9 RFQ verification plan

**File:** `idea1-rfq-verification-plan-v0.md`

**Use:** Convert screen-only thesis into quote-verified evidence.

**Contains:**
- confidence ladder: hypothesis, screen-only, quote-verified, trade-verified;
- quote priorities;
- counterparty categories;
- RFQ templates for treasury covered calls, put-spread collars, miner hedges, cross-venue vol spreads;
- quote storage schema;
- quote comparison rules;
- first 10 internal RFQs.

**Key conclusion:**
- The RFQ workflow is the bridge from story to proof.
- No investor return claim until screen dislocations are quote-verified after costs, margin, collateral, and basis.

---

## 3. Package assembly order

### Internal diligence package

1. `idea1-package-index-v0.md`
2. `idea1-btc-vol-desk-plan.md`
3. `idea1-data-evidence-appendix.md`
4. `idea1-btc-vol-monitor-v0.md`
5. `idea1-legal-perimeter-memo-v0.md`
6. `idea1-rfq-verification-plan-v0.md`

### Investor package v1

1. One-page executive summary.
2. Investor deep-dive v0.
3. Data evidence appendix.
4. Treasury case study.
5. Miner case study.
6. Legal perimeter summary.
7. 30-day execution sprint.

### Client proof package — BTC treasury

1. BTC treasury one-page pitch.
2. 10,000 BTC case study.
3. Board policy template.
4. RFQ/quote-verification summary.
5. Legal/accounting/covenant checklist.

### Client proof package — miner

1. Miner runway hedge one-page pitch.
2. Miner production hedge case study.
3. Production/capex/runway calculator.
4. RFQ/quote-verification summary.
5. Lender/covenant checklist.

---

## 4. Open gaps

### Data gaps
- Harden iShares BTC/share parser.
- Add OPRA/vendor-grade ETF options source.
- Add CME licensed/vendor/broker source.
- Store daily surfaces historically.
- Add skew/risk-reversal/calendar analytics.

### Quote gaps
- No quotes have been collected yet.
- All economics remain screen-only.
- Need legal role approval before client-facing RFQs.

### Legal gaps
- Need counsel memo on permissible MVP activities.
- Need entity/wrapper plan.
- Need compensation model review.
- Need partner model for execution/RFQ if not registering.

### Product gaps
- Treasury board policy template not yet drafted.
- Miner hedge calculator not yet drafted.
- One-page executive summary not yet drafted.
- Deck outline not yet drafted.

---

## 5. Next build sequence

1. Draft one-page executive summary.
2. Draft 30-day execution sprint.
3. Draft deck outline.
4. Draft BTC treasury board policy template.
5. Draft miner hedge calculator spec.
6. Build quote-verification board from first 10 RFQs.
7. Harden data pipeline.

---

## 6. Final strategic note

The package is now beyond brainstorming. It has:

- product definition;
- data evidence;
- screen-level quantification;
- client mapping;
- proof case studies;
- legal gating;
- RFQ verification workflow.

The next threshold is quote verification and counsel-approved go-to-market design. Until then, all pricing remains **screen-only illustrative research**.
