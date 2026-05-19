# Idea 1 Investor Deep-Dive v0

**Concept:** BTC Volatility Intermediation Desk — ETF / Deribit / CME / OTC volatility analytics, treasury yield/protection, and miner hedging.

**Prepared:** 2026-05-14 19:10 CDT

**Status:** Investor-grade first draft, but not final. Data evidence is screen-only unless explicitly marked otherwise. Legal/regulatory structure requires counsel before any execution or client-facing activity.

---

## 1. Executive thesis

Bitcoin has become institutionally accessible through multiple wrappers: spot ETFs, CME futures/options, crypto-native options, and OTC derivatives. These wrappers reference substantially similar economic exposure but trade through different legal, margin, collateral, custody, and client-flow regimes.

That fragmentation creates a business opportunity: build an institutional BTC risk-intermediation platform that helps treasuries, miners, funds, and allocators transform BTC volatility into yield, protection, and structured risk.

The desk should not be framed as a generic crypto-options prop fund. The higher-quality institutional wedge is:

> Board-ready BTC treasury and miner hedging programs, powered by a proprietary cross-venue BTC volatility monitor.

The analytics engine normalizes ETF options, CME futures/options, Deribit options, and OTC/RFQ quotes into BTC-equivalent terms. The client-facing business translates that intelligence into covered calls, collars, put spreads, production hedges, and eventually structured notes.

---

## 2. Why now

### 2.1 BTC moved into institutional wrappers

The BTC market is no longer only spot exchange flow. Institutions can now access BTC exposure through:

- US spot BTC ETFs and listed ETF options.
- CME Bitcoin futures and options.
- Crypto-native options venues such as Deribit.
- OTC bilateral derivatives and structured products.

Each wrapper attracts different users and constraints:

- ETF options: securities accounts, RIAs, family offices, public-company treasury comfort, OCC-listed options.
- CME: regulated futures accounts, FCM margin, institutional futures infrastructure.
- Deribit: deep crypto-native vol surface, broad strikes/expiries, 24/7 market, crypto collateral.
- OTC/RFQ: custom tenors, notional sizes, payoff shapes, legal/collateral terms.

### 2.2 BTC treasury companies and miners create natural client flow

BTC treasury companies and miners are structurally long BTC. But they also face fiat obligations: debt, preferred dividends, operating expenses, power costs, ASIC purchases, site capex, and public-market dilution pressure.

They need a framework that lets them:

- monetize volatility without selling core BTC;
- buy disaster protection without abandoning upside;
- protect capex/runway windows;
- reduce forced-equity or forced-BTC-sale risk;
- explain the policy to boards, auditors, lenders, and investors.

### 2.3 Volatility fragmentation creates pricing and hedging intelligence

A BTC option is not just a BTC option. The economic comparison depends on:

- ETF wrapper mechanics;
- futures basis;
- BTC/share conversion;
- contract multipliers;
- fees and expense drag;
- bid/ask and executable size;
- margin and collateral currency;
- custody/venue eligibility;
- assignment and settlement mechanics;
- regulatory/mandate restrictions.

Most simple IV comparisons ignore these. The desk’s edge is to normalize them.

---

## 3. Core product menu

### 3.1 BTC treasury covered-call program

**Client:** BTC treasury companies, operating companies with BTC reserves, wealth platforms holding BTC ETFs.

**Structure:** Sell calls against a defined minority tranche of BTC/ETF exposure.

**Purpose:** Generate recurring premium income without selling spot BTC.

**Variants:**

- monthly OTM covered-call ladder;
- quarterly covered-call program;
- call-spread overwrite to preserve extreme upside;
- BTC-denominated premium reinvestment;
- ETF-option version for securities-account clients.

**Revenue model:** Structuring fee, execution commission/spread, recurring monitoring/reporting retainer, later principal risk spread if permitted.

**Key control:** Do not cap too much upside. Keep notional modest and policy-driven.

---

### 3.2 BTC treasury collar / downside floor

**Client:** BTC treasury companies, family offices, miners with BTC inventory, firms with debt/preferred obligations.

**Structure:** Buy puts or put spreads; finance partly/fully by selling calls or call spreads.

**Purpose:** Create a board-approved downside floor while preserving strategic BTC ownership.

**Variants:**

- 3-month zero-cost collar;
- 6-month put-spread collar;
- debt/coupon/refi-window protection;
- LTV/collateral protection around BTC-backed loans;
- drawdown-budget hedge for public-company boards.

**Revenue model:** Structuring fee, RFQ/execution spread, advisory retainer, risk recycling.

**Key control:** Define maximum BTC encumbrance, acceptable upside cap, and collateral process before quoting.

---

### 3.3 Miner production hedge

**Client:** Public/private Bitcoin miners.

**Structure:** Hedge a conservative slice of expected production or unencumbered BTC inventory using puts, put spreads, collars, or limited forwards.

**Purpose:** Protect cash-flow runway for power, capex, debt service, and ASIC/site investment.

**Variants:**

- monthly production collar;
- 3–6 month put spread on conservative production forecast;
- treasury BTC collar on 10–30% of unencumbered inventory;
- BTC-backed loan collateral protection;
- power/hashprice stress-window hedge.

**Revenue model:** Advisory/structuring, recurring execution, monitoring/reporting retainer.

**Key control:** Do not overhedge production. Size to conservative output, not optimistic hashrate targets.

---

### 3.4 Cross-venue BTC vol spread package

**Client:** Hedge funds, macro/RV funds, market makers, internal prop sleeve.

**Structure:** Relative-value packages across ETF options, CME, Deribit, and OTC.

**Examples:**

- ETF-vs-Deribit ATM vol spread;
- skew/risk-reversal spread;
- calendar spread;
- CME futures/options basis-vol package;
- event-vol package;
- client-flow hedge recycling.

**Revenue model:** Analytics subscription, execution/RFQ commission, later principal spread.

**Key control:** Only promote as an investor return engine after quote-verified or trade-verified evidence.

---

### 3.5 Structured notes / defined-outcome products

**Client:** Private banks, structured-product distributors, family offices, wealth platforms.

**Structure:** BTC-linked reverse convertibles, autocallables, buffered notes, principal-protected notes, income notes.

**Purpose:** Package BTC volatility into defined yield/upside/downside profiles.

**Revenue model:** Structuring margin, hedge spread, distribution economics.

**Key control:** Phase 2 only. Legal, suitability, issuer, disclosure, and valuation complexity is materially higher.

---

## 4. Data and analytics engine

### 4.1 Verified and available sources

**Deribit:** Verified public API. Live BTC options book summary returned ~936 contracts across 12 expiries in the first monitor pass. Fields include bid/ask/mid/mark, mark IV, underlying price, OI, volume, and instrument metadata.

**IBIT/Nasdaq:** Public Nasdaq JSON returned IBIT option chain records and bid/ask data. This is usable for pilot research but semi-official/undocumented. Production-grade use requires OPRA/vendor data.

**BlackRock/iShares:** Official holdings source supports BTC/share normalization. Previously verified official values showed BTC/share around `0.000564212717166`, or roughly `1,772.38` IBIT shares per BTC, using observed BTC holdings and shares outstanding.

**OCC:** Official listed-series/OI source, useful for series validation but not quotes/IV.

**CME:** Official API/feed paths exist, but public pages should not be scraped. Production access requires credentials/licensing, DataMine, WebSocket/MDP feed, broker/vendor access, or terminal export.

**OTC/RFQ:** Not yet verified. Must be quote-verified before economics are treated as executable.

### 4.2 First quantified screen-only monitor

On the first v0 monitor, Deribit ATM IV term structure was roughly:

- DTE 1: ~31.3%
- DTE 4: ~30.1%
- DTE 8: ~34.9%
- DTE 15: ~36.3%
- DTE 43: ~38.0%
- DTE 134: ~40.8%
- DTE 225: ~43.7%

IBIT ATM IV estimates from Nasdaq mid prices and a rough Black-Scholes model were roughly:

- DTE 1: ~41.5%
- DTE 4: ~34.1%
- DTE 8: ~36.9%
- DTE 15: ~37.9%

Initial screen-only comparison:

- DTE 1: IBIT richer than Deribit by ~10 vol points.
- DTE 4: IBIT richer by ~4 vol points.
- DTE 8: IBIT richer by ~2 vol points.
- DTE 15: IBIT richer by ~1.5–2 vol points.

This does not prove arbitrage. It proves there are screen-level differences worth quote verification.

### 4.3 Monitor output standard

Every candidate must show:

- dislocation type: wrapper, skew, calendar, basis, margin/collateral, flow;
- exact legs;
- native prices;
- BTC-equivalent normalized prices;
- gross spread;
- estimated costs;
- net spread;
- size/capacity;
- client application;
- risk notes;
- source confidence;
- execution confidence.

Confidence labels:

- **Hypothesis:** theory only.
- **Screen-only:** visible from public/vendor screens.
- **Quote-verified:** indicative counterparty quote received.
- **Trade-verified:** executed trade or firm executable quote.

---

## 5. Natural client map

### 5.1 BTC treasury companies — first wedge

**Examples:** Strategy/MicroStrategy, Metaplanet, Twenty One Capital, Semler Scientific, KULR, The Blockchain Group/Capital B, BTC-heavy miners.

**Pain:** Capital raises, dilution optics, preferred/convert obligations, operating runway, BTC NAV volatility.

**Product fit:** Covered calls, zero-cost collars, put-spread collars, policy/reporting package.

**Hook:**

> Monetize a capped, board-approved slice of BTC convexity to fund obligations or reduce dilution, while preserving strategic BTC ownership and adding downside disaster protection.

### 5.2 Bitcoin miners — second wedge

**Examples:** MARA, Riot, CleanSpark, Cipher, TeraWulf, IREN, Core Scientific, Hut 8, Bitfarms, Bitdeer.

**Pain:** Power costs, capex, ASIC purchases, debt/refi needs, BTC-backed loans, production variability.

**Product fit:** Production collars, put spreads, inventory collars, runway protection.

**Hook:**

> Protect cash-flow runway for power, capex, and debt service by collaring a conservative slice of expected production or unencumbered BTC inventory, while leaving most upside intact.

### 5.3 RIAs, family offices, and wealth platforms

**Pain:** BTC drawdown tolerance, client income needs, custody restrictions, ETF workflow preference.

**Product fit:** ETF-option covered calls, collars, defined-loss call spreads, option-income SMA.

**Hook:**

> Add a volatility overlay to BTC ETF exposure without changing custody or introducing crypto exchange operations.

### 5.4 Hedge funds / macro / RV funds

**Pain:** Need normalized cross-venue data, execution proof, margin-adjusted edge.

**Product fit:** RV spread packages, analytics subscription, RFQ sourcing.

**Hook:**

> We normalize ETF, CME, Deribit, and OTC wrappers for basis, margin, collateral, bid/ask, and executable size, then label opportunities by quote/trade confidence.

### 5.5 Structured-product desks / private banks

**Pain:** Need defined BTC payoff products and hedge-source transparency.

**Product fit:** Autocallables, reverse convertibles, buffered notes, principal-protected notes.

**Hook:**

> Provide hedge-source transparency and BTC vol intelligence for defined-outcome BTC products.

---

## 6. Operating model

### Phase 1: Research and analytics platform

- Build daily BTC Vol Desk Monitor.
- Ingest Deribit, IBIT, ETF holdings, OCC, and eventually CME/vendor data.
- Produce screen-only dislocation board.
- Create treasury/miner case studies.
- Maintain evidence appendix.

**Revenue possibility:** Research/analytics/advisory only, if legally structured.

### Phase 2: Advisory and structuring

- Build board-ready treasury and miner hedging policies.
- Produce sample term sheets.
- Advise on permitted instruments, notional limits, collateral process, reporting, and risk governance.
- Partner with regulated execution counterparties.

**Revenue possibility:** Advisory retainer, structuring fee, consulting/project fee.

### Phase 3: RFQ and execution workflow

- Source executable quotes from listed venues, brokers, OTC desks, or market makers.
- Compare quote-verified economics to screen economics.
- Provide best-execution-style analysis where legally permitted.

**Revenue possibility:** Commission, spread, referral/economic sharing where legally allowed.

### Phase 4: Risk intermediation / principal sleeve

- Selectively warehouse/recycle risk across venues.
- Hedge client flow using ETF/CME/Deribit/OTC instruments.
- Operate inside proper fund, dealer, CTA/CPO, broker, or other legal wrapper as determined by counsel.

**Revenue possibility:** Principal spread, carry, management/performance fees.

### Phase 5: Structured products

- Work with issuers/distributors to create BTC-linked defined-outcome products.
- Use analytics engine for hedge sourcing and pricing support.

**Revenue possibility:** Structuring margin, hedging spread, distribution economics.

---

## 7. Economics model

### Potential revenue lines

1. **Advisory/structuring fees**
   - Treasury policy design.
   - Miner hedge design.
   - Board memo and risk framework.

2. **Monitoring/reporting retainer**
   - Daily/weekly hedge marks.
   - Exposure reporting.
   - Risk dashboard.
   - Board/lender reporting package.

3. **Execution economics**
   - Commission or spread through permitted broker/dealer/OTC relationships.
   - RFQ sourcing fee if legally allowed.

4. **Principal/risk spread**
   - Later-stage only.
   - Requires legal wrapper, capital, risk limits, and controls.

5. **Analytics subscription**
   - Hedge funds, allocators, internal desks.
   - Requires production-grade data and history.

6. **Structured-product economics**
   - Phase 2+.
   - Requires issuer/distribution/legal framework.

### Unit economics to model next

For each sample structure:

- notional BTC or ETF share exposure;
- premium generated or paid;
- annualized yield;
- option bid/ask spread;
- hedge cost;
- RFQ spread;
- collateral/margin cost;
- reporting/advisory fee;
- maximum loss/upside cap;
- expected client value vs unhedged BTC;
- desk gross and net revenue.

---

## 8. Legal and regulatory perimeter

This is the main gating issue. The memo must not overclaim.

Potential activities touch multiple regimes:

- securities/options advice;
- commodity/futures/options advice;
- broker-dealer or introducing-broker activity;
- CTA/CPO analysis;
- investment adviser issues;
- OTC derivatives/ISDA counterparty activity;
- offshore exchange access;
- structured-product issuance/distribution;
- custody/collateral/prime brokerage arrangements.

### Correct phased stance

**Start with:**

- research;
- analytics;
- hypothetical/screen-only monitor;
- education;
- internal memo;
- sample term sheets clearly marked illustrative;
- legal analysis.

**Do not start with:**

- quoting derivatives as principal to public companies;
- taking discretion over treasury assets;
- acting as broker-dealer/FCM/IB without structure;
- implying executable RFQ economics before quotes;
- offering securities/commodity advice without counsel.

### Legal workstream deliverables

- regulatory activity map;
- permitted MVP activities;
- required disclaimers;
- entity structure options;
- partner/counterparty model;
- CTA/CPO/RIA/broker-dealer/IB analysis;
- OTC documentation checklist;
- public-company derivatives-policy template.

---

## 9. Risk controls

### Market risk

- BTC delta and gamma.
- Volatility shifts.
- Skew/correlation.
- Calendar mismatch.
- Gap risk and weekend/24-7 risk.

### Basis/wrapper risk

- ETF vs BTC NAV/premium-discount.
- CME futures basis.
- Deribit index/settlement differences.
- ETF options assignment and share settlement.

### Liquidity risk

- Screen liquidity vs executable size.
- Wide short-dated markets.
- Stress-day spread widening.
- Position limits.

### Margin/collateral risk

- USD vs BTC collateral.
- FCM margin calls.
- Crypto exchange margin/liquidation.
- OTC CSA terms.
- Public-company collateral restrictions.

### Counterparty/venue risk

- Exchange/clearing/custody risk.
- OTC counterparty credit.
- Prime broker/FCM dependency.
- Offshore venue restrictions.

### Operational/legal risk

- Inappropriate advice or execution without proper wrapper.
- Board/auditor/lender non-approval.
- Disclosure/regulatory issues.
- Valuation/model governance failure.

### Narrative/reputation risk

- Treasury companies may view hedging as anti-BTC.
- Covered calls can be attacked if BTC rallies through strikes.
- Public companies need careful disclosure and policy framing.

---

## 10. 30-day pilot

### Week 1 — data spine

Deliverables:

- Deribit BTC options ingestion.
- IBIT option-chain ingestion.
- iShares BTC/share parser hardened.
- OCC series/OI cross-check.
- CME data-source decision.
- Daily screen-only BTC Vol Desk Monitor v0.

Success criteria:

- Produce ATM IV term structure for Deribit and IBIT.
- Produce first normalized screen-only dislocation board.
- Every output labeled with source and execution confidence.

### Week 2 — product prototypes

Deliverables:

- Treasury covered-call calculator.
- Zero-cost collar calculator.
- Put-spread collar calculator.
- Miner production hedge template.
- Sample term sheets.
- Board memo template.

Success criteria:

- One hypothetical BTC treasury case study.
- One hypothetical miner case study.
- All prices labeled illustrative/screen-only.

### Week 3 — client mapping and commercial validation

Deliverables:

- 20–50 target accounts.
- Decision-maker/persona map.
- Outreach hooks.
- Qualification checklist.
- Counterparty/RFQ list.

Success criteria:

- At least 3 client archetype packages.
- At least 3 standard RFQ templates.
- Identify legal gating questions before any outreach.

### Week 4 — investor memo and quote verification plan

Deliverables:

- Investor memo v1.
- Evidence appendix.
- Data architecture.
- Legal perimeter memo draft.
- RFQ verification plan.
- 90-day roadmap.

Success criteria:

- Clearly separate thesis, screen evidence, quote evidence, and execution requirements.
- No unsupported economics in investor-facing materials.

---

## 11. 90-day roadmap

### Days 31–45

- Integrate CME via vendor/broker/terminal export or official credentialed path.
- Add OPRA-grade ETF option data if budget allows.
- Store daily surfaces historically.
- Add skew/risk-reversal and calendar analytics.
- Validate IBIT IV model against vendor or broker data.

### Days 46–60

- Collect indicative RFQs for standard structures:
  - 1-month covered call;
  - 3-month zero-cost collar;
  - 6-month miner put spread;
  - ETF vs Deribit vol spread.
- Convert screen-only dislocation board into quote-verified board.
- Build first counterparty matrix.

### Days 61–75

- Build treasury/miner reporting package.
- Draft legal operating model with counsel.
- Define entity/registration/partner structure.
- Create first client-ready board deck.

### Days 76–90

- Run 3–5 live commercial conversations if legal perimeter permits.
- Finalize investor deck.
- Decide whether next vehicle is analytics/advisory company, fund, desk partnership, or hybrid.
- Define capital, hires, vendors, and counterparties required for launch.

---

## 12. Team and infrastructure needs

### Team

- BTC/options structurer.
- Quant/data engineer.
- Derivatives legal counsel.
- Institutional sales / capital markets lead.
- Operations/risk person with FCM/prime/OTC experience.

### Infrastructure

- Market data: Deribit, OPRA/vendor, CME/vendor.
- Data store for option surfaces and historical marks.
- Pricing library and validation process.
- RFQ/counterparty tracker.
- Risk dashboard.
- Compliance archive.
- Board/client report generator.

### Counterparties

- FCM / CME access.
- OPRA/options data vendor.
- ETF/options broker.
- OTC crypto options desks.
- Custodian/prime broker.
- Legal/accounting advisers.

---

## 13. Investment case

### What makes the idea attractive

- BTC volatility is structurally high and institutionally relevant.
- BTC exposure now exists across fragmented wrappers.
- Public BTC treasuries and miners create natural hedging/yield flows.
- ETF/CME/Deribit/OTC access constraints create intermediation value.
- The first product is understandable: treasury income and protection.
- The analytics engine can become a proprietary pricing and risk-recycling layer.

### What must be proven

- Screen dislocations are executable after costs.
- Clients will accept capped upside on a defined tranche.
- Legal structure permits advisory/execution economics.
- Data pipeline can produce reliable normalized surfaces.
- Counterparties will quote size.
- Risk can be hedged across wrappers under stress.

### Main red flags

- Legal/regulatory perimeter may be heavier than expected.
- Pure vol-arb edge may be competed away.
- Public companies may avoid derivatives due to optics.
- OTC/RFQ spreads may consume apparent screen edge.
- Margin/collateral costs may erase relative-value profits.
- Deribit access may be unusable for key US clients.

---

## 14. Recommended investor-facing positioning

Lead with:

> We help institutional BTC holders transform volatility into income, protection, and structured risk through board-ready treasury and miner hedging programs, powered by a proprietary cross-venue BTC volatility monitor.

Avoid leading with:

> We run a BTC options arbitrage fund.

The former is a business. The latter is a trade.

---

## 15. Immediate next deliverables

1. Harden the data pipeline:
   - fix iShares BTC/share parser;
   - store Deribit/IBIT snapshots;
   - compute IV consistently;
   - add bid/ask filters.

2. Build first treasury case study:
   - 10,000 BTC hypothetical holder;
   - 10% covered-call sleeve;
   - 3-month collar;
   - 6-month put-spread collar;
   - board-policy summary.

3. Build first miner case study:
   - 3-month conservative production forecast;
   - 25–50% hedge ratio;
   - put-spread/collar;
   - cash runway impact.

4. Start legal perimeter memo:
   - what can be done as research;
   - what requires registration/partner;
   - what cannot be done before counsel.

5. Start RFQ verification plan:
   - list standard structures;
   - identify counterparties;
   - define quote fields;
   - label quote-verified vs screen-only.

---

## Bottom line

Idea 1 is viable enough to continue, but only if built in the disciplined order:

1. evidence spine;
2. screen-only monitor;
3. treasury/miner product prototypes;
4. legal perimeter;
5. quote verification;
6. investor memo;
7. execution vehicle.

The strongest launch product is not a prop-vol fund. It is a board-ready BTC treasury and miner hedging platform with a proprietary cross-venue volatility engine behind it.
