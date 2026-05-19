# Idea 1 — BTC Volatility Intermediation Desk Plan

**Channel:** Discord #institutional / Ein Od Milvado

**Goal:** Build an investor-grade institutional concept for a BTC ETF / Deribit / CME volatility spread and BTC treasury/miner hedging desk.

**Core thesis:** The business is not generic crypto options trading. It is institutional BTC risk transformation: treasury companies need yield/protection, miners need cash-flow hedges, funds need relative-value BTC vol exposure, and the desk uses cross-venue analytics plus legal/operational plumbing to intermediate these flows across ETF options, CME, Deribit, and OTC/RFQ channels.

---

## Non-negotiable standards

1. Do not overclaim execution or regulatory permission before verifying legal perimeter.
2. Separate screen dislocations from executable dislocations.
3. Normalize all instruments to BTC-equivalent exposure before comparing prices.
4. Separate analytics product, advisory/structuring product, execution/RFQ product, and principal risk-taking.
5. Build commercial proof before building a complex prop-vol fund.
6. Every investor-facing claim needs source/evidence or must be labeled as a hypothesis.

---

## Workstreams

### 1. Product/menu definition

Status: first pass complete.

Products:
- Treasury covered call programs
- Treasury collars/downside protection
- Miner production/inventory hedges
- ETF / Deribit / CME vol-spread packages
- RFQ/execution desk
- Structured notes/yield products, phase 2

Initial priority:
1. Treasury collars / covered calls
2. Miner hedge programs
3. Cross-venue vol analytics
4. RFQ/execution workflow
5. Structured notes later

### 2. Data map

Status: in progress.

Verified public data so far:
- Deribit BTC options API works and returns live BTC option book summaries.
- Nasdaq IBIT option-chain endpoint works and returns IBIT option chain records.
- CME public page blocked automated fetch with HTTP 403; need alternate verified CME data source.

Required data domains:
- BTC spot/reference
- IBIT price and option chain
- ETF NAV/BTC-per-share mapping
- Deribit BTC options surface
- Deribit futures/perps/basis
- CME BTC futures curve
- CME BTC options chain
- OTC/RFQ indicative quotes
- margin/collateral assumptions by venue
- fees/slippage/execution-size assumptions

### 3. Opportunity quantification

To build:
- ATM IV comparison by expiry
- 25-delta put/call IV comparison
- risk reversal comparison
- calendar slope comparison
- futures basis and wrapper-adjusted forward comparison
- bid/ask and executable-size filter
- margin/collateral-adjusted return model
- top dislocation report with confidence labels

Dislocation labels:
- wrapper
- skew
- calendar
- basis
- margin/collateral
- flow

### 4. Client mapping

Client archetypes:
- BTC treasury companies
- public/private BTC miners
- crypto hedge funds
- family offices / RIAs
- structured-product distributors
- market makers / OTC desks / prime brokers

Need to map each by:
- pain point
- product fit
- likely constraints
- decision maker
- urgency trigger
- first conversation hook

### 5. Investor-grade deep-dive

Final memo sections:
- executive thesis
- market structure and why now
- venue/data map
- product menu
- natural client flows
- economics/revenue model
- operating model
- legal perimeter and phased approach
- risk controls
- 30-day pilot
- 90-day roadmap
- evidence appendix

---

## 30-day pilot architecture

Week 1: Evidence and data spine
- Deribit ingestion
- IBIT ingestion
- CME data source discovery
- normalize option expiries/strikes
- build first daily IV/skew/calendar report

Week 2: Client product prototypes
- treasury covered call calculator
- zero-cost collar calculator
- miner hedge ladder
- sample term sheets
- board memo template

Week 3: Client list and commercial proof
- map BTC treasury companies
- map public miners
- identify 20–50 target accounts
- draft outreach angles
- test product-market fit through paper case studies/RFQs

Week 4: Investor memo
- data-backed opportunity examples
- sample economics
- legal operating perimeter
- risk controls
- first hires/vendors/counterparties
- capital requirements and milestones

---

## Next task

Build the institutional-grade data map and evidence appendix:
1. Verify official/public sources for Deribit, IBIT/Nasdaq/OCC, and CME.
2. Define exact fields required from each venue.
3. Identify source gaps and paid/professional data options.
4. Produce normalized BTC-equivalent comparison schema.
5. Produce first draft of the data architecture and daily report format.
