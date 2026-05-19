# BTC Volatility Hedging Desk — Investor POC Site Next Spec v1

**Date:** 2026-05-15 CDT  
**Owner:** Joseph / internal build team  
**Status:** Build next  
**Control language:** Internal proof-of-concept. Screen-only market data. Not an offer. Not executable.

---

## 1. Clarification: What Was Meant by “Blocked”

The prior wording was too loose. There is no outside person magically blocking Joseph from building. The distinction is between **build authority** and **regulated/commercial authority**.

### What Joseph can build now

Joseph can build all of this now:

- Investor proof-of-concept site.
- Internal BTC volatility monitor.
- Live public-data evidence dashboard.
- Fund/deal memo.
- Treasury/miner hedge case studies.
- RFQ templates.
- Manual quote-evidence ledger.
- Counsel/investor diligence packet.
- Demo workflows that show what would happen after quote verification.

### What should not be represented as live/executable yet

The system should not imply any of the following until Joseph deliberately sets up the required wrapper:

- That displayed IBIT/Deribit economics are executable.
- That the site is a client portal.
- That the desk is already operating as an investment adviser, broker, dealer, CTA/CPO, fund manager, or execution venue.
- That RFQs have been sent to counterparties unless they actually have.
- That screen prices are tradeable quotes.
- That investor capital can be accepted through the site.

### “Legal/counsel approval” means approval from Joseph’s chosen structure

That means one or more of:

- Joseph personally, if this remains internal/research-only.
- Joseph’s fund/entity counsel, if investor materials are being circulated.
- Compliance counsel, if advisory/execution/fund language is used.
- Fund administrator / broker / execution partner, if the workflow touches real trading or client accounts.
- The investor’s counsel/team, if they request diligence before seed/check/legal setup.

So the better phrasing is:

> “Execution, investor onboarding, and client-facing RFQ workflows are gated by Joseph’s chosen business/legal wrapper — not by the software. The software can be built now, but the claims and actions it enables must match the structure Joseph chooses.”

---

## 2. Recommended Route

Build a **polished investor proof-of-concept site** that demonstrates the thesis and proves the data engine works, while keeping all market economics screen-only.

This should not be a generic “crypto dashboard.” It should read like an institutional derivatives/hedging desk platform.

### Positioning

**BTC Treasury & Miner Hedging Desk powered by a proprietary cross-venue volatility evidence engine.**

The site proves three things:

1. There is a real structural market opportunity across BTC volatility venues.
2. The team can observe, normalize, and evidence the opportunity better than a generic fund deck.
3. There is a credible path from research dashboard → quote verification → treasury/miner hedging mandates → fund or desk economics.

---

## 3. The Site We Should Build Next

### Site name options

- **VolDesk BTC**
- **Satoshi Volatility Desk**
- **BTC Treasury Vol Desk**
- **Digital Asset Hedging Desk**
- **CrossVenue BTC Vol Monitor**

Working title for now:

> **BTC Vol Desk — Institutional Hedging & Volatility Evidence Platform**

### Audience

Primary:

- Investor and investor’s team.
- Potential seed/strategic partner.
- Counsel reviewing the business model.

Secondary later:

- Corporate BTC treasury CFOs.
- Bitcoin miners.
- OTC desks/market makers.
- Fund allocator due-diligence teams.

---

## 4. Site Architecture

Build as a static-but-polished front-end first. Use generated JSON artifacts from the monitor. Do not build accounts, auth, payments, order entry, or live RFQ execution yet.

### Recommended stack

- Next.js or Vite/React for polished investor demo.
- Static export/deployable site.
- Local JSON artifact ingestion from `artifacts/institutional/data`.
- No database required for this phase.
- No backend required unless we want scheduled refresh later.

### Pages / Sections

#### 1. Landing / Executive Summary

Purpose: Investor can understand the whole thesis in 90 seconds.

Content:

- What we are building.
- Why BTC treasury/miner hedging is underbuilt.
- Why cross-venue volatility fragmentation matters.
- What the current evidence engine already proves.
- Clear disclaimer: research / screen-only / not executable.

Key visual:

- “From fragmented BTC volatility → hedging intelligence → quote-verified execution workflow.”

#### 2. Live Evidence Dashboard

Purpose: Show that this is not just a deck.

Content:

- Latest BTC reference.
- IBIT BTC/share conversion.
- Deribit ATM IV curve.
- IBIT ATM IV curve.
- IBIT vs Deribit dislocation board.
- Candidate triage.
- Source freshness.
- Data quality.
- Evidence bundle hash.

Current static dashboard already does much of this; next step is to make it investor-grade visually and narratively.

#### 3. Thesis Explainer

Purpose: Explain why the opportunity exists.

Modules:

- BTC volatility venues are fragmented: ETF options, Deribit, CME, OTC.
- ETF wrapper introduces different investor base and constraints.
- Corporate treasury/miner hedging needs are practical, not speculative.
- Screen dislocations are not enough; the edge is quote verification + structuring.

#### 4. Treasury Hedging Case Study

Purpose: Show use case for corporate BTC holders.

Example modules:

- Covered call income program.
- Downside put-spread collar.
- Board-approved hedge policy.
- Risk limits and mark-to-market effects.
- Output: policy memo + RFQ template + scenario table.

#### 5. Miner Hedging Case Study

Purpose: Show use case for miners with revenue/collateral constraints.

Example modules:

- Production runway protection.
- 90D floor/collar structures.
- Avoiding overhedging.
- Collateral/liquidity stress.
- Hashprice and BTC price sensitivity later.

#### 6. Quote Verification Workflow

Purpose: Show the bridge from screen-only research to institutional evidence.

Flow:

1. Screen dislocation detected.
2. Candidate triaged.
3. RFQ package generated.
4. Two counterparties quote same economics.
5. Quote evidence recorded.
6. Candidate promoted from screen-only → quote-verified.
7. Only then can it appear in investor evidence.

Important: This is a demo workflow until actual counterparties are contacted.

#### 7. Fund / Desk Business Model

Purpose: Show how this becomes a real business.

Model sequence:

1. Research/evidence engine.
2. Advisory/structuring for treasuries/miners.
3. Partner-led RFQ/execution support.
4. Principal risk recycling / relative-value fund only after legal wrapper and execution rails.
5. Structured product opportunities later.

Revenue lines:

- Research/analytics subscription.
- Advisory/structuring fees.
- Execution/RFQ support fees through partner wrapper.
- Fund management/performance fees if launched.
- Hedge program retainers for miners/treasuries.

#### 8. Evidence Room

Purpose: Give investor team diligence artifacts.

Content:

- Latest report.
- Evidence manifest.
- Bundle SHA-256.
- Run history.
- Completion readout.
- Data-source appendix.
- Legal perimeter memo.
- RFQ template board.

Download links can point to generated static files.

---

## 5. What To Build Next — Exact Spec

### Phase A — Investor Narrative Site Shell

Build a polished single-page site or multi-section landing page.

Required sections:

1. Hero / thesis.
2. Problem.
3. Solution.
4. Live evidence snapshot.
5. Treasury case study.
6. Miner case study.
7. Quote verification workflow.
8. Fund/desk business model.
9. Evidence room.
10. Next steps / diligence ask.

Acceptance criteria:

- Looks institutional, not crypto-retail.
- Uses deep charcoal / navy / off-white / Bitcoin gold / ice blue.
- Every market/economics display says screen-only or not executable.
- Investor can understand the business without reading the raw reports.
- Site can be opened locally and screenshotted for presentation.

### Phase B — Data Adapter From Current Monitor Artifacts

Create a small adapter that reads latest run artifacts:

- `run_manifest.jsonl`
- latest report path
- latest evidence bundle path
- latest candidate triage ledger
- latest ATM CSVs
- evidence manifest JSON

Output a clean `site-data.json` for the front-end.

Acceptance criteria:

- Front-end never parses raw messy files directly.
- Missing artifacts degrade gracefully.
- All data includes `evidence_status` and `source_confidence` where applicable.

### Phase C — Investor-Grade Live Evidence Components

Replace the current raw dashboard feel with components:

- KPI cards.
- Vol curve comparison chart.
- Dislocation candidate cards.
- Source freshness/status panel.
- Evidence bundle verification badge.
- Run-history trend strip.
- “What this proves / what it does not prove” sidebar.

Acceptance criteria:

- A non-technical investor can understand why a candidate matters.
- A technical diligence reviewer can see source, freshness, and hash proof.
- No language implies trade execution.

### Phase D — Case Study Calculators

Add two calculator/demo modules:

#### Treasury BTC Covered Call / Collar Demo

Inputs:

- BTC held.
- Hedge ratio.
- Tenor.
- OTM call strike.
- Put floor.
- BTC spot.

Outputs:

- BTC-equivalent notional.
- Premium estimate placeholder.
- Scenario payoff table.
- Board-policy language.
- RFQ package preview.

#### Miner Runway Protection Demo

Inputs:

- Monthly BTC production.
- Cash runway months.
- Hedge ratio.
- Floor price.
- Cap price.
- Tenor.

Outputs:

- Protected BTC amount.
- Downside floor scenario.
- Upside give-up scenario.
- Liquidity/collateral warning.
- RFQ package preview.

Acceptance criteria:

- Uses conservative defaults.
- Avoids pretending premiums are executable unless quote evidence exists.
- Can produce a case-study PDF/markdown export later.

### Phase E — Quote Evidence Demo Workflow

Build a non-executing workflow screen:

- Candidate selected.
- RFQ template preview.
- Counterparty quote placeholders.
- Evidence status transition:
  - screen-only
  - quote-requested
  - quote-verified
  - trade-verified
- Required fields before promotion.

Acceptance criteria:

- Cannot mark firm/executable without required fields.
- Can show investor how the process becomes real.
- Clearly marked demo/manual workflow.

### Phase F — Investor Packet Export

Generate a clean investor packet as markdown/PDF later:

- Executive memo.
- Site screenshots.
- Latest evidence dashboard.
- Case studies.
- Evidence manifest / hash.
- Business model.
- Risk/legal perimeter.
- Roadmap.

Acceptance criteria:

- One file Joseph can send before a meeting.
- No raw credentials/secrets.
- Screen-only language preserved.

---

## 6. What Not To Build Yet

Do not build these yet:

- Client login.
- Payments/subscriptions.
- Live RFQ sending.
- Trading/order entry.
- Broker/dealer workflow.
- Investor onboarding/KYC.
- Multi-user admin console.
- Production database unless static artifacts become limiting.

These would distract from the immediate goal: investor-grade proof of concept.

---

## 7. Investor Meeting Deliverable

The target deliverable should be:

1. A polished site/demo URL or local browser artifact.
2. A 5–8 page investor memo.
3. A current evidence bundle with SHA-256 hash.
4. A dashboard screenshot.
5. Two case studies: treasury and miner.
6. A clear ask:
   - fund seed;
   - strategic build capital;
   - broker/OTC partnership;
   - pilot treasury/miner mandate;
   - or legal structuring support.

---

## 8. Recommended Immediate Build Order

1. Build the investor POC site shell.
2. Add artifact-to-site-data adapter.
3. Replace raw dashboard with investor-grade components.
4. Add treasury case study calculator.
5. Add miner runway calculator.
6. Add quote verification demo workflow.
7. Generate investor packet export.
8. Run visual QA and produce screenshots.

This route preserves what is already built and upgrades it into a credible presentation asset.

---

## 9. Decision Needed From Joseph

The only strategic decision needed before implementation is positioning:

### Option A — Fund-first

Lead with “BTC volatility relative-value fund.”

Pros:
- Clear investor monetization.
- Direct fund conversation.

Cons:
- More legal/compliance burden.
- Looks more like prop trading.
- Harder to prove client pain.

### Option B — Desk/platform-first

Lead with “Treasury/miner hedging desk powered by a proprietary volatility evidence engine.”

Pros:
- More institutional.
- Easier to show client need.
- Better path to strategic investors/partners.
- Fund can be a later capital sleeve.

Cons:
- Slightly more complex story.

### Recommendation

Use **Option B: Desk/platform-first**, with fund economics as a later capital sleeve. This is more credible and better aligned with the current evidence engine.
