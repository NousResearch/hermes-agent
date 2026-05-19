# BTC Treasury & Miner Hedging Desk Investor Memo

**Status:** SCREEN-ONLY · NOT EXECUTABLE  
**Positioning:** BTC Treasury & Miner Hedging Desk powered by a purpose-built cross-venue volatility evidence engine

> Internal evidence prototype. Public screen/model data only. No RFQ sent. No executable quote.

> Not a client portal. Not an execution venue. Not a fund offering. This memo summarizes an internal proof-of-concept and evidence workflow.

## Thesis

BTC volatility markets are fragmented across ETF-listed options, offshore crypto options venues, CME-linked institutional channels, and OTC counterparties. Corporate BTC treasuries and miners need hedge design, evidence integrity, and quote-verification discipline more than another generic crypto trading screen.

The opportunity is a desk/platform: use a purpose-built cross-venue volatility evidence engine to identify screen-only dislocations, convert them into treasury/miner hedge structures, and promote only independently documented opportunities through a manual quote-verification workflow.

## Evidence Snapshot

- Run ID: `btcvol-20260518-163034`
- As-of CST: `2026-05-18 16:30:34 CDT`
- BTC reference: `$77,457.06`
- IBIT BTC/share: `0.0005679118586151414`
- Configured-source quality: `GREEN` / `80`
- Current screen-source availability: `Current screen/vendor captures available: Deribit + IBIT + CME Databento`
- Overall evidence readiness: `YELLOW until real quote evidence exists`
- Freshness: `GREEN`
- Screen-only dislocations: `6`
- Quote-review flags: `3`
- Bundle SHA-256: `6504204d3adf8ecd1d5bac94775f2639e08baec722a06f41623346dd7724210b`

All displayed economics are public screen marks or model-estimated values. They are evidence inputs, not executable quotes.

## Screen-Only Candidate Examples

- #1 IBIT 0D ATM vs Deribit 1D ATM — -16.49 vol pts; priority `high`; SCREEN-ONLY · NOT EXECUTABLE.
- #2 IBIT 2D ATM vs Deribit 2D ATM — +8.39 vol pts; priority `high`; SCREEN-ONLY · NOT EXECUTABLE.
- #3 IBIT 4D ATM vs Deribit 4D ATM — +5.46 vol pts; priority `high`; SCREEN-ONLY · NOT EXECUTABLE.
- #4 IBIT 8D ATM vs Deribit 11D ATM — -1.51 vol pts; priority `low`; SCREEN-ONLY · NOT EXECUTABLE.
- #5 IBIT 11D ATM vs Deribit 11D ATM — +1.27 vol pts; priority `low`; SCREEN-ONLY · NOT EXECUTABLE.

## Treasury Case Study

- Structure: corporate BTC treasury hedge policy sleeve.
- Hedged BTC: 350 BTC hedged.
- Caveat: hypothetical scenario only; no suitability, premium, margin, tax, accounting, liquidity, or counterparty commitment is implied.
- Illustrative model output — protected value: $20,332,477.31 illustrative floor-protected sleeve.
- Illustrative model output — floor / cap: $58,092.79 / $96,821.32.
- Control: Premium and executable levels require two-counterparty quote verification.

## Miner Case Study

- Structure: runway-protection hedge for conservative monthly production.
- Hedged monthly production: 60 BTC/month hedged.
- Caveat: hypothetical scenario only; no hedge recommendation, suitability review, margin model, tax/accounting treatment, or executable quote is implied.
- Illustrative model output — monthly floor revenue: $3,253,196.37.
- Pre-hedge cash runway: 3.75 months.
- Control: Indicative economics require quote verification before investor/client use.

## Quote Verification Workflow

Manual demo workflow only; no RFQ is sent and no executable quote is implied.

Current stage counts:

- Screen-only: 5
- Reviewed: 0
- Internal RFQ draft: 0
- Indicative quote 1: 0
- Indicative quote 2: 0
- Quote verified: 0
- Post-trade record verified: 0

Workflow gates:

1. Screen-only dislocation detected by the monitor.
2. Candidate triaged for materiality and structure fit.
3. RFQ package generated for review, not auto-sent.
4. After approval and manual outreach, counterparties may provide indicative quotes.
5. Evidence ledger may mark candidate quote-verified after required indicative quote records are captured.
6. Post-trade verification only occurs after an actual external execution record exists; this memo does not imply execution.

## Business Model Sequence

1. Research/evidence engine
2. Treasury and miner hedge structuring
3. Partner-led RFQ and execution support
4. Principal risk sleeve only after chosen legal wrapper

The recommended route remains desk/platform-first. The fund sleeve is an expansion path after evidence, legal wrapper, counterparty access, and investor demand are established.

## Investor Ask

- Seed build capital for data licensing, counterparty connectivity, compliance review, and proof-of-market pilots.
- Introductions to BTC treasury decision-makers, miners, OTC desks, derivatives counsel, and execution partners.
- Approval to harden the evidence engine into a controlled diligence workflow without launching a client portal or executable RFQ system prematurely.

## Evidence Room

- Report: `artifacts/institutional/data/reports/btc-vol-desk-monitor-2026-05-18-163034.md`
- Evidence bundle: `artifacts/institutional/data/normalized/btcvol-20260518-163034/btcvol-20260518-163034-evidence-bundle.zip`
- Evidence manifest: `artifacts/institutional/data/normalized/btcvol-20260518-163034/evidence_manifest.json`
- Candidate ledger: `artifacts/institutional/data/normalized/btcvol-20260518-163034/candidate_triage.jsonl`
- Bundle SHA-256: `6504204d3adf8ecd1d5bac94775f2639e08baec722a06f41623346dd7724210b`

## Limitations / Gating Items

- SCREEN-ONLY · NOT EXECUTABLE: all current market economics are public screen marks, model-estimated IVs, or internal review outputs.
- CME remains unavailable until licensed/vendor/broker feed is configured.
- Quote-verified economics require two independently captured indicative quote records.
- Trade/post-trade verification requires an actual external execution record; this POC has no trading capability.
- Screen-only dislocation differences are gross IV observations only; they exclude bid/ask width, slippage, financing, borrow, margin, collateral, tax/accounting, settlement, exercise style, venue hours, operational constraints, and counterparty risk.
- Tenor matching between IBIT ETF options and Deribit BTC options is approximate and not an arbitrage claim or trading recommendation.
- Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow.
