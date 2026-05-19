# BTC Vol Desk Monitor v0 — Screen-Only Pilot Snapshot

**Run time:** 2026-05-14 18:53 CST  
**Purpose:** First quantified, evidence-labeled snapshot for Idea 1: BTC ETF / Deribit / CME volatility spread + treasury/miner hedging desk.

**Important status:** This is **screen-only**. It is not quote-verified or trade-verified. It uses live/public Deribit data, Nasdaq public IBIT option-chain data, and BlackRock/iShares normalization data. CME and OTC/RFQ are not yet integrated.

---

## 1. Source confidence

### Deribit BTC options
- **Source confidence:** verified public API
- **Execution confidence:** screen-only
- **Endpoint used:** `public/get_book_summary_by_currency?currency=BTC&kind=option`
- **Contracts returned:** ~936 BTC option contracts
- **Expiries observed:** 12
- **Fields used:** underlying price, strike, expiry, call/put, bid, ask, mid, mark IV, open interest, USD volume

### IBIT options
- **Source confidence:** semi-official/public Nasdaq JSON
- **Execution confidence:** screen-only
- **Endpoint used:** `api.nasdaq.com/api/quote/IBIT/option-chain?assetclass=etf&limit=1000`
- **IBIT last:** `$46.17` as of May 14, 2026
- **Expiries observed in pulled chain:** 7 near-dated expiries through May 29, 2026
- **Fields used:** strike, call bid/ask, put bid/ask, volume, open interest
- **IV method:** estimated from mid prices using Black-Scholes with a rough 4.5% risk-free assumption. This is a pilot estimate only; production needs OPRA/vendor IV/Greeks or our own validated model.

### IBIT BTC/share normalization
- **Source confidence:** official BlackRock/iShares holdings source, but the live parser missed the BTC row in this run.
- **Previously verified official values:**
  - BTC holdings: `817,092.857 BTC`
  - shares outstanding: `1,448,200,000`
  - BTC/share: `0.000564212717166`
  - shares per BTC: `~1,772.38`
- **Note:** Need harden parser for the iShares CSV before automating the monitor.

### CME
- **Source confidence:** official documentation verified; live market data not integrated.
- **Execution confidence:** not available in this monitor.
- **Status:** Requires CME credentials/licensing, DataMine, broker/vendor feed, or manual/pro terminal export.

### OTC/RFQ
- **Source confidence:** hypothesis only.
- **Execution confidence:** not available until indicative quotes are collected.

---

## 2. Deribit BTC ATM IV term structure

Screen-only live snapshot:

- 15MAY26, DTE 1: ATM IV ~31.28%, OI ~25,070 BTC, volume ~$2.79M
- 16MAY26, DTE 2: ATM IV ~34.42%, OI ~3,020 BTC, volume ~$0.62M
- 17MAY26, DTE 3: ATM IV ~29.11%, OI ~728 BTC, volume ~$0.16M
- 18MAY26, DTE 4: ATM IV ~30.07%, OI ~1,510 BTC, volume ~$0.63M
- 22MAY26, DTE 8: ATM IV ~34.86%, OI ~14,293 BTC, volume ~$3.10M
- 29MAY26, DTE 15: ATM IV ~36.33%, OI ~79,250 BTC, volume ~$7.78M
- 05JUN26, DTE 22: ATM IV ~37.00%, OI ~382 BTC, volume ~$0.59M
- 26JUN26, DTE 43: ATM IV ~37.98%, OI ~118,296 BTC, volume ~$10.39M
- 31JUL26, DTE 78: ATM IV ~39.12%, OI ~11,931 BTC, volume ~$4.05M
- 25SEP26, DTE 134: ATM IV ~40.79%, OI ~68,290 BTC, volume ~$2.75M
- 25DEC26, DTE 225: ATM IV ~43.71%, OI ~72,231 BTC, volume ~$6.97M
- 26MAR27, DTE 316: ATM IV ~44.85%, OI ~10,581 BTC, volume ~$0.50M

**Read:** Deribit term structure is upward sloping from near-dated ~31–36% into year-end ~44%. The highest OI concentrations in this snapshot were around May 29, Jun 26, Sep 25, and Dec 25.

---

## 3. IBIT ATM IV estimate from listed option mid prices

Screen-only, model-estimated from Nasdaq bid/ask mids:

- May 15, DTE 1: ATM IV estimate ~41.51%
- May 18, DTE 4: ATM IV estimate ~34.08%
- May 20, DTE 6: ATM IV estimate ~35.67%
- May 22, DTE 8: ATM IV estimate ~36.92%
- May 26, DTE 12: ATM IV estimate ~35.51%
- May 27, DTE 13: ATM IV estimate ~36.25%
- May 29, DTE 15: ATM IV estimate ~37.89%

**Read:** IBIT near-dated ATM vol is in the mid/high 30s for most of the first two weeks, with the 1-day expiry estimate higher and noisier. Because these are short-dated options, small bid/ask/model assumptions can move IV materially.

---

## 4. First rough IBIT vs Deribit screen-only comparison

Closest DTE comparisons:

- DTE 1:
  - IBIT May 15 ATM IV estimate: ~41.5%
  - Deribit 15MAY26 ATM IV: ~31.3%
  - Screen spread: IBIT richer by ~10.2 vol points
  - Confidence: low/moderate; 1-day options are noisy and highly sensitive to microstructure.

- DTE 4:
  - IBIT May 18 ATM IV estimate: ~34.1%
  - Deribit 18MAY26 ATM IV: ~30.1%
  - Screen spread: IBIT richer by ~4.0 vol points
  - Confidence: moderate screen-only.

- DTE 8:
  - IBIT May 22 ATM IV estimate: ~36.9%
  - Deribit 22MAY26 ATM IV: ~34.9%
  - Screen spread: IBIT richer by ~2.1 vol points
  - Confidence: moderate screen-only.

- DTE 15:
  - IBIT May 29 ATM IV estimate: ~37.9%
  - Deribit 29MAY26 ATM IV: ~36.3%
  - Screen spread: IBIT richer by ~1.6 vol points
  - Confidence: moderate screen-only.

**Initial read:** The very near-dated IBIT surface appears richer than Deribit in this snapshot, especially at 1-day and 4-day maturities. By 8–15 days, the spread compresses to ~1.5–2 vol points. This could be real wrapper/flow premium, but it could also reflect model differences, ETF-wrapper assumptions, settlement timing, bid/ask, and stale/semi-official data.

---

## 5. What this does and does not prove

### It does prove
- The pilot monitor is feasible using public/semi-public sources.
- Deribit and IBIT can be compared on a rough DTE/ATM basis immediately.
- IBIT short-dated options appear active enough to matter.
- There are visible screen-level vol differences worth investigating.

### It does not prove
- Executable arbitrage.
- Cross-venue net edge after spreads, fees, margin, collateral, and basis.
- Tradable size.
- CME relative value.
- OTC/RFQ economics.
- Legal ability to intermediate client structures.

---

## 6. Immediate quantitative improvements needed

1. **Harden IBIT holdings parser**
   - Reliably parse BlackRock/iShares BTC row and shares outstanding.
   - Store BTC/share daily.

2. **Use OPRA/vendor-grade options data**
   - Nasdaq public data is acceptable for pilot but not production.
   - Need clean timestamps, quote conditions, full option chain, IV/Greeks, and history.

3. **Validate IBIT IV calculation**
   - Compare our Black-Scholes IV estimates against Nasdaq Greeks endpoint, vendor IV, or broker data.
   - Account for ETF fee, NAV premium/discount, BTC/share, and settlement timing.

4. **Add bid/ask execution filter**
   - Minimum OI/volume.
   - Maximum bid/ask width.
   - Exclude weird/stale quotes.

5. **Add Deribit 25-delta skew**
   - Use Deribit order book Greeks selectively or solve delta ourselves.
   - Compare with IBIT 25-delta put/call IV.

6. **Add CME source**
   - Use broker/vendor feed or CME credentialed API.
   - No scraping CME public website.

7. **Add RFQ verification loop**
   - Standard structures:
     - 1-month covered call
     - 3-month collar
     - 6-month miner put spread
   - Label quote-verified once actual counterparties respond.

---

## 7. First screen-only dislocation board

### Candidate 1 — IBIT near-dated richness vs Deribit

- **Type:** wrapper / flow / access dislocation
- **Legs:** short IBIT near-dated ATM vol vs long Deribit same-DTE ATM vol, delta/basis adjusted
- **Observed:** IBIT DTE 1–4 appeared ~4–10 vol points richer than closest Deribit ATM terms.
- **Client application:** covered-call monetization for ETF/BTC treasury holders; potential hedge source for funds.
- **Desk use:** potential call overwrite execution venue or relative-value screen.
- **Confidence:** screen-only.
- **Main risks:** ETF/BTC basis, model assumptions, early/physical settlement mechanics, assignment risk, short-dated gamma, bid/ask/slippage, borrow/financing, inability to legally or operationally pair legs for clients.

### Candidate 2 — Deribit medium-term depth for hedge recycling

- **Type:** liquidity / hedge venue
- **Observed:** Deribit Jun 26 and Dec 25 expiries showed large OI concentrations and meaningful USD volume.
- **Client application:** hedge treasury/miner collar risk or source medium-term BTC vol.
- **Desk use:** potential hedge/recycle venue behind OTC/treasury structures.
- **Confidence:** screen-only.
- **Main risks:** crypto-native collateral/margin, venue/counterparty risk, access constraints for regulated clients, basis under stress.

### Candidate 3 — Treasury covered-call/collar product fit

- **Type:** client-flow opportunity, not pure arb
- **Observed:** IBIT options show active near-dated markets and Deribit provides broader BTC vol term structure.
- **Client application:** BTC treasury yield/protection program.
- **Desk use:** use screen vol to generate indicative term sheets, then verify with RFQ/execution partners.
- **Confidence:** thesis plus screen evidence; not quote-verified.
- **Main risks:** legal perimeter, accounting/board constraints, cap-table/treasury policy, upside opportunity cost, headline risk.

---

## 8. Strategic conclusion from v0

The early data supports the direction but also validates the discipline: this should begin as an analytics + structuring pilot, not a claimed arbitrage machine.

The best first product remains:

> BTC treasury covered-call/collar program, powered by a cross-venue BTC vol monitor.

The monitor gives the desk pricing intelligence; the treasury/miner product gives the business commercial relevance. The spread screen becomes a tool for better pricing and risk recycling, not the entire company.
