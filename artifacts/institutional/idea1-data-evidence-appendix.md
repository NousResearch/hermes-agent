# Idea 1 — BTC Vol Desk Data & Evidence Appendix

**Purpose:** Establish the evidence spine for a BTC ETF / Deribit / CME / OTC volatility intermediation desk before writing investor narrative.

**Standard:** Every data source is classified as verified, credentialed/paid, or hypothesis. Screen-only data is not treated as executable.

---

## 1. Source classification

### A. Deribit BTC options — verified public API

**Status:** Verified public API access.

**Official docs / endpoints:**
- OpenAPI spec: `https://docs.deribit.com/specifications/deribit_openapi.json`
- Instruments: `GET https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false`
- Book summary: `GET https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option`
- Order book: `GET https://www.deribit.com/api/v2/public/get_order_book?instrument_name=BTC-...&depth=1`
- Historical volatility: `GET https://www.deribit.com/api/v2/public/get_historical_volatility?currency=BTC`

**Live verification from May 14, 2026 pass:**
- Active BTC option instruments: ~936
- Expiries observed: 12
- Near-ATM IV samples:
  - 15MAY26: ~31.1%
  - 22MAY26: ~35.1%
  - 29MAY26: ~36.5%
  - 26JUN26: ~38.1%
  - 25SEP26: ~40.8%

**Useful fields:**
- Metadata: `instrument_name`, `instrument_id`, `expiration_timestamp`, `strike`, `option_type`, `state`, `is_active`, `contract_size`, `settlement_currency`, `quote_currency`, `price_index`
- Market snapshot: `bid_price`, `ask_price`, `mid_price`, `mark_price`, `mark_iv`, `underlying_price`, `underlying_index`, `interest_rate`, `open_interest`, `volume`, `volume_usd`, `estimated_delivery_price`
- Top-of-book: `best_bid_price`, `best_bid_amount`, `best_ask_price`, `best_ask_amount`, `bid_iv`, `ask_iv`, `mark_iv`, `greeks.delta`, `greeks.gamma`, `greeks.vega`, `greeks.theta`, `greeks.rho`
- Historical vol: live payload returns `[timestamp, value]` pairs despite docs schema implying `timestamp` / `value` objects.

**Caveats:**
- Deribit prices are crypto-native and must be normalized into USD/BTC-equivalent terms.
- Use WebSocket subscriptions for real-time production instead of polling every instrument.
- Rate limits are credit-based; `get_instruments` has stricter limits than normal endpoints.

---

### B. IBIT / listed ETF options — verified semi-official + official supporting sources

#### Nasdaq option chain

**Status:** Verified public JSON, but undocumented/semi-official rather than guaranteed production API.

**Endpoint:**
- `https://api.nasdaq.com/api/quote/IBIT/option-chain?assetclass=etf`
- Greeks endpoint: `https://api.nasdaq.com/api/quote/IBIT/option-chain/greeks?assetclass=etf`

**Live verification from May 14, 2026 pass:**
- IBIT last shown: `$46.17 as of May 14, 2026`
- `totalRecord`: 412
- Expiries returned included May 15, May 18, May 20, May 22, May 26, May 27, May 29.
- Retrieved chain sample showed:
  - call OI: ~1.12M contracts
  - put OI: ~640k contracts
  - call volume: ~336k
  - put volume: ~109k

**Useful fields:**
- Chain fields: `expiryDate`, `strike`, `c_Last`, `c_Bid`, `c_Ask`, `c_Volume`, `c_Openinterest`, `p_Last`, `p_Bid`, `p_Ask`, `p_Volume`, `p_Openinterest`
- Greeks endpoint: `cDelta`, `cGamma`, `cRho`, `cTheta`, `cVega`, `cIV`, `pDelta`, `pGamma`, `pRho`, `pTheta`, `pVega`, `pIV`

**Caveats:**
- Not documented as a stable API contract.
- `table.asOf` observed as null.
- Display-oriented records, not OPRA-grade records.
- No guaranteed history, timestamps, quote conditions, NBBO provenance, or depth.
- Greeks methodology/source not specified.

#### OCC series search

**Status:** Official public source for listed series and open interest; not a quote/IV source.

**Endpoint tested:**
- `https://marketdata.theocc.com/series-search?symbolType=U&symbol=IBIT`

**Useful fields:**
- Product symbol
- Expiration year/month/day
- Strike
- Call/put indicator
- Call OI
- Put OI
- Position limit
- Exchanges where product trades

**Caveats:**
- No bid/ask/last/volume/IV/Greeks.
- Useful as official series/OI reference, not as tradeable market data.

---

### C. IBIT NAV / BTC-per-share normalization — verified official BlackRock/iShares source

**Status:** Verified official public iShares data.

**Sources:**
- Product page: `https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf`
- Holdings CSV: `https://www.ishares.com/us/products/333011/fund/1467271812596.ajax?fileType=csv&fileName=IBIT_holdings&dataType=fund`

**Observed official fields from May 13/14 pass:**
- NAV: `$45.05` as of May 13, 2026
- Sponsor fee: `0.25%`
- Net assets: `$65,235,245,502`
- Shares outstanding: `1,448,200,000`
- BTC quantity in holdings CSV: `817,092.85700 BTC`
- Benchmark index: CME CF Bitcoin Reference Rate - New York Variant
- Bloomberg index ticker: BRRNY

**Normalization calculation:**
- BTC/share = `BTC quantity / shares outstanding`
- Using observed values: `817,092.857 / 1,448,200,000 = 0.000564212717166 BTC/share`
- Shares per BTC ≈ `1,772.38117748`

**Use in vol monitor:**
- Convert IBIT option strikes to BTC-equivalent strikes via BTC/share.
- Convert IBIT option premiums to BTC-equivalent or USD-per-BTC premium.
- Compare IV/Greeks only after wrapper and forward assumptions are normalized.

**Caveats:**
- Public holdings/NAV data is generally as-of/T-1, not real-time.
- Holdings CSV warns market value/weight/notional may use third-party vendor pricing.
- Use official quantity and shares outstanding for BTC/share; avoid inferring from market value.

---

### D. CME BTC futures/options — verified official docs, credentialed for production data

**Status:** Official public web pages are reference-only; automated production data requires credentials/licensing/API/vendor.

**Public product pages:**
- Bitcoin futures: `https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.html`
- Bitcoin futures quotes: `https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.quotes.html`
- Bitcoin futures specs: `https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.contractSpecs.html`
- Bitcoin options: `https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.quotes.options.html`
- Bitcoin options specs: `https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.contractSpecs.options.html`

**Automation caveat:**
- CME public website returned HTTP 403 to scripted access and explicitly blocks scraping under its website data terms.
- Treat public pages as human reference/display only.

**Official CME API/feed paths:**
- Real-Time Futures and Options WebSocket API docs: `https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457414253/CME+Market+Data+Over+WebSocket+API`
- Message spec: `https://cmegroupclientsite.atlassian.net/wiki/display/EPICSANDBOX/CME+Market+Data+Over+WebSocket+API+-+Message+Specification`
- Production WebSocket: `wss://markets.api.cmegroup.com/marketdatastream/v1`
- OAuth token production: `https://auth.cmegroup.com/as/token.oauth2`
- MDP 3.0 market data docs: `https://cmegroupclientsite.atlassian.net/wiki/display/EPICSANDBOX/CME+MDP+3.0+Market+Data`
- DataMine: `https://www.cmegroup.com/datamine.html`
- DataMine API: `https://www.cmegroup.com/datamine/datamine-api.html`

**Useful official data classes:**
- Top of book: bid/ask price, quantity, order count, timestamps, instrument metadata.
- Trade summary: trade price/quantity, aggressor side, timestamps, instrument metadata.
- Statistics: cleared volume, open interest, settlement price, settlement flags/date/timestamp.
- Historical DataMine: end-of-market summary, BBO, volume/OI.
- CME options analytics product appears to include Greeks and implied volatility, but requires credentials/entitlements.

**Practical path:**
- MVP: use broker/vendor data for CME BTC futures/options if available.
- Institutional: CME WebSocket API + DataMine historical after entitlements.
- Low-latency institutional: MDP 3.0 direct feed only if justified.

---

### E. OTC / RFQ layer — not yet verified; must be quote-verified

**Status:** Hypothesis until live counterparties provide indicative quotes or executed RFQs.

**Target sources:**
- Crypto OTC desks
- Options market makers
- Prime brokers
- FCMs / institutional crypto brokers
- RFQ venues
- Structured-product desks

**Required fields:**
- Indicative bid/ask for standard structures
- Minimum ticket size
- Max executable size
- Counterparty credit terms
- Collateral requirements
- Settlement method
- Legal documents required
- Quote expiry
- Venue hedge assumptions
- Fees/commission/spread

**Confidence labels:**
- Hypothesis: based on expected market behavior only.
- Screen-only: based on listed market data only.
- Quote-verified: at least one real indicative quote from a counterparty.
- Trade-verified: actual executed trade or executable firm market.

---

## 2. Normalized BTC-equivalent schema

Every option record should be transformed into a common schema before comparison.

### Required normalized fields

- `source`: Deribit / Nasdaq / OCC / CME / OTC / vendor
- `venue`: Deribit / ETF-listed / CME / OTC
- `instrument_id`
- `instrument_name`
- `underlying_wrapper`: BTC index / IBIT ETF / CME futures / OTC reference
- `underlying_price_native`
- `btc_reference_price`
- `expiry_date`
- `days_to_expiry`
- `strike_native`
- `strike_btc_equivalent`
- `option_type`: call / put
- `bid_native`
- `ask_native`
- `mid_native`
- `mark_native`
- `bid_usd_equivalent`
- `ask_usd_equivalent`
- `mid_usd_equivalent`
- `bid_btc_equivalent`
- `ask_btc_equivalent`
- `iv_bid`
- `iv_ask`
- `iv_mid`
- `iv_mark`
- `delta`
- `gamma`
- `vega`
- `theta`
- `rho`
- `open_interest_native`
- `volume_native`
- `bid_size`
- `ask_size`
- `bid_ask_width_pct`
- `forward_price_assumption`
- `rate_assumption`
- `funding_or_basis_assumption`
- `margin_model`
- `collateral_currency`
- `fee_assumption`
- `slippage_assumption`
- `data_timestamp`
- `source_confidence`: verified_public / semi_official / credentialed / paid_vendor / hypothesis
- `execution_confidence`: screen_only / quote_verified / trade_verified

### Wrapper-specific normalization

**Deribit:**
- Native quote is often in BTC terms for options.
- Use Deribit `underlying_price` / `estimated_delivery_price` to convert to USD-equivalent.
- Confirm exact option contract settlement and PnL convention per instrument metadata.

**IBIT:**
- Compute BTC/share from official iShares holdings.
- Convert IBIT strike to BTC-equivalent: `strike_native / BTC_per_share` only after checking whether comparing to BTC-per-one-BTC notional or per-share notional.
- Convert option premium to BTC-equivalent by scaling share option contract multiplier and BTC/share.
- Account for ETF fee, NAV premium/discount, and T-1 holdings data.

**CME:**
- Normalize from futures option to BTC spot-equivalent using futures price/basis.
- Include contract multiplier and whether using standard or micro BTC contract.
- Use settlement/open-interest stats from official CME feed/vendor.

**OTC/RFQ:**
- Normalize quote terms manually from term sheet.
- Include collateral and credit terms; these may dominate apparent vol difference.

---

## 3. Daily BTC Vol Desk Monitor — MVP format

### Section 1: Market state
- BTC spot/reference
- IBIT last / NAV / premium-discount
- IBIT BTC/share
- CME front futures and basis
- Deribit futures/perps basis
- realized/historical vol reference

### Section 2: Venue surface summary
- Deribit ATM IV by expiry
- IBIT ATM IV by expiry
- CME ATM IV by expiry
- 25-delta put IV by venue
- 25-delta call IV by venue
- risk reversal by venue
- term-structure slope by venue
- volume/OI by venue
- bid/ask quality by venue

### Section 3: Dislocation board
For each candidate:
- Dislocation type: wrapper / skew / calendar / basis / margin / flow
- Instruments/legs
- Native prices
- BTC-equivalent normalized prices
- Gross edge
- Estimated costs: bid/ask, fees, margin, collateral, slippage
- Net edge
- Capacity estimate
- Execution confidence: screen-only / quote-verified / trade-verified
- Client application: treasury / miner / fund / family office / structured product
- Risk notes

### Section 4: Client structure examples
- Treasury covered call quote grid
- Zero-cost collar grid
- Miner hedge ladder
- Fund relative-value spread package

### Section 5: Missing data / warnings
- Source gaps
- stale fields
- wide markets
- unverified quote assumptions
- regulatory/venue constraints

---

## 4. Data gaps and vendor needs

### Immediate gaps
- CME credentialed/production data source.
- OPRA-grade IBIT quote/trade history and live NBBO.
- Historical IBIT option IV/Greeks.
- Historical Deribit full surface data unless stored going forward or sourced from vendor.
- OTC/RFQ quote verification.
- Legal/compliance mapping for advisory/execution/principal activity.

### Candidate vendors / infrastructure
- CME official WebSocket + DataMine.
- OPRA-licensed feeds.
- Databento for OPRA and possibly CME data depending license/product coverage.
- Cboe DataShop / LiveVol.
- OptionMetrics IvyDB US.
- ORATS.
- Polygon/Massive options APIs.
- dxFeed / Interactive Brokers / Bloomberg / Refinitiv / FactSet.
- Crypto options vendors for Deribit history/analytics if cheaper than self-storing.

---

## 5. Build sequence recommendation

### Step 1: Deribit + IBIT daily snapshot
Build the first monitor using verified public/semi-official data:
- Deribit book summary + instruments.
- Nasdaq IBIT chain + Greeks.
- BlackRock iShares holdings CSV for BTC/share.
- OCC series search for official listed-series/OI cross-check.

### Step 2: CME data-source decision
Choose one:
- Broker/vendor API for quick pilot.
- CME WebSocket/DataMine if institutional path and budget justify.
- Bloomberg/Refinitiv/manual export for investor memo proof only.

### Step 3: Normalize and label confidence
Every result must show:
- source confidence
- execution confidence
- stale/as-of timestamp
- cost/margin assumptions

### Step 4: Produce first daily report
Output should be suitable for two audiences:
- internal trader/structurer: exact instruments and dislocations
- investor/client: plain-English structures and why they exist

---

## 6. Current conclusion

The data spine is feasible, but the honest MVP must be labeled correctly:

- Deribit: usable immediately from official public API.
- IBIT: usable immediately for pilot via Nasdaq + BlackRock + OCC, but production-grade data needs OPRA/vendor.
- CME: official automated access exists, but requires credentials/licensing or vendor/broker access; public web pages must not be scraped.
- OTC/RFQ: must be quote-verified before any economics are presented as executable.

This supports the strategic thesis: the opportunity is not just IV comparison. The edge is normalizing different wrappers, margin regimes, collateral regimes, venue access constraints, and client flows into structures that a treasury, miner, fund, or allocator can actually use.
