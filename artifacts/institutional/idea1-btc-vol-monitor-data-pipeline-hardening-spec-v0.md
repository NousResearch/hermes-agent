# Idea 1 BTC Vol Desk Monitor Data Pipeline Hardening Spec v0

**Purpose:** Define the production-grade data pipeline needed to turn the current BTC Vol Desk Monitor from a one-off screen-only snapshot into a reliable daily evidence engine.

**Status:** Technical/product specification. Not trading advice. Monitor outputs remain screen-only unless linked to quote/trade records.

---

## 1. Objective

Build a repeatable daily monitor that normalizes BTC volatility across:

- Deribit BTC options;
- IBIT / spot BTC ETF options;
- iShares/BlackRock ETF holdings and BTC-per-share data;
- OCC series/open-interest validation;
- CME BTC futures/options once licensed/vendor/broker data is available;
- OTC/RFQ quotes once legally approved.

The pipeline should produce:

- daily normalized option surfaces;
- ATM IV term structures;
- skew/risk-reversal tables;
- calendar spreads;
- wrapper-relative dislocation board;
- bid/ask/liquidity filters;
- margin/collateral annotations;
- source and execution confidence labels;
- auditable raw-data archive.

---

## 2. Core design principle

The monitor must never confuse **screen evidence** with **execution evidence**.

Every row, chart, and dislocation must carry:

1. **Source confidence**
   - model estimate;
   - public screen;
   - semi-official public JSON;
   - vendor screen;
   - official exchange/API;
   - broker/FCM source;
   - indicative quote;
   - firm quote;
   - trade.

2. **Execution confidence**
   - hypothesis;
   - screen-only;
   - quote-verified;
   - trade-verified.

---

## 3. Data source map

## 3.1 Deribit

**Role:** Crypto-native BTC volatility reference surface.

**Source:** Official public API.

**Endpoints:**
- get instruments;
- get book summary by currency;
- get order book by instrument;
- historical volatility if needed.

**Fields:**
- instrument name;
- expiry;
- strike;
- call/put;
- bid;
- ask;
- mark;
- mark IV;
- underlying/index price;
- open interest;
- volume;
- Greeks from order book endpoint;
- timestamp.

**Hardening requirements:**
- retry logic;
- schema validation;
- stale timestamp checks;
- expiry parser;
- strike parser;
- remove inactive/expired instruments;
- bid/ask sanity checks;
- mark IV unit normalization;
- OI/volume conversion consistency.

---

## 3.2 IBIT / spot BTC ETF options

**Role:** TradFi listed-options BTC volatility surface.

**Research source:** Nasdaq public JSON / other public chain source.

**Production source needed:** OPRA/vendor/broker data.

**Fields:**
- ticker;
- expiry;
- strike;
- call/put;
- bid;
- ask;
- last;
- volume;
- open interest;
- implied volatility if available;
- Greeks if available;
- timestamp;
- underlying ETF price.

**Hardening requirements:**
- source fallback;
- numeric parser for commas, percents, blanks, dashes;
- expiry normalization;
- stale quote check;
- bid/ask crossed/zero filters;
- American exercise/assignment annotation;
- option multiplier handling;
- underlying ETF price timestamp check.

**Caveat:**
Public Nasdaq JSON is research-only. Investor/client materials should call it semi-official/public unless replaced by OPRA/vendor/broker data.

---

## 3.3 iShares / BlackRock holdings and BTC-per-share

**Role:** Convert IBIT ETF option exposure into BTC-equivalent notional.

**Fields:**
- BTC holdings;
- shares outstanding;
- NAV;
- market price;
- cash/other holdings;
- expense ratio if needed;
- data timestamp.

**Derived:**
- BTC per ETF share;
- ETF shares per BTC;
- ETF option contract BTC-equivalent;
- ETF strike to BTC-equivalent strike;
- ETF option notional in BTC.

**Hardening requirements:**
- robust CSV parser;
- identify BTC row by multiple possible labels;
- parse shares outstanding even if format changes;
- fallback to official factsheet/holdings page if CSV changes;
- store raw file;
- compare day-over-day BTC/share drift;
- alert if BTC/share changes unexpectedly.

**Known issue:**
The first live parser missed the BTC row in one run. This is priority hardening.

---

## 3.4 OCC

**Role:** Official validation for listed option series/open interest where available.

**Fields:**
- underlying;
- expiry;
- strike;
- call/put;
- contract symbol;
- open interest;
- series status;
- deliverable/multiplier if relevant.

**Hardening requirements:**
- map OCC symbology to vendor/Nasdaq option chain;
- reconcile series count;
- reconcile OI totals where possible;
- flag missing or stale series.

---

## 3.5 CME

**Role:** Regulated futures/options BTC volatility and basis layer.

**Production source needed:** CME licensed API, DataMine, broker/FCM feed, vendor, or manual terminal export.

**Fields:**
- futures contract;
- futures bid/ask/settlement;
- basis vs spot;
- option contract;
- expiry;
- strike;
- call/put;
- bid;
- ask;
- settlement;
- implied vol;
- open interest;
- volume;
- contract multiplier;
- initial/maintenance margin;
- block liquidity;
- fees;
- position/accountability limits.

**Hardening requirements:**
- do not scrape blocked public pages;
- define production data contract/source;
- normalize futures basis;
- map options to futures expiry;
- convert contract multiplier to BTC-equivalent;
- add FCM margin data;
- label as missing until sourced.

---

## 3.6 OTC/RFQ quotes

**Role:** Move economics from screen-only to quote-verified.

**Source:** Legally approved quote process only.

**Fields:**
- RFQ ID;
- quote ID;
- counterparty;
- indicative/firm/trade;
- structure legs;
- notional;
- bid/offer/premium;
- IV/Greeks if provided;
- margin/collateral;
- settlement;
- documentation;
- quote timestamp;
- quote good-until;
- legal capacity;
- source/execution confidence.

**Hardening requirements:**
- strict audit trail;
- raw response archive;
- quote-age checks;
- quote-to-model comparison;
- legal role field required;
- no client name unless approved.

---

## 4. Normalized schema

Every normalized option row should include:

```yaml
record_id:
run_id:
as_of_cst:
source_name:
source_confidence:
execution_confidence:
venue:
wrapper: spot_btc | etf_option | cme_future_option | deribit_option | otc_quote
underlying_reference:
native_symbol:
option_type:
expiry:
dte:
strike_native:
strike_btc_equivalent:
spot_native:
btc_spot_reference:
forward_reference:
moneyness_spot:
moneyness_forward:
delta:
gamma:
vega:
theta:
iv_bid:
iv_ask:
iv_mid:
iv_mark:
price_bid:
price_ask:
price_mid:
price_mark:
open_interest:
volume:
bid_ask_width_abs:
bid_ask_width_pct:
contract_multiplier:
btc_equivalent_per_contract:
notional_btc:
notional_usd:
settlement_method:
collateral_currency:
margin_initial:
margin_maintenance:
fee_estimate:
timestamp_source:
stale_flag:
quality_flags:
raw_pointer:
```

---

## 5. Derived analytics

## 5.1 ATM IV term structure

For each venue/wrapper:
- select nearest ATM by forward moneyness where available;
- fallback to spot moneyness;
- compute bid/mid/ask IV;
- report DTE/expiry;
- include liquidity filters.

Output:
- table;
- line chart;
- term slope metrics.

---

## 5.2 Skew / risk reversals

For each expiry:
- identify 25-delta put;
- identify 25-delta call;
- compute 25D risk reversal = call IV - put IV or put IV - call IV, explicitly define convention;
- compute 10D tails later.

If Greeks unavailable:
- approximate delta by model;
- label as model-estimated.

---

## 5.3 Calendar spreads

For ATM and selected moneyness:
- compare near vs far IV;
- compute term slopes;
- identify event humps;
- avoid comparing stale/illiquid expiries.

---

## 5.4 Wrapper dislocation board

Compare:
- IBIT vs Deribit;
- IBIT vs CME;
- CME vs Deribit;
- OTC vs screen;
- ETF option vs BTC-equivalent option.

Every candidate row includes:
- gross IV difference;
- premium difference;
- bid/ask cost;
- basis adjustment;
- collateral/margin adjustment;
- liquidity score;
- legal usability;
- confidence label;
- next action: watch, quote, reject, trade candidate.

---

## 6. Quality controls

### 6.1 Source checks
- endpoint status;
- response timestamp;
- schema version;
- record count;
- missing fields;
- duplicate instruments;
- expired instruments;
- unexpected zeroes.

### 6.2 Market sanity checks
- bid <= ask;
- IV within plausible bounds;
- price non-negative;
- call/put parity warning;
- monotonic strike sanity where possible;
- OI/volume non-negative;
- DTE positive;
- moneyness in expected range.

### 6.3 Cross-source checks
- BTC spot across sources within tolerance;
- IBIT price vs BTC/share implied BTC value;
- OCC series vs option-chain series;
- Deribit index vs external BTC spot;
- CME futures basis vs spot.

### 6.4 Alert conditions
- source failed;
- record count drops >X%;
- BTC/share changes unexpectedly;
- IV spike/drop >Y vol points;
- bid/ask width expands >threshold;
- large new wrapper dislocation;
- stale data;
- quote good-until expired.

---

## 7. Storage design

Minimum storage:

```text
artifacts/institutional/data/
  raw/
    deribit/YYYY-MM-DD/run_id.json
    ibit/YYYY-MM-DD/run_id.json
    ishares/YYYY-MM-DD/run_id.csv
    occ/YYYY-MM-DD/run_id.csv
    cme/YYYY-MM-DD/run_id.*
    rfq/YYYY-MM-DD/quote_id.*
  normalized/
    options_surface/YYYY-MM-DD/run_id.parquet_or_csv
    atm_term_structure/YYYY-MM-DD/run_id.csv
    skew/YYYY-MM-DD/run_id.csv
    dislocations/YYYY-MM-DD/run_id.csv
  reports/
    btc-vol-desk-monitor-YYYY-MM-DD.md
```

Every run gets:
- run ID;
- CST timestamp;
- source versions;
- raw pointers;
- quality report;
- generated monitor report.

---

## 8. Report layout

Daily report sections:

1. Header
   - date/time CST;
   - run ID;
   - source status;
   - confidence legend.

2. BTC/ETF reference
   - BTC spot;
   - IBIT price;
   - BTC/share;
   - ETF shares per BTC;
   - CME futures basis if available.

3. Deribit surface
   - ATM IV by expiry;
   - OI/volume;
   - skew;
   - liquidity notes.

4. IBIT/ETF surface
   - ATM IV by expiry;
   - OI/volume;
   - skew;
   - BTC-equivalent notional.

5. CME surface
   - futures curve;
   - options IV;
   - margin notes;
   - if unavailable, explicit missing-data status.

6. Cross-venue comparison
   - wrapper IV differences;
   - calendar/skew spreads;
   - bid/ask filters.

7. Dislocation board
   - candidate;
   - venue pair;
   - gross spread;
   - all-in caveats;
   - confidence;
   - next action.

8. RFQ/quote updates
   - new quotes;
   - stale quotes;
   - quote-verified changes.

9. Quality warnings
   - source failures;
   - parser issues;
   - stale/missing fields.

10. Action list
   - watch;
   - quote;
   - reject;
   - fix data.

---

## 9. Implementation plan

### Phase 1 — Reliable daily snapshot

- Deribit ingestion hardened.
- IBIT chain parser hardened.
- iShares BTC/share parser fixed.
- Normalized CSV output.
- Daily markdown monitor.
- Quality report.

### Phase 2 — Surface analytics

- ATM term structures;
- skew/risk reversal;
- calendar spreads;
- bid/ask liquidity filters;
- dislocation board.

### Phase 3 — CME integration

- choose source;
- ingest futures curve;
- ingest options chain;
- add margin/basis;
- cross-venue comparison.

### Phase 4 — RFQ integration

- quote repository;
- quote-to-screen comparison;
- update confidence labels;
- quote expiration/staleness tracking.

### Phase 5 — Dashboard/API

- interactive filters;
- historical charts;
- client-specific scenario export only after legal approval;
- API or SaaS layer later.

---

## 10. Acceptance criteria

The data pipeline is v1-ready when it can:

- run repeatedly without manual edits;
- store raw and normalized data;
- calculate BTC/share correctly;
- produce Deribit and IBIT ATM IV tables;
- compute BTC-equivalent ETF option notionals;
- flag stale/bad data;
- generate daily markdown report;
- preserve confidence labels;
- show CME as either integrated or explicitly missing;
- accept RFQ quote records later;
- produce an auditable trail from report row to raw source.

---

## 11. Immediate technical priorities

1. Fix and harden iShares BTC holdings/BTC-share parser.
2. Create normalized schema CSV output.
3. Store raw source files by run ID.
4. Add quality flags and source status.
5. Implement ATM IV selection consistently.
6. Implement ETF BTC-equivalent conversion.
7. Add dislocation board output.
8. Define CME source path.
9. Define RFQ quote record import.

---

## 12. Bottom line

The monitor is the evidence engine for the entire business.

If the monitor is sloppy, the fund/desk concept becomes narrative. If the monitor is reliable, timestamped, normalized, and confidence-labeled, it becomes the spine for:

- investor credibility;
- treasury/miner structuring;
- RFQ verification;
- quote/trade evidence;
- eventual risk-intermediation economics.
