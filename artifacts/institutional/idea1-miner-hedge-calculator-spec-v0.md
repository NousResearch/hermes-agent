# Idea 1 Miner Hedge Calculator Specification v0

**Purpose:** Define the calculator needed to convert the miner hedging concept into a repeatable board/lender-ready analysis package.

**Status:** Product specification. Not investment, derivatives, legal, tax, or accounting advice. All pricing remains screen-only unless quote-verified.

---

## 1. Objective

The miner hedge calculator answers one question:

> How much BTC price protection should a miner buy or structure against conservative expected production so that power, debt service, capex, and runway are protected without overhedging or eliminating strategic upside?

The calculator should produce:

- conservative production exposure;
- hedge notional;
- structure economics;
- premium/cost;
- downside runway protection;
- upside cap/give-up;
- collateral/margin impact;
- lender/covenant checklist;
- board-ready recommendation.

---

## 2. Target users

### Internal users
- structuring analyst;
- BTC Vol Desk Monitor operator;
- risk lead;
- sales/capital markets lead.

### External users later
- miner CFO;
- treasurer;
- CEO;
- board/risk committee;
- lender/project-finance counterparty;
- auditor/counsel.

---

## 3. Core design principle

The calculator must prevent overhedging.

It should default to conservative assumptions and force the user to distinguish:

- current BTC inventory;
- unencumbered BTC inventory;
- pledged/restricted BTC;
- forecast production;
- conservative production;
- optimistic production;
- BTC price hedge;
- hashprice hedge;
- power cost hedge.

Initial calculator scope is **BTC price hedge only**, not full hashprice/power hedging.

---

## 4. Input sections

## 4.1 Company / scenario metadata

Required fields:
- company name;
- ticker, if public;
- date/time/timezone;
- analyst;
- scenario name;
- source confidence;
- execution confidence;
- legal status: internal / counsel reviewed / client approved;
- notes.

---

## 4.2 BTC market assumptions

Required fields:
- BTC spot/reference price;
- source: Deribit index / CME reference / Coinbase / composite / vendor;
- reference timestamp;
- implied volatility source;
- risk-free rate;
- funding/collateral rate;
- quote source: model / screen / indicative / firm / trade;
- bid/ask spread assumption;
- slippage assumption;
- fees assumption.

Derived fields:
- BTC shock scenarios: -10%, -20%, -30%, -40%, -50%, +20%, +50%, +100%;
- spot scenario prices.

---

## 4.3 Miner operating assumptions

Required fields:
- current deployed hashrate;
- expected deployed hashrate over hedge period;
- fleet efficiency;
- uptime assumption;
- pool fee;
- curtailment assumption;
- network difficulty assumption or production forecast source;
- monthly production forecast;
- conservative production haircut;
- conservative monthly production;
- hedge horizon in months;
- total conservative production.

Simplified v0 option:
- enter monthly BTC production directly;
- enter conservative haircut;
- calculator derives conservative production.

Example:
- monthly production: 150 BTC;
- conservative haircut: 0%;
- horizon: 3 months;
- conservative production: 450 BTC.

---

## 4.4 Treasury / balance sheet inputs

Required fields:
- current BTC inventory;
- unencumbered BTC inventory;
- pledged/restricted BTC;
- cash balance;
- debt service due in hedge window;
- power cost due in hedge window;
- capex due in hedge window;
- other fixed obligations;
- minimum liquidity reserve;
- BTC-backed loan exposure, if any;
- lender collateral thresholds.

Derived fields:
- total obligations in hedge window;
- unhedged runway at current BTC price;
- unhedged runway at BTC shock prices;
- required protected value.

---

## 4.5 Hedge policy inputs

Required fields:
- hedge ratio: default 25–50% of conservative production;
- maximum hedge ratio allowed;
- permitted instruments;
- max premium budget;
- max collateral usage;
- minimum acceptable floor;
- maximum acceptable upside cap;
- hedge horizon;
- approved venues/counterparties;
- whether BTC inventory can be used as collateral;
- whether cash collateral is available.

Derived fields:
- hedge notional in BTC;
- hedge notional in USD;
- hedge notional as % of conservative production;
- hedge notional as % of unencumbered BTC inventory;
- hedge notional as % of total enterprise BTC exposure.

---

## 5. Supported structures

## 5.1 Outright protective put

Inputs:
- strike percent of spot: default 80%, 85%, 90%;
- expiry;
- premium source;
- premium quote.

Outputs:
- premium BTC/USD;
- premium as % of hedged notional;
- floor price;
- payoff at shock scenarios;
- net protection after premium;
- no upside cap.

Use case:
- miner wants disaster protection and no upside give-up.

---

## 5.2 Production collar

Inputs:
- put strike percent: default 80%, 85%, 90%;
- call strike percent: default 115%, 120%, 125%;
- expiry;
- premium source.

Outputs:
- put premium;
- call premium;
- net debit/credit;
- floor price;
- cap price;
- downside protection;
- upside give-up above cap;
- net scenario P&L;
- collateral/margin impact.

Use case:
- miner wants low-cost protection and can accept upside cap on limited production.

---

## 5.3 Put-spread collar

Inputs:
- long put percent: default 90%;
- short put percent: default 75–80%;
- short call percent: default 115–120%;
- long call percent: default 130–140%;
- expiry;
- premium source.

Outputs:
- net premium;
- protected downside band;
- max protection;
- upside band sold;
- max upside give-up;
- residual tail risk below short put;
- scenario P&L;
- collateral/margin.

Use case:
- board/lender-friendly hedge that controls cost and avoids unlimited upside sale.

---

## 5.4 Forward/futures hedge — later version

Not in v0 unless explicitly enabled.

Reason:
- eliminates upside;
- creates margin/collateral needs;
- more sensitive to accounting/covenant concerns.

Future outputs:
- locked price;
- basis;
- margin;
- cash-flow impact;
- hedge accounting analysis.

---

## 6. Pricing engine requirements

## 6.1 v0 model pricing

For internal screen-only estimates:
- Black-Scholes style approximation for European options;
- use BTC spot/reference;
- risk-free/funding rate;
- IV from Deribit/IBIT/CME/quote source;
- interpolate IV by moneyness/expiry where needed;
- show model limitations.

All v0 model results must be labeled:

> screen-only model estimate — quote verification required.

## 6.2 Quote-verified pricing

When quotes are available, quote data overrides model pricing.

Quote fields:
- quote ID;
- counterparty;
- quote timestamp;
- quote good-until;
- firm/indicative;
- bid/ask;
- size;
- fees;
- margin/collateral;
- settlement;
- documentation required.

## 6.3 Price hierarchy

Use this hierarchy:

1. executed trade;
2. firm quote;
3. indicative quote;
4. executable broker/venue screen;
5. vendor screen;
6. public screen;
7. model estimate.

---

## 7. Scenario outputs

The calculator should produce a scenario table for BTC:

- -50%;
- -40%;
- -30%;
- -20%;
- -10%;
- unchanged;
- +20%;
- +50%;
- +100%.

For each scenario, show:
- BTC price;
- unhedged production value;
- hedge payoff;
- premium cost;
- net hedged value;
- incremental runway protection;
- cap/give-up if BTC rallies;
- collateral/margin stress;
- obligations covered;
- remaining liquidity gap.

---

## 8. Board output summary

The top of the output should contain a plain-English recommendation:

Example:

> Hedge 225 BTC, equal to 50% of conservative 3-month production, using a 90D 85% floor / 120% cap collar. Based on current screen-only pricing, the structure protects approximately $X of downside value in a 30% BTC drawdown while preserving upside up to $Y/BTC. Quote verification and lender/covenant review are required before execution.

Required board summary fields:
- recommended hedge ratio;
- hedge notional;
- structure;
- floor;
- cap;
- premium/debit/credit;
- downside protection at -20%, -30%, -40%;
- upside give-up at +20%, +50%, +100%;
- collateral impact;
- legal/accounting/covenant status;
- confidence label.

---

## 9. Lender/covenant output

The calculator should flag:

- hedge notional vs allowed hedging limits;
- whether hedge obligations may count as debt;
- whether collateral posting creates liens/restricted assets;
- BTC-backed loan LTV impact;
- cross-default/cross-acceleration risk;
- consent required;
- cash liquidity for margin/collateral;
- whether structure is speculative or tied to bona fide exposure.

Output labels:
- clear;
- review required;
- consent likely required;
- prohibited unless amended;
- unknown.

---

## 10. Accounting/disclosure output

The calculator should not decide accounting treatment, but it should produce a checklist:

- ASC 815 derivative accounting review required;
- hedge accounting feasibility review;
- FASB crypto fair-value impact;
- fair-value source hierarchy;
- collateral/restricted asset disclosure;
- market-risk disclosure impact;
- MD&A liquidity impact;
- risk-factor update;
- Form 8-K/materiality analysis if relevant.

---

## 11. Risk warnings

The calculator must warn if:

- hedge ratio exceeds conservative production;
- hedge ratio exceeds 50% without approval;
- pledged/restricted BTC is included;
- premium exceeds budget;
- collateral exceeds limit;
- upside cap is below policy threshold;
- tenor exceeds approved range;
- quote is stale;
- source is model-only or public-screen-only;
- venue/counterparty is not approved;
- legal status is not counsel-reviewed.

---

## 12. Data model

Suggested output schema:

```yaml
scenario_id:
created_at_cst:
company:
analyst:
confidence:
  source: model | public_screen | vendor_screen | indicative_quote | firm_quote | trade
  execution: hypothesis | screen_only | quote_verified | trade_verified
market:
  btc_spot:
  spot_source:
  timestamp:
  iv_source:
  risk_free_rate:
  funding_rate:
miner:
  monthly_production_btc:
  conservative_haircut:
  conservative_monthly_production_btc:
  hedge_horizon_months:
  conservative_total_production_btc:
  current_inventory_btc:
  unencumbered_inventory_btc:
  pledged_inventory_btc:
obligations:
  power_cost_usd:
  debt_service_usd:
  capex_usd:
  other_usd:
  total_usd:
policy:
  hedge_ratio:
  hedge_notional_btc:
  max_premium_usd:
  max_collateral_usd:
  permitted_instruments:
structures:
  - type: protective_put | collar | put_spread_collar
    expiry:
    tenor_days:
    legs:
      - side:
        option_type:
        strike_pct:
        strike_abs:
        premium:
        iv:
        delta:
    net_premium_usd:
    margin_usd:
    collateral_usd:
    scenario_pnl:
recommendation:
  selected_structure:
  rationale:
  blockers:
  next_steps:
```

---

## 13. Interface v0

Simplest v0 implementation:

- spreadsheet or markdown-backed Python script;
- inputs in YAML/JSON;
- outputs markdown report and CSV scenario table;
- later upgrade to dashboard.

Recommended files:
- `miner_hedge_inputs.yaml`
- `miner_hedge_calculator.py`
- `miner_hedge_scenarios.csv`
- `miner_hedge_report.md`

---

## 14. Example default scenario

Default demo inputs:

- monthly production: 150 BTC;
- hedge horizon: 3 months;
- conservative total production: 450 BTC;
- hedge ratio: 50%;
- hedge notional: 225 BTC;
- BTC spot: latest monitor spot;
- structures:
  1. 90D 85% put;
  2. 90D 85% floor / 120% cap collar;
  3. 135D 90/75/115/130 put-spread collar.

Default output:
- compare net cost;
- show BTC -20/-30/-40 protection;
- show upside give-up;
- recommend one structure;
- mark as screen-only unless quote data loaded.

---

## 15. Acceptance criteria

The calculator is usable when it can:

- accept miner production and balance sheet inputs;
- prevent hedge notional exceeding conservative production unless explicitly overridden;
- price three standard structures;
- generate downside/upside scenario table;
- flag collateral, covenant, legal, accounting issues;
- label confidence correctly;
- produce a board-ready one-page summary;
- store assumptions and outputs for audit trail.

---

## 16. Bottom line

The miner calculator should make one thing obvious:

> A miner hedge is not a bearish BTC bet. It is a disciplined runway-protection policy applied to conservative production and bounded by collateral, covenant, and governance limits.

If the calculator cannot express that clearly, it is not ready for client use.
