# Idea 1 Miner Production Hedge Case Study v0

**Purpose:** Create a proof artifact for the Bitcoin miner hedging wedge: protect production/cash-flow runway without forcing spot sales or abandoning BTC upside.

**Prepared:** 2026-05-14 19:24 CDT  
**Data basis:** Screen-only Deribit BTC options snapshot from 2026-05-14 ~19:24 CDT.  
**Execution confidence:** Screen-only, not quote-verified, not trade-verified.  
**Legal status:** Illustrative research only. Not an offer, recommendation, or executable quote.

---

## 1. Hypothetical client profile

**Company:** Public Bitcoin miner  
**Monthly BTC production assumption:** 150 BTC  
**3-month production forecast:** 450 BTC  
**Hedge ratio:** 50% of conservative 3-month production  
**Hedged BTC:** 225 BTC  
**BTC reference price:** approximately $81,600–$82,200  
**Hedged notional:** approximately $18.4M

Illustrative operating context:

- quarterly power/opex burden: high and USD-denominated;
- ASIC/site/capex needs: ongoing;
- equity story: long BTC beta and operational leverage;
- board concern: avoid forced BTC sales or dilutive equity during BTC drawdowns;
- lender concern: protect liquidity/collateral if BTC falls.

The product should be framed as **runway protection**, not bearish BTC speculation.

---

## 2. Why miners need a different hedge than treasuries

BTC treasury companies primarily hedge **inventory/NAV volatility**.

Miners hedge a moving target:

- future production volume is uncertain;
- realized BTC revenue depends on hashprice, difficulty, uptime, and pool economics;
- USD liabilities are recurring and non-discretionary;
- power costs and curtailment can spike;
- capex timing can force liquidity decisions;
- BTC inventory may already be pledged to lenders.

Therefore the first rule is:

> Hedge conservative production, not optimistic hashrate targets.

A miner hedge should usually start at 25–50% of conservative expected production, not 100%.

---

## 3. Product objective

The objective is not to maximize option P&L.

The objective is to protect a minimum USD revenue band for a defined production window while preserving most upside.

A board-ready miner policy should answer:

- How much expected production can be hedged?
- Are existing BTC holdings pledged or unencumbered?
- Is the hedge tied to power/capex/debt-service dates?
- What downside price creates forced-sale or equity-raise risk?
- What upside can the company afford to cap on the hedged slice?
- What collateral/margin/counterparty path is allowed?

---

## 4. Structure A — 43-day protective puts

**Tenor:** Deribit 26JUN26, DTE 43  
**BTC reference:** ~$81,580  
**ATM IV:** ~37.9%  
**Hedged amount:** 225 BTC

Protective put alternatives:

**$74,000 put**
- Floor: ~90.7% of spot
- Premium: ~0.01938 BTC per BTC
- Cost on 225 BTC: ~$356k
- Cost as % of hedged notional: ~1.94%
- Best for: near-term high-confidence protection around power/capex/lender window.

**$70,000 put**
- Floor: ~85.8% of spot
- Premium: ~0.01117 BTC per BTC
- Cost on 225 BTC: ~$205k
- Cost as % of hedged notional: ~1.12%
- Best for: budget protection with less premium drag.

**$65,000 put**
- Floor: ~79.7% of spot
- Premium: ~0.00583 BTC per BTC
- Cost on 225 BTC: ~$107k
- Cost as % of hedged notional: ~0.58%
- Best for: disaster insurance only.

### Interpretation

Outright puts are easiest for a board to understand: pay premium, retain all upside, receive protection below the strike.

But miners are often cash-sensitive. A $356k premium on a 225 BTC hedge may be acceptable around a specific risk window, but recurring outright puts can become expensive.

---

## 5. Structure B — 43-day collars

**Tenor:** 43 days  
**Hedged amount:** 225 BTC

Screen-only collar alternatives:

**$74,000 put / $94,000 call**
- Floor: ~90.7% of spot
- Cap: ~115.2% of spot
- Net cost: ~$201k debit
- Fit: strong floor, still reasonable upside cap.

**$70,000 put / $100,000 call**
- Floor: ~85.8% of spot
- Cap: ~122.6% of spot
- Net cost: ~$145k debit
- Fit: more balanced miner hedge.

**$65,000 put / $100,000 call**
- Floor: ~79.7% of spot
- Cap: ~122.6% of spot
- Net cost: ~$47k debit
- Fit: low-cost disaster hedge.

### Interpretation

For miners, the **$70k put / $100k call** looks like the cleanest 43-day structure in this snapshot:

- protects below roughly 86% of spot;
- allows BTC to rally more than 20% before cap binds;
- costs materially less than outright put protection;
- matches a short runway/capex window.

---

## 6. Structure C — 43-day put-spread collar

**Structure:** Buy $74,000 put / sell $62,000 put, financed partly by selling $94,000 call / buying $100,000 call.  
**Tenor:** 43 days  
**Hedged amount:** 225 BTC

Economics:

- Protection starts: $74,000
- Protection ends: $62,000
- Protected price band: $12,000 per BTC
- Protected USD band on 225 BTC: ~$2.70M
- Upside sold from $94,000 to $100,000
- Upside band sold on 225 BTC: ~$1.35M
- Net cost: ~$186k debit

### Interpretation

This is a useful miner structure because it protects a defined stress band and avoids unlimited upside give-up.

It is better than a tight full collar when the miner wants to preserve upside beta for equity investors.

---

## 7. Structure D — 134-day protective puts

**Tenor:** Deribit 25SEP26, DTE 134  
**BTC reference:** ~$82,150  
**ATM IV:** ~40.7%  
**Hedged amount:** 225 BTC

Protective put alternatives:

**$74,000 put**
- Floor: ~90.1% of spot
- Cost on 225 BTC: ~$1.02M
- Cost as % of hedged notional: ~5.54%
- Fit: expensive but strong protection for a full quarter-plus risk window.

**$70,000 put**
- Floor: ~85.2% of spot
- Cost on 225 BTC: ~$754k
- Cost as % of hedged notional: ~4.08%
- Fit: balanced but still premium-heavy.

**$66,000 put**
- Floor: ~80.3% of spot
- Cost on 225 BTC: ~$545k
- Cost as % of hedged notional: ~2.95%
- Fit: disaster protection.

### Interpretation

Outright 134-day puts are clean but expensive. They may be appropriate if the miner faces a specific debt maturity, site energization delay, ASIC payment, or lender review period.

For normal recurring hedging, collars or put spreads are likely more efficient.

---

## 8. Structure E — 134-day collars

**Tenor:** 134 days  
**Hedged amount:** 225 BTC

Screen-only collar alternatives:

**$74,000 put / $94,000 call**
- Floor: ~90.1% of spot
- Cap: ~114.4% of spot
- Net cost: ~$192k debit
- Fit: strong protection but relatively tight cap over 4.5 months.

**$70,000 put / $98,000 call**
- Floor: ~85.2% of spot
- Cap: ~119.3% of spot
- Net cost: ~$132k debit
- Fit: balanced quarterly/capex hedge.

**$66,000 put / $102,000 call**
- Floor: ~80.3% of spot
- Cap: ~124.2% of spot
- Net cost: ~$78k debit
- Fit: low-cost disaster hedge with wide upside room.

### Interpretation

The **$70k put / $98k call** is the best balanced quarterly hedge in this snapshot. It protects below ~85% of spot while leaving nearly 20% upside on the hedged production.

For a miner, this is commercially easier than outright puts because it limits premium outlay.

---

## 9. Structure F — 134-day put-spread collar

**Structure:** Buy $74,000 put / sell $62,000 put, financed partly by selling $94,000 call / buying $102,000 call.  
**Tenor:** 134 days  
**Hedged amount:** 225 BTC

Economics:

- Protection starts: $74,000
- Protection ends: $62,000
- Protected price band: $12,000 per BTC
- Protected USD band on 225 BTC: ~$2.70M
- Upside sold from $94,000 to $102,000
- Upside band sold on 225 BTC: ~$1.80M
- Net cost: ~$265k debit

### Interpretation

This is the cleanest board/lender structure when the miner wants protection but also wants to avoid a full upside cap.

It creates a defined stress buffer for the next 4.5 months while preserving upside outside the sold call-spread band.

---

## 10. Downside scenario impact

Using the 43-day snapshot and a 225 BTC hedge:

If BTC falls **20%** from ~$81,580 to roughly ~$65,264:

- unhedged 225 BTC production value declines by roughly $3.67M versus spot reference;
- a $74,000 put floor pays approximately **$1.97M** before premium/friction;
- this can fund a meaningful portion of power, payroll, or capex during drawdown.

If BTC falls **30%** to roughly ~$57,106:

- unhedged 225 BTC production value declines by roughly $5.51M versus spot reference;
- a $74,000 put floor pays approximately **$3.80M** before premium/friction;
- this materially reduces forced-sale or emergency-equity risk.

Using the 134-day snapshot:

- a $74,000 put floor pays approximately **$1.86M** at -20%;
- approximately **$3.71M** at -30%;
- before premium/friction.

The point is not that every miner should buy 90% floors. The point is that the product directly translates BTC volatility into runway protection.

---

## 11. Recommended miner launch policy

### Hedge-sizing policy

- Hedge only 25–50% of conservative expected production at launch.
- Do not hedge optimistic production targets.
- Exclude production already committed under financing or hosting arrangements.
- Separately classify treasury BTC as:
  - pledged;
  - restricted;
  - operational liquidity;
  - strategic reserve.

### Product policy

Preferred first structures:

1. **Quarterly production collar**
   - floor: 80–90% of spot;
   - cap: 115–125% of spot;
   - tenor: 60–150 days.

2. **Put-spread collar**
   - protect a defined stress band;
   - use call spread to avoid unlimited upside give-up;
   - best for board/lender conversations.

3. **Outright puts**
   - use around specific high-risk windows;
   - debt maturity, lender review, site capex, power-price stress, or expected financing.

Avoid:

- 100% production hedges;
- full forward sales unless the company explicitly wants no upside;
- short naked optionality beyond board-approved BTC inventory/production;
- structures requiring collateral the company cannot post under stress.

---

## 12. Board/lender package language

A miner can present the program as:

> The company is implementing a limited production hedge program on a conservative portion of expected BTC production to protect power, capex, and debt-service runway during BTC drawdowns, while preserving the majority of upside exposure for shareholders.

For lenders:

> The hedge reduces downside revenue volatility and supports minimum liquidity planning during collateral stress periods.

For shareholders:

> The program does not eliminate BTC upside. It only hedges a minority of expected production and uses collars/put spreads to retain upside participation.

---

## 13. Qualification checklist before client terms

1. Actual trailing monthly BTC production.
2. Conservative 3–6 month production forecast.
3. Hashrate, uptime, pool, and difficulty assumptions.
4. Expected monthly power cost and curtailment plan.
5. ASIC/site/capex payment schedule.
6. Debt, convert, lease, or project-finance covenants.
7. BTC held in treasury and whether pledged/restricted.
8. Normal policy: sell production daily, hold BTC, or mixed?
9. Permitted derivatives and approved venues/counterparties.
10. Collateral and margin capacity.
11. Board/audit/lender approval process.
12. Required accounting/tax treatment.

---

## 14. Why this proof artifact matters

This case study makes the miner wedge concrete:

- It ties option structures to power/capex/debt runway.
- It avoids overhedging.
- It preserves the miner’s upside equity narrative.
- It gives lenders and boards a practical risk framework.
- It creates recurring monitoring/reporting demand.

This is a real client-flow product, not a speculative vol-arb pitch.

---

## 15. Best next miner package

For the first client-ready prototype:

- assume 450 BTC conservative 3-month production;
- hedge 225 BTC, or 50%;
- show three options:
  1. 85% floor / 120% cap quarterly collar;
  2. 90/75 put-spread funded by 115/125 call-spread;
  3. outright 80–85% disaster put;
- present the effect on cash runway under BTC -20%, -30%, and -40%.

The preferred institutional product is the **quarterly production collar or put-spread collar**, not a full forward sale.
