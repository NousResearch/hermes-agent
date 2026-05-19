# Idea 1 Treasury Case Study v0 — 10,000 BTC Holder

**Purpose:** Create the first board/investor proof artifact for the BTC treasury covered-call/collar wedge.

**Prepared:** 2026-05-14 CDT  
**Data basis:** Screen-only Deribit BTC options snapshot from 2026-05-14 ~19:16 CDT.  
**Execution confidence:** Screen-only, not quote-verified, not trade-verified.  
**Legal status:** Illustrative research only. Not an offer, recommendation, or executable quote.

---

## 1. Hypothetical client profile

**Company:** Public BTC treasury company  
**BTC holdings:** 10,000 BTC  
**BTC reference price used in snapshot:** approximately $81,600–$82,200  
**Gross BTC treasury value:** approximately $816M–$822M  
**Program objective:** Generate recurring premium income and/or define a downside floor while preserving the strategic long-term BTC thesis.

**Core governance principle:** Only a minority tranche is used.

Suggested initial program sizing:

- **Total BTC holdings:** 10,000 BTC
- **Program sleeve:** 1,000 BTC, or 10% of holdings
- **Uncovered/unencumbered strategic BTC:** 9,000 BTC, or 90% of holdings

This framing matters. The company is not “selling its BTC thesis.” It is monetizing or protecting a controlled treasury tranche.

---

## 2. Board-level framing

The board question should not be:

> Are we bearish on BTC?

The correct question is:

> Can a defined minority tranche of BTC volatility be converted into recurring income or downside protection without undermining the company’s long-term BTC accumulation strategy?

A treasury derivatives policy should define:

- maximum BTC encumbrance percentage;
- permitted instruments;
- maximum upside cap;
- minimum acceptable floor;
- tenor limits;
- counterparty/venue limits;
- margin/collateral rules;
- independent valuation process;
- reporting cadence;
- board/audit committee approvals;
- disclosure thresholds.

---

## 3. Structure A — 43-day covered-call sleeve

**Snapshot tenor:** Deribit 26JUN26, DTE 43  
**Underlying reference:** ~$81,607  
**ATM IV:** ~37.9%  
**Sleeve:** 1,000 BTC

### Call overwrite alternatives

**10% OTM call — strike $90,000**
- Moneyness: ~110.3% of spot
- Premium: ~0.01604 BTC per BTC
- Premium on 1,000 BTC: ~16.04 BTC
- USD equivalent: ~$1.31M
- Annualized premium vs BTC notional: ~13.6%
- Tradeoff: strongest income, but upside capped close to spot.

**15% OTM call — strike $94,000**
- Moneyness: ~115.2% of spot
- Premium: ~0.00856 BTC per BTC
- Premium on 1,000 BTC: ~8.56 BTC
- USD equivalent: ~$699k
- Annualized premium vs BTC notional: ~7.3%
- Tradeoff: more upside retained, lower income.

**20%+ OTM call — strike $100,000**
- Moneyness: ~122.5% of spot
- Premium: ~0.00334 BTC per BTC
- Premium on 1,000 BTC: ~3.34 BTC
- USD equivalent: ~$273k
- Annualized premium vs BTC notional: ~2.8%
- Tradeoff: preserves more convexity, but income is modest.

### Interpretation

For a BTC treasury company, the 15% OTM version is the cleaner board candidate than the 10% OTM version. It generates visible income while reducing the risk that the company is perceived as selling too much upside.

**Indicative board framing:**

> Sell 43-day calls against 10% of BTC holdings, targeting 10–20% OTM strikes. Retain 90% of BTC fully uncapped. Use premium for operating expenses, debt service, preferred dividends, or BTC accumulation.

---

## 4. Structure B — 43-day downside floor / collar

**Structure:** Buy 74,000 put, sell 90,000 call on 1,000 BTC sleeve.  
**Tenor:** 43 days.  
**Spot reference:** ~$81,607.

**Protection leg:**
- Put strike: $74,000
- Floor: ~90.7% of spot
- Put premium: ~0.01939 BTC per BTC
- Cost on 1,000 BTC: ~19.39 BTC / ~$1.58M

**Financing leg:**
- Call strike: $90,000
- Cap: ~110.3% of spot
- Net collar cost: ~0.00335 BTC per BTC debit
- Net USD cost on 1,000 BTC: ~-$273k

### Interpretation

This structure creates a short-term downside floor about 9% below spot and caps upside about 10% above spot on the 10% sleeve. It is close to zero-cost but still a small net debit in the screen snapshot.

**Board fit:** Moderate. Useful around a financing/refi/covenant window, but the cap is tight for a BTC-maximalist treasury.

**Better use case:**
- debt issuance window;
- preferred dividend/coupon period;
- expected market stress;
- temporary protection around a major corporate event.

---

## 5. Structure C — 43-day put-spread collar

**Structure:** Buy 74,000 put / sell 65,000 put, financed partly by selling 94,000 call / buying 105,000 call.  
**Tenor:** 43 days.  
**Spot reference:** ~$81,607.

**Downside protection:**
- Protection starts near $74,000 (~90.7% of spot)
- Protection band extends to $65,000 (~79.6% of spot)
- Protected band: ~11% of spot

**Upside sold:**
- Call spread starts near $94,000 (~115.2% of spot)
- Call spread ends near $105,000 (~128.7% of spot)
- Upside give-up band: ~13.5% of spot

**Net screen cost:**
- ~0.00678 BTC per BTC
- On 1,000 BTC: ~6.78 BTC
- USD equivalent: ~$553k

### Interpretation

This is a cleaner BTC-treasury structure than a tight full collar. It protects a defined downside band while avoiding unlimited upside give-up above the call-spread cap. The company gives up some upside between ~$94k and ~$105k, but regains participation above the long call.

**Board fit:** Stronger than the full collar for high-conviction BTC treasuries.

---

## 6. Structure D — 134-day covered-call sleeve

**Snapshot tenor:** Deribit 25SEP26, DTE 134  
**Underlying reference:** ~$82,180  
**ATM IV:** ~40.7%  
**Sleeve:** 1,000 BTC

### Call overwrite alternatives

**~10% OTM call — strike $90,000**
- Moneyness: ~109.5% of spot
- Premium: ~0.05936 BTC per BTC
- Premium on 1,000 BTC: ~59.36 BTC
- USD equivalent: ~$4.88M
- Annualized premium vs BTC notional: ~16.2%
- Tradeoff: large premium, tight cap for a 4.5-month tenor.

**~15% OTM call — strike $95,000**
- Moneyness: ~115.6% of spot
- Premium: ~0.04204 BTC per BTC
- Premium on 1,000 BTC: ~42.04 BTC
- USD equivalent: ~$3.45M
- Annualized premium vs BTC notional: ~11.5%
- Tradeoff: attractive income but still meaningful upside cap.

**~20% OTM call — strike $98,000**
- Moneyness: ~119.3% of spot
- Premium: ~0.03381 BTC per BTC
- Premium on 1,000 BTC: ~33.81 BTC
- USD equivalent: ~$2.78M
- Annualized premium vs BTC notional: ~9.2%
- Tradeoff: more palatable cap with still significant income.

### Interpretation

The longer tenor creates more meaningful premium. But it also creates more headline risk if BTC rallies sharply. For a public BTC treasury, a 4.5-month overwrite should probably be implemented as a ladder, not a single block.

**Preferred design:**
- 3–4 staggered monthly/quarterly expiries;
- strikes 15–25% OTM;
- max 10% BTC encumbrance at launch;
- explicit board review before scaling.

---

## 7. Structure E — 134-day zero/low-cost collar

**Structure:** Buy 74,000 put, sell 90,000 call on 1,000 BTC sleeve.  
**Tenor:** 134 days.  
**Spot reference:** ~$82,180.

**Protection leg:**
- Put strike: $74,000
- Floor: ~90.0% of spot
- Put premium: ~0.05538 BTC per BTC
- Cost on 1,000 BTC: ~55.38 BTC / ~$4.55M

**Financing leg:**
- Call strike: $90,000
- Cap: ~109.5% of spot
- Net collar credit: ~0.00398 BTC per BTC
- Net USD credit on 1,000 BTC: ~$327k

### Interpretation

This collar is approximately self-financing in the screen snapshot, but it caps upside tightly for 134 days. That is commercially dangerous for a BTC treasury company unless tied to a specific obligation or risk window.

**Use only when:**
- the company has a specific financing/covenant/refi deadline;
- the board prioritizes balance-sheet floor over upside on the sleeve;
- the sleeve is small and explicitly disclosed as risk management.

---

## 8. Structure F — 134-day put-spread collar

**Structure:** Buy 74,000 put / sell 66,000 put, financed by selling 95,000 call / buying 105,000 call.  
**Tenor:** 134 days.  
**Spot reference:** ~$82,180.

**Downside protection:**
- Protection starts near $74,000 (~90.0% of spot)
- Protection extends to $66,000 (~80.3% of spot)
- Protected band: ~9.7% of spot

**Upside sold:**
- Call-spread starts near $95,000 (~115.6% of spot)
- Call-spread ends near $105,000 (~127.8% of spot)
- Upside give-up band: ~12.2% of spot

**Net screen cost:**
- ~0.00432 BTC per BTC
- On 1,000 BTC: ~4.32 BTC
- USD equivalent: ~$355k

### Interpretation

This is the best board-friendly structure in the screen set. It protects a meaningful downside band, avoids unlimited upside cap, and has a relatively modest net cost versus the protected notional.

**Board fit:** Strong.

**Narrative:**

> The company protects a defined drawdown band on 10% of BTC holdings while preserving most upside and retaining 90% of BTC entirely unencumbered.

---

## 9. Recommended launch policy

### Initial policy limits

- Maximum program sleeve: 10% of BTC holdings.
- Maximum single-expiry sleeve: 3–5% of BTC holdings.
- Preferred tenors: 30–150 days.
- Preferred covered-call strikes: 15–25% OTM.
- Preferred collar floors: 80–90% of spot.
- Avoid full upside caps unless tied to specific liability/covenant windows.
- Prefer call-spread financing over naked call overwrites for longer tenors.

### Recommended first live-style package

For a 10,000 BTC holder:

1. **Income sleeve:**
   - 500 BTC in staggered covered calls.
   - 15–25% OTM.
   - 30–90 day maturities.

2. **Protection sleeve:**
   - 500 BTC in put-spread collars.
   - floor starts near 85–90% of spot.
   - use call spread rather than outright call sale where possible.

3. **Unencumbered reserve:**
   - 9,000 BTC untouched.

This creates proof of discipline without over-optimizing the first trade.

---

## 10. Economics summary from screen snapshot

For a 1,000 BTC sleeve:

### 43-day tenor

- 15% OTM covered call could generate roughly **8.56 BTC / ~$699k**.
- 20%+ OTM covered call could generate roughly **3.34 BTC / ~$273k**.
- 90% floor / 110% cap collar costs roughly **3.35 BTC / ~$273k**.
- 90/80 put-spread funded by 115/129 call-spread costs roughly **6.78 BTC / ~$553k**.

### 134-day tenor

- 15% OTM covered call could generate roughly **42.04 BTC / ~$3.45M**.
- 20% OTM covered call could generate roughly **33.81 BTC / ~$2.78M**.
- 90% floor / 109.5% cap collar produces roughly **3.98 BTC / ~$327k credit**.
- 90/80 put-spread funded by 116/128 call-spread costs roughly **4.32 BTC / ~$355k**.

Again: these are screen-only approximations, not executable quotes.

---

## 11. Key risk disclosure for board package

### Upside opportunity cost

Covered calls and collars may underperform unhedged BTC in sharp rallies.

### Mark-to-market volatility

Even if held to expiry, option positions can show interim losses/gains.

### Collateral and liquidity

Some structures may require collateral, margin, or asset encumbrance.

### Counterparty/venue risk

Deribit, CME, ETF options, and OTC each carry different counterparty, clearing, custody, and settlement risks.

### Basis/wrapper risk

BTC direct holdings, ETF shares, CME futures, and Deribit options are not identical under stress.

### Disclosure and governance

A public company must consider materiality, risk disclosure, board authorization, auditor review, and insider/public communication controls.

### Narrative risk

Investors may misunderstand hedging as bearishness. Communications must emphasize limited sleeve, treasury discipline, and preservation of long-term BTC ownership.

---

## 12. What needs verification before client use

1. Quote-verify each structure with at least two counterparties or executable venues.
2. Confirm whether client holds direct BTC, ETF shares, or both.
3. Confirm unencumbered BTC and custody/collateral restrictions.
4. Review board-approved treasury policy.
5. Review debt/preferred/convert covenants.
6. Confirm accounting treatment with auditor.
7. Confirm legal authority to advise, structure, or execute.
8. Validate independent pricing/marks.
9. Stress test BTC +25%, +50%, -25%, -50% scenarios.
10. Produce disclosure-safe language.

---

## 13. Conclusion

This case study supports the commercial wedge.

A 10,000 BTC treasury can use a **10% sleeve** to generate meaningful premium income or create a downside floor without compromising the strategic ownership of the remaining 90%.

The most board-friendly first structures are not aggressive full covered calls. They are:

1. staggered 15–25% OTM covered calls on a small sleeve; and
2. put-spread collars that protect a drawdown band while avoiding unlimited upside give-up.

This is the product that can open client conversations. The cross-venue vol monitor is the pricing and hedge engine behind it.
