# Idea 1 Legal Perimeter Memo v0

**Concept:** BTC volatility intermediation desk: treasury/miner hedging, ETF/CME/Deribit/OTC vol analytics, RFQ workflow, principal sleeve, structured products.

**Prepared:** 2026-05-14 CDT  
**Status:** Research memo for strategy/counsel review.  
**Not legal advice.** This is a gating map to prevent overclaiming and to define what must be reviewed by derivatives/securities counsel before client-facing execution.

---

## 1. Executive conclusion

The legal perimeter is the main gating issue for Idea 1.

The concept is commercially strong, but the desk cannot safely jump from analytics to “we quote BTC derivative structures to public companies” without a defined regulatory wrapper.

The correct launch sequence is:

1. **Research and internal analytics** — lowest friction.
2. **Illustrative term sheets and education** — possible with careful disclaimers and no tailored advice.
3. **Bespoke advisory/structuring** — likely registration/partner/counsel analysis.
4. **RFQ/execution support** — high broker/IB/FCM/broker-dealer risk.
5. **Principal risk-taking / warehousing** — possible, but depends on whether spot, futures, swaps, securities, or notes are involved.
6. **Structured products/distribution** — highest legal/product-approval complexity.

The MVP should be framed as:

> Research, analytics, illustrative case studies, legal perimeter design, and quote-verification planning.

It should not be framed as:

> A live principal derivatives desk quoting public-company BTC collars.

---

## 2. Instrument classification spine

Every product must first be classified by instrument.

| Instrument / wrapper | Main perimeter |
|---|---|
| Spot BTC | CFTC anti-fraud/anti-manipulation; state money transmission/custody issues may apply depending activity |
| BTC ETF shares / ETF options | SEC, securities/options, FINRA/broker-dealer/RIA issues |
| CME BTC futures/options | CFTC/NFA, FCM/IB/CTA/CPO issues |
| Deribit BTC options | Jurisdiction/offshore exchange/counterparty/custody; CFTC/NFA implications if US-facing advice/execution |
| OTC BTC swaps/options | CFTC swap/commodity-option/ECP/SD rules; ISDA/CSA documentation |
| BTC-linked notes | Securities Act, Exchange Act, broker-dealer/FINRA/RIA, structured-products governance |
| Pooled vehicle trading BTC derivatives | CPO/CTA; potentially Advisers Act / Investment Company Act issues |

This is why the memo must separate research, advice, execution, principal trading, and product distribution.

---

## 3. Activity phase map

## Phase 0 — Internal research / proprietary analytics

**Activities:**
- Build BTC Vol Desk Monitor.
- Analyze Deribit/IBIT/CME/OTC surfaces.
- Run internal case studies.
- Generate screen-only dislocation board.
- Trade own capital internally if separately approved.

**Lower-risk boundaries:**
- No client-specific advice.
- No order solicitation.
- No transaction-based compensation.
- No client asset custody.
- No claims of executable pricing unless verified.

**Regulatory note:**
Pure internal research is generally lowest friction. If internal proprietary trading uses listed futures/options, exchange/clearing/position/reporting rules still apply.

**Go/no-go:** Green for MVP.

---

## Phase 1 — Generic published research / education

**Activities:**
- Market commentary.
- Generic BTC vol research.
- Educational treasury/miner hedging content.
- Illustrative term sheets marked hypothetical.
- Public white paper or investor concept memo.

**Main gates:**
- CTA risk if compensated advice about futures, commodity options, swaps, or commodity interests.
- RIA risk if compensated advice about securities, ETF options, BTC-linked notes, funds, or investment contracts.
- Broker-dealer/FINRA risk if research is tied to securities solicitation or transaction compensation.

**Allowed MVP posture:**
- General education only.
- No personalized recommendation.
- No “buy/sell this option” language.
- No performance claim from screen-only data.
- Prominent disclaimer: illustrative, not executable, not legal/tax/accounting advice.

**Go/no-go:** Green/yellow. Green for generic research; yellow if paid or tailored.

---

## Phase 2 — Bespoke advisory / structuring

**Activities:**
- Designing a BTC treasury collar for a specific company.
- Advising a miner on production hedge ratios.
- Recommending ETF option overlays.
- Designing swap/OTC structures.
- Producing client-specific board packages.

**Main gates:**
- **CTA:** if advising for compensation on futures, commodity options, swaps, or commodity interests.
- **RIA:** if advising on securities, BTC ETF shares/options, BTC-linked notes, funds, or securities-based swaps.
- **CPO:** if operating/soliciting a pool that trades commodity interests.
- **Broker-dealer:** if structuring is tied to distribution, placement, transaction compensation, or securities sales.

**Practical counsel questions:**
1. Is the advice about securities, commodity interests, swaps, or both?
2. Is compensation advisory, subscription, transaction-based, or performance-based?
3. Is the client a public company, RIA client, fund, ECP, accredited investor, or retail?
4. Are we only delivering analysis, or inducing a transaction?
5. Are we using a registered partner or relying on registration/exemption?

**Go/no-go:** Yellow/red. Do not do client-specific advisory until counsel defines wrapper or partner model.

---

## Phase 3 — RFQ and execution support

**Activities:**
- Sending RFQs to dealers.
- Introducing client to market maker or OTC desk.
- Soliciting/accepting orders.
- Negotiating option/swap/structured-note terms.
- Routing trades.
- Receiving spread, commission, referral fee, markup, or transaction compensation.

**Main gates:**
- **CFTC Introducing Broker / FCM:** if soliciting or accepting orders for futures/options/swaps and not accepting margin/funds.
- **FCM:** if accepting customer funds/margin/property for futures/options/swaps.
- **Broker-dealer/FINRA:** if effecting or inducing securities transactions, including ETF options, notes, funds, or securities-based swaps.
- **Swap dealer:** if regularly making markets/holding out as counterparty in swaps.

**Hard rule:**
Transaction-based compensation is a major risk factor. Avoid any RFQ/execution economics until counsel approves the role.

**Permissible lower-risk planning:**
- Build RFQ templates.
- Define fields required for quote verification.
- Identify categories of counterparties.
- Run purely internal mock RFQs.
- Do not solicit live trades for clients.

**Go/no-go:** Red until legal/registration/partner model is established.

---

## Phase 4 — Principal risk-taking / warehousing

**Activities:**
- Desk trades own BTC/options/futures book.
- Warehouses client-flow risk.
- Makes markets in OTC derivatives.
- Hedges structured-product exposure.

**Main gates by activity:**

| Principal activity | Gate |
|---|---|
| Own-account spot BTC trading | Generally lower registration risk, but anti-fraud/manipulation/custody/tax controls apply |
| Own-account CME futures/options | CFTC exchange/clearing/reporting/position rules |
| OTC swap market-making | Swap dealer analysis |
| Principal securities/notes trading | SEC dealer/broker-dealer analysis |
| Warehousing client-flow risk | Depends on whether acting as dealer, broker, adviser, counterparty, or fund |

**Risk-control requirement:**
Principal risk cannot be a casual extension of advisory. It needs:
- entity structure;
- capital allocation;
- risk limits;
- compliance policy;
- counterparty documentation;
- valuation controls;
- market-conduct policy;
- custody/margin controls.

**Go/no-go:** Yellow/red. Possible later, not MVP.

---

## Phase 5 — Structured products / distribution

**Activities:**
- BTC-linked notes.
- Reverse convertibles.
- Autocallables.
- Buffered notes.
- Principal-protected notes.
- BTC income products.
- Wealth-channel distribution.

**Main gates:**
- BTC-linked notes are generally securities.
- Public offerings require Securities Act registration unless an exemption applies.
- Distribution usually triggers broker-dealer/FINRA analysis.
- RIA use requires fiduciary/best-interest analysis.
- Underlying hedge may trigger CFTC/swap/FCM issues.
- Product approval, suitability, Reg BI, options disclosure, FINRA communications, and structured-product guidance apply.

**Controls needed:**
- registered issuer/shelf or private-placement exemption;
- product committee approval;
- target-market definition;
- client eligibility grid;
- payoff/scenario disclosure;
- issuer credit-risk disclosure;
- tax memo;
- secondary-market/liquidity policy;
- hedge-source transparency.

**Go/no-go:** Red for MVP. Phase 2+ only with issuer/distributor/legal framework.

---

## 4. Public-company BTC treasury program checklist

Before any BTC treasury derivatives program is client-ready:

### Governance
- Board or committee approval.
- Written treasury derivatives policy.
- Permitted instruments.
- Maximum BTC encumbrance.
- Tenor/strike/floor limits.
- Counterparty/venue list.
- Authority matrix for trade approval and collateral movement.
- Escalation rules for margin/collateral stress.

### SEC disclosure
Potential disclosure areas:
- Reg S-K Item 105 risk factors.
- Reg S-K Item 303 MD&A trends/liquidity.
- Reg S-K Item 305 market-risk disclosures.
- Reg S-X derivative accounting-policy disclosure.
- Form 8-K analysis for material agreements/events.
- Reg FD controls for material hedge information.
- SEC crypto exposure/counterparty/custody disclosure themes.

### Accounting/auditor
- FASB ASU 2023-08 for crypto fair value.
- ASC 815 derivative accounting.
- Hedge accounting feasibility, if any.
- Independent valuation and fair-value hierarchy.
- Collateral/restricted asset treatment.
- Internal controls over trade approval, valuation, custody, collateral, and journal entries.

### Debt/covenant/collateral
- Are derivatives permitted?
- Are hedge obligations permitted debt?
- Are BTC/cash collateral liens permitted?
- Are pledged BTC assets available?
- Cross-default/cross-acceleration with ISDA?
- Liquidity stress under BTC gap moves?

### OTC/ISDA
- ISDA Master, Schedule, CSA/credit support.
- Digital asset definitions.
- ECP status.
- Reference price/source/fallbacks.
- Fork/airdrop/protocol disruption clauses.
- Eligible collateral/haircuts.
- Netting/close-out enforceability.
- Valuation agent and dispute process.

---

## 5. Miner-specific checklist

Miners need all public-company controls above, plus exposure-specific controls:

### Exposure definition
- BTC inventory hedge vs future production hedge.
- Hashprice exposure vs BTC price exposure.
- Power-price exposure vs BTC revenue hedge.
- Production from owned mining vs hosting/customer exposure.

### Hedge ratio
- Use conservative production estimates.
- Start with 25–50% of expected production.
- Avoid hedging machine deployment targets or optimistic hashrate.
- Stress test difficulty, curtailment, downtime, pool luck, and power prices.

### Covenant/collateral
- Check equipment finance, project finance, power agreements, hosting agreements, and BTC-backed loans.
- Determine whether BTC is pledged/restricted.
- Confirm lender consent before posting collateral or entering secured hedges.

### Accounting
- Forecasted production hedges raise probability and hedge-accounting questions.
- Cash-settled derivatives may simplify delivery/custody but create liquidity/margin needs.
- Physical settlement requires custody, wallet, sanctions, and transfer controls.

---

## 6. Wealth/RIA/structured-product checklist

For ETF-option overlays, wealth products, and notes:

### Client channel
- Broker-dealer, RIA, dual registrant, bank, private fund, or institutional counterparty?
- Retail, accredited, QP, QIB, ECP, or institutional?

### Recommendation standard
- Reg BI for retail broker-dealer recommendations.
- FINRA suitability/KYC/supervision/communications.
- RIA fiduciary duty: care, loyalty, mandate fit, cost/conflicts, alternatives.
- Options account approval and OCC disclosure if clients directly trade options.

### Product governance
- New product committee.
- Target market / negative target market.
- Training and sales scripts.
- Scenario analysis.
- Concentration limits.
- Post-sale surveillance.
- Conflict and compensation disclosures.

### Structured notes
- Securities Act registration or exemption.
- Prospectus/pricing supplement or private placement memo.
- Issuer credit-risk disclosure.
- Secondary liquidity process.
- Embedded fees/hedging conflicts.
- BTC/ETF/options/futures risk factors.

---

## 7. Red/yellow/green activity matrix

| Activity | Status for MVP | Comment |
|---|---:|---|
| Internal BTC vol monitor | Green | Build now |
| Screen-only public research | Green/yellow | Avoid tailored advice/performance claims |
| Illustrative treasury/miner case studies | Green/yellow | Mark hypothetical/screen-only/not executable |
| Client-specific hedge recommendation | Yellow/red | Counsel/registration/partner required |
| Sending client RFQs | Red | Execution/IB/BD risk |
| Receiving transaction compensation | Red | High broker/IB/BD risk |
| Principal OTC swap dealing | Red | Swap dealer/counterparty analysis |
| Own-account listed derivatives trading | Yellow | Possible with proper account/risk/compliance |
| Pooled vehicle trading derivatives | Red until structured | CPO/CTA/investment adviser analysis |
| BTC-linked structured notes | Red | Securities/product/distribution framework required |
| Wealth-platform ETF-option overlay | Yellow/red | BD/RIA/options suitability and supervision |

---

## 8. Required counsel workstream

Before any client-facing business launch, counsel should produce:

1. Activity classification memo.
2. Entity/wrapper recommendation.
3. Registration/exemption analysis:
   - CTA/CPO;
   - IB/FCM;
   - RIA;
   - broker-dealer/FINRA;
   - swap dealer;
   - securities offering exemptions.
4. Communications/disclaimer policy.
5. Research publication policy.
6. Client eligibility matrix.
7. RFQ/execution role policy.
8. Compensation model review.
9. Public-company treasury derivatives policy template.
10. ISDA/CSA/digital asset terms checklist.
11. Structured-product go/no-go memo.

---

## 9. Source references

### CFTC / NFA
- CFTC Commodity Exchange Act and Regulations: https://www.cftc.gov/LawRegulation/CommodityExchangeAct/index.htm
- Commodity Exchange Act definitions, 7 U.S.C. § 1a: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title7-section1a&num=0&edition=prelim
- CFTC Commodity Trading Advisors: https://www.cftc.gov/IndustryOversight/Intermediaries/CTAs/index.htm
- CFTC Commodity Pool Operators: https://www.cftc.gov/IndustryOversight/Intermediaries/CPOs/index.htm
- CFTC Introducing Brokers: https://www.cftc.gov/IndustryOversight/Intermediaries/IBs/index.htm
- NFA who has to register: https://www.nfa.futures.org/registration-membership/who-has-to-register/index.html
- NFA CTA: https://www.nfa.futures.org/registration-membership/who-has-to-register/cta.html
- NFA CPO: https://www.nfa.futures.org/registration-membership/who-has-to-register/cpo.html
- NFA IB: https://www.nfa.futures.org/registration-membership/who-has-to-register/ib.html
- CFTC Coinflip Bitcoin commodity release: https://www.cftc.gov/PressRoom/PressReleases/7231-15

### SEC / FINRA / OCC
- SEC broker-dealer overview: https://www.sec.gov/about/divisions-offices/division-trading-markets/broker-dealers
- Exchange Act broker-dealer definitions, 15 U.S.C. § 78c: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section78c&num=0&edition=prelim
- Exchange Act broker-dealer registration, 15 U.S.C. § 78o: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section78o&num=0&edition=prelim
- Securities Act definition of security, 15 U.S.C. § 77b: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section77b&num=0&edition=prelim
- Securities Act registration, 15 U.S.C. § 77e: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section77e&num=0&edition=prelim
- Advisers Act definitions, 15 U.S.C. § 80b-2: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section80b-2&num=0&edition=prelim
- Advisers Act registration, 15 U.S.C. § 80b-3: https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section80b-3&num=0&edition=prelim
- SEC spot bitcoin ETP approval order: https://www.sec.gov/files/rules/sro/nysearca/2024/34-99306.pdf
- SEC ISE options on iShares Bitcoin Trust order: https://www.sec.gov/files/rules/sro/ise/2024/34-101128.pdf
- OCC options disclosure document: https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document
- FINRA broker-dealer registration: https://www.finra.org/registration-exams-ce/broker-dealers
- FINRA Rule 2360 Options: https://www.finra.org/rules-guidance/rulebooks/finra-rules/2360
- FINRA Rule 2111 Suitability: https://www.finra.org/rules-guidance/rulebooks/finra-rules/2111
- FINRA Rule 2090 KYC: https://www.finra.org/rules-guidance/rulebooks/finra-rules/2090
- FINRA Rule 3110 Supervision: https://www.finra.org/rules-guidance/rulebooks/finra-rules/3110
- FINRA Rule 2210 Communications: https://www.finra.org/rules-guidance/rulebooks/finra-rules/2210
- SEC Reg BI: https://www.sec.gov/rules-regulations/2019/06/regulation-best-interest-form-crs
- SEC structured notes bulletin: https://www.sec.gov/investor/pubs/structurednotes.htm
- FINRA Notice 05-59 structured products: https://www.finra.org/rules-guidance/notices/05-59
- FINRA Notice 12-03 complex products: https://www.finra.org/rules-guidance/notices/12-03

### Public-company / accounting / OTC
- Reg S-K Item 105: https://www.law.cornell.edu/cfr/text/17/229.105
- Reg S-K Item 303: https://www.law.cornell.edu/cfr/text/17/229.303
- Reg S-K Item 305: https://www.law.cornell.edu/cfr/text/17/229.305
- SEC Market Risk Disclosure FAQ: https://www.sec.gov/divisions/corpfin/guidance/derivfaq.htm
- Reg S-X Rule 4-08: https://www.law.cornell.edu/cfr/text/17/210.4-08
- SEC crypto sample letter: https://www.sec.gov/rules-regulations/staff-guidance/disclosure-guidance/sample-letter-companies-regarding-recent
- FASB ASU 2023-08: https://storage.fasb.org/ASU%202023-08.pdf
- SEC SAB 122: https://www.sec.gov/rules-regulations/staff-guidance/staff-accounting-bulletins/staff-accounting-bulletin-122
- CFTC swap disclosure rule: https://www.law.cornell.edu/cfr/text/17/23.431
- CFTC swap confirmation rule: https://www.law.cornell.edu/cfr/text/17/23.501
- CFTC swap documentation rule: https://www.law.cornell.edu/cfr/text/17/23.504
- ISDA Digital Asset Derivatives Definitions release: https://cdn.aws.isda.org/a/EEygE/ISDA-Launches-Standard-Definitions-for-Digital-Asset-Derivatives.pdf
- Delaware DGCL §141: https://delcode.delaware.gov/title8/c001/sc04/index.html#141

---

## 10. Bottom line

The legally clean next step is not live execution.

The legally clean next step is:

1. keep outputs as research/analytics/illustrative;
2. produce a counsel-ready activity map;
3. build RFQ templates but do not send client RFQs;
4. decide whether the business launches as:
   - analytics/research company;
   - registered/partnered advisory/structuring shop;
   - fund/prop vehicle;
   - broker/FCM/OTC partner model;
   - structured-product partnership.

Until counsel defines the wrapper, every market price and structure remains **screen-only illustrative research**.
