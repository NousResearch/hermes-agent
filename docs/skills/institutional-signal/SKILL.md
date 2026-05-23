# Institutional Signal Skill

## Purpose

Reference for the 13F institutional signal pipeline - from SEC EDGAR filing ingestion through coinvest\_score\_z production signal and cohort quarantine governance. This is the dominant selector signal \(100% weight in v1.14.0\).

This skill is organized into two sections:

1. **Framework Reference** \- Stable architecture, rules, and processes \(changes only with code updates\)
2. **Operational State** \- Volatile status snapshots that require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Operator Context

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin \(Holland, MI\)
- **Relevant expertise**: 30+ years institutional investment management. Extensive manager research & selection across all asset classes \(led $110M EM equity mandate search\). CFA-credentialed with deep experience in institutional portfolio oversight \($14B+ AUM as Deputy CIO/Interim CIO at NDRIO\).
- **13F analysis background**: Career-long buy-side experience evaluating manager conviction through portfolio disclosures. Tracks Fairmount Funds, Deep Track Capital, and Logos Capital as elite biotech-focused managers. Applies coinvest signals to identify high-conviction biotech positions where multiple specialist managers independently converge.
- **Manager selection lens**: Evaluates active vs. passive implementation, manager style consistency, and position concentration. The coinvest\_score\_z signal extends this institutional manager-selection discipline to a systematic screening framework.

## Architecture Overview

```
SEC EDGAR 13F-HR filings
  -> warm_13f_cache.py (per-CIK PIT cache dirs)
  -> build_institutional_summary.py (canonical summary)
  -> coinvest_score_z (selector signal, 100% weight v1.14.0)
  -> inst_delta_z (governance-controlled; active in ranker only as of v1.14.0)
```

## Manager Registry

**File**: `production_data/manager_registry.json`
**Never edit directly** \- use `tools/onboard_manager.py`

### Tiers

| Tier | Description | Signal Weight |
| --- | --- | --- |
| elite\_core | Highest-conviction biotech-focused managers | Full weight |
| conditional | Broader institutional managers with biotech exposure | Reduced weight |

### Onboarding Flow

```bash
python tools/onboard_manager.py \
  --cik 1802528 \
  --name "Fairmount Funds Management" \
  --aum-b 1.3 \
  --style concentrated_clinical_stage \
  --tier elite_core \
  --notes "..."
```

One-shot flow: registry append -> backfill across every existing PIT dir \(lookback=40, approx 10y\) -> warm current as-of date -> run `tools/test_manager_integration.py` \(6/6 gate\).

Partial reruns: `--skip-registry`, `--skip-backfill`, `--skip-current`, `--skip-test`.

---

## coinvest\_score\_z

The production selector signal. Measures institutional co-investment conviction across elite biotech managers.

### Key Properties

- Drives 100% of selector weight \(v1.14.0, coinvest-only after inst\_delta\_z demotion\)
- Correlation with final\_score: rho = +0.882 \(double-count concern, documented in T1 ranker anatomy\)
- Checklist v2: 3/5 standalone, but bundle \(with inst\_delta\) is 5/5
- Collapse guard: SD floor = 0.10 \(below this, snapshot integrity check FAILs\)

### Data Flow

1. PIT cache: `data/caches/sec_13f/PIT/{YYYY-MM-DD}/` per manager
2. Canonical summary: `production_data/institutional_summary.json`
3. Delta computation: `institutional_summary_delta.json` \(pre vs post refresh\)
4. Score: z-scored across eligible universe per snapshot

---

## inst\_delta\_z

Quarter-over-quarter change in institutional holdings. Measures whether smart money is accumulating or distributing.

### Governance Rules

- Reinstatement requires IC recovery evidence documented in governance log
- When active, contributes to ranker \(dominant positive discriminator, NW-t = +3.32\)
- When zeroed, selector runs on coinvest\_score\_z alone

---

## insider\_net\_buy\_value\_90d \(Spec 104, Diagnostic Only\)

Form 4-derived insider buying signal. Tracks net insider purchase value over a trailing 90-day window.

### Status: DIAGNOSTIC ONLY

- Listed in `DIAGNOSTIC_FIELDS`, NOT in `ALPHA_FEATURE_REGISTRY`
- Tracked and exported for observability
- Does NOT enter the scoring model, ranker, or selector
- Does NOT affect ranks, actions, or position sizing

### Blank vs. Zero Semantics \(CRITICAL\)

| Value | Meaning |
| --- | --- |
| NaN / None / blank | Not fetched, no Form 4 coverage for this ticker |
| 0.0 | Fetched successfully, no insider buy activity in 90-day window |

Never collapse blank and zero. Never impute zero for missing or blank for zero.

### Expectation Model Isolation Guard \(Spec 104, R4a\)

The expectation model has an `insider_net_buy_z` weight that activates silently if `insider_net_buy_value_90d` flows into `market_features`. This is the fragile path that "diagnostic-only" depends on. Spec 104 requires an explicit guard: either runtime assertion that the field is NOT in `market_features`, or weight zeroing, or a pre-inference drop guard.

### Promotion Criteria \(future, not current build\)

Requires ALL of: 20+ stable snapshots with >= 60% non-null coverage, blank/zero integrity verified, IC > 0 at p < 0.05, Checklist v2 battery pass, explicit written approval. Until all five are met, insider stays diagnostic.

### Relationship to Crowd-Belief Estimation

If insider data eventually proves useful for crowd-belief estimation \(market expectation modeling\), it would enter through the expectation pipeline, NOT through the institutional signal pipeline. The 13F-based signals \(coinvest\_score\_z, inst\_delta\_z\) measure institutional conviction from quarterly portfolio disclosures. Insider buying measures company-insider conviction from Form 4 filings. They are different data sources with different provenance rules.

---

## 13F Refresh Cycle

SEC 13F filings have a 45-day lag from quarter-end. Filings typically cluster in the final 3 business days before the deadline.

### Pre-Refresh Readiness \(`tools/prep_13f_refresh.py`\)

5 guards, all must PASS:

| Guard | Check |
| --- | --- |
| 1 | Most recent snapshot has valid institutional\_summary\_delta.json |
| 2 | coinvest\_score\_z has healthy variance \(SD > 0.10\) |
| 3 | PIT cache has entries within 3 days of today |
| 4 | SEC EDGAR endpoint is reachable |
| 5 | Dry-run: build\_institutional\_summary\(\) produces valid output \(>=80% coverage\) |

Writes baseline artifact: `artifacts/13f_pre_refresh_baseline_{date}.json`

### Cohort Quarantine \(`tools/check_13f_cohort_quarantine.py`\)

Run after new filings land. Compares pre-refresh vs post-refresh snapshots.

**Sections:**

- A: Manager-level diff \(filing counts, coverage\)
- B: Coverage diff \(tickers\_with\_signal, signal\_coverage\_pct\)
- C: Per-ticker score diff \(coinvest\_score\_z, inst\_delta\_z distributions\)
- D: Top-30 churn \(Jaccard similarity, entries/exits, rank movement\)

**Verdicts:**

| Verdict | Meaning | Action |
| --- | --- | --- |
| CLEAN | Normal refresh, minimal churn | Proceed |
| QUARANTINE | Significant score/rank disruption | Hold for review |
| PRODUCER\_AUDIT\_REQUIRED | Anomalous coverage or manager changes | Deep investigation |

Telegram alerting on QUARANTINE/PRODUCER\_AUDIT\_REQUIRED \(suppressible with `--no-alert`\).

### Contamination Window

After adding new managers, a contamination window opens \(typically 20 trading days\). IC measurements during this window are flagged as contaminated and excluded from clean IC calculations.

---

## Data Provenance Rules

- **Holdings truth source**: `production_data/institutional_summary.json` is canonical
- **CUSIP-first, not issuer-first**: Always reason from CUSIP -> canonical ticker
- **Raw EDGAR XML is debug-only**: Never build narratives from raw filing parses
- **If raw count != summary count**: investigate the summary pipeline first

---

## Key Biotech 13F Filers to Track

Per user preference: Fairmount Funds, Deep Track Capital, Logos Capital.
Also monitor BioPharm IQ Twitter \([https://twitter.com/BioPharmIQ](https://twitter.com/BioPharmIQ)\).

---

## Source Files

| Component | File |
| --- | --- |
| Manager Onboarding | `tools/onboard_manager.py` |
| 13F Cache Warmer | `tools/warm_13f_cache.py` |
| Institutional Summary Builder | `build_institutional_summary.py` |
| 13F Refresh Readiness | `tools/prep_13f_refresh.py` |
| Cohort Quarantine | `tools/check_13f_cohort_quarantine.py` |
| Snapshot Collapse Guards | `tools/verify_snapshot_integrity.py` |
| Manager Registry | `production_data/manager_registry.json` |
| Institutional Summary | `production_data/institutional_summary.json` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** \- The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## inst\_delta\_z Current Status

*Last reviewed: 2026-05-18*

- **Zeroed in selector** \(2026-05-04\): ALERT confirmed, mean IC = -0.097 over 36 dates
- **Active in ranker**: dominant positive discriminator within top-30 \(NW-t = +3.32\)
- **Reinstatement conditions**: documented in governance log, requires IC recovery evidence

## 13F Filing Cycle Status

*Last reviewed: 2026-05-16*

- **Completed cycle**: Q1 2026 \(period ending March 31, 2026\) -- ALL THREE FILED May 15, 2026
- **Accession numbers**: Fairmount 0001104659-26-062419, Deep Track 0001856083-26-000003, Logos Global 0001172661-26-002196
- **Filing pattern**: All three filed on deadline day \(May 15\), consistent with Q1 2025 pattern \(all three also filed May 15, 2025\)
- **CIKs**: Fairmount 0001802528, Deep Track 0001856083, Logos Global 0001792126
- **Post-filing action sequence**: \(1\) Warm 13F cache, \(2\) Run cohort quarantine, \(3\) Check collapse guards \(coinvest\_score\_z SD\), \(4\) Refresh IC decomposition, \(5\) 5-day observation window before treating as production-grade
- **Next cycle**: Q2 2026 \(period ending June 30, 2026\). Filing deadline \~August 14, 2026. Monitor EDGAR starting \~August 11.
- **SEC compliance**: SEC\_USER\_AGENT preflight check added \(2026-05-13\) for EDGAR fair-use policy

## Q1 2026 13F-HR Filing Summary \(COMPLETE\)

*Filed: 2026-05-15*

**Fairmount Funds** \(accession 0001104659-26-062419\):

- AUM \~$1.38B. Key new: DAMORA THERAPEUTICS \($225.7M, 16.3% of portfolio -- largest new position, NOT signaled by 13D/13G pre-filing\). Massive APGE trim \(-85.4%\), COGT trim \(-38.9%\). Exits: KINIKSA, NUVALENT. VRDN held \(3.9M shares at 3/31\). Post-Q1: VRDN stake raised to 14.04% via $20M purchase May 11 \(13D/A filed May 13\). ORKA 19.99% held.

**Deep Track Capital** \(accession 0001856083-26-000003\):

- AUM $6,124M \(was $5,609M, +9.2%\). 63 positions \(was 55\). SPDR ETF hedge $1,277M. Top equity: GH $308M, IMVT $286M, TARS $252M, VCYT $250M, GPCR $206M. 16 new positions including ALMS \($149M\), NUVL \($141M\), GMAB \($98M\), DFNT \($57M\). Biggest adds: JANX +1225%, OCUL +144%, COGT +121%. Exits: DVAX \($242M largest\), MNMD, XENE, BHVN, RAPT.

**Logos Global** \(accession 0001172661-26-002196\):

- AUM $2,003M \(was $1,655M, +21.0%\). 66 positions. Top: RVMD $194.5M, ERSA $180.4M, IDYA $115M, TERN $105.4M. Massive CNTA add \(+963%, now $84.4M top-6 holding\). New: UTHR \($47M\), MDGL \($44.5M\), XENE \($26M\). 15 exits including CDTX \($68.5M\).

**Coinvest signals**: VRDN \(FM 3.9M shares + DT 1.4M shares at 3/31; DT accumulated to 5.4M post-Q1\). ORKA \(FM 3.7M + DT 2.0M\). Triple overlap on CRESCENT BIOPHARMA only. DT+Logos 22 overlaps; top by combined value: IMVT $313M, VCYT $267M, GPCR $230M, CNTA $210M \(new -- Logos +963%\).

## Q1 2026 Early Signals \(13G/13D, pre-13F-HR\) -- VALIDATED

*Captured: 2026-05-11. Validated against 13F-HR filings 2026-05-15.*

**Fairmount**: 13D/13G signals mostly validated. COGT trim confirmed \($243M sale 3/31\). ORKA 19.99% confirmed. VRDN accumulation confirmed \(3.9M shares at 3/31\). SURPRISE: DAMORA THERAPEUTICS \($225.7M, 16.3% of portfolio\) was NOT signaled by any pre-filing 13D/13G -- largest new position was invisible until 13F-HR dropped.

**Deep Track**: VRDN accumulation was primarily post-Q1 \(only 1.4M shares at 3/31 vs 5.4M by May per 13G\). ALMS confirmed \($149M new\). New positions not signaled by 13G: NUVL \($141M\), GMAB \($98M\), DFNT \($57M\).

**Logos**: TENX and AVLO confirmed. CNTA massive add \(+963%\) was NOT signaled pre-filing.

**Lesson**: 13D/13G pre-signals capture \~60-70% of major moves but systematically miss sub-5% positions and non-reporting-threshold changes. The largest surprises \(DAMORA for Fairmount, CNTA for Logos\) were invisible until 13F-HR.

## External Benchmarks and Platforms \(May 2026\)

### BiotechEdge Platform

- Tracks 20 specialist biotech hedge funds, $46.5B+ total assets, 1,558+ biotech companies, 2,318+ upcoming catalysts
- Automated NLP parsing of 13F, Form 4, and activist stakes from SEC EDGAR
- Fund convergence signal: when 3+ independent specialist funds independently initiate positions in same biotech company during same quarter - identified as "highest-conviction signal in biotech investing"
- Validates the DEM's coinvest\_score\_z: the fund convergence concept \(independent expert consensus reduces uncertainty\) is the same underlying insight

### Other 13F/Catalyst Platforms

- RxDataLab BioHedge Weekly: 17-18 biotech-focused funds, 13D/13G tracking
- BiotechSigns: 970 companies, 74,988 active signals
- CatalystAlert: 1,624 companies, 14,310 pipelines, 3,815 upcoming catalysts
- BioCatalysts.AI: Bio-Score algorithm predicting volatility magnitude per catalyst

### ODIN Engine \(External Benchmark for Clinical Scoring\)

- L2-regularized logistic regression, 51 engineered features, 8 signal categories
- AUC: 0.9363 on 2,210 historical FDA events \(2000-2025\)
- Verified accuracy: 96.2% on 53 outcomes \(Aug 2025-Feb 2026\), SHA-256 cryptographic proof
- ODIN feature categories not currently in DEM: manufacturing/CMC risk, FDA era effects, options market implied probability, sponsor historical approval rate by therapeutic area
- These are Tier 4 evaluation candidates through the standard T5 promotion path

### Industry Adoption Data

- 92% of hedge funds with $1B+ AUM now use AI/ML \(up from 56% in 2022\)
- 67% describe AI as "integral" rather than supplementary
- AI-integrated funds outperform traditional systematic strategies by 3-4 percentage points annually
