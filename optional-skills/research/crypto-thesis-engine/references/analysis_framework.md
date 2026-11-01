# Analysis Framework Reference

## Valuation Heuristics for Crypto Assets

### MCap/FDV Ratio Interpretation
| Ratio | Interpretation | Risk Level |
|-------|---------------|------------|
| >90% | Fully circulating, minimal dilution | LOW |
| 60-90% | Moderate unlock schedule remaining | LOW-MEDIUM |
| 30-60% | Significant supply still locked | MEDIUM-HIGH |
| <30% | Heavy dilution ahead, early stage | HIGH |

### Volume/MCap Ratio Interpretation
| Ratio | Interpretation | Liquidity Grade |
|-------|---------------|----------------|
| >20% | Extremely active trading | VERY HIGH |
| 10-20% | High trading interest | HIGH |
| 3-10% | Normal range for most tokens | MODERATE |
| 1-3% | Light trading, wider spreads expected | LOW |
| <1% | Thin liquidity, high slippage risk | VERY LOW |

### ATH Distance Interpretation
| Distance | Phase | Typical Implication |
|----------|-------|-------------------|
| -0% to -20% | Near ATH / Discovery | High momentum, possible overheating |
| -20% to -50% | Correction | Normal pullback, depends on trend |
| -50% to -80% | Deep Correction | Value zone if fundamentals intact |
| -80% to -95% | Capitulation | High risk/reward. Most won't recover. |
| >-95% | Dead zone | Project likely failed unless major catalyst |

### Momentum Classification
| Signal | Definition |
|--------|-----------|
| ACCELERATING_UP | All timeframes positive, short-term fastest |
| STEADY_UP | All timeframes positive, consistent |
| MIXED_BULLISH | 30d positive but short-term choppy |
| NEUTRAL | No clear directional trend |
| MIXED_BEARISH | 30d negative but short-term bouncing |
| STEADY_DOWN | All timeframes negative, consistent |
| ACCELERATING_DOWN | All timeframes negative, getting worse |

## Comparable Analysis Methodology

### How to Select Comparables
1. **Same Layer**: L1 vs L1, L2 vs L2, DeFi protocol vs DeFi protocol
2. **Same Narrative**: AI tokens compare to AI tokens, not DeFi
3. **Similar Stage**: Don't compare a new L2 to Ethereum
4. **MCap Proximity**: Compare within 2-3x of market cap range when possible
5. **Functional Overlap**: Projects solving similar problems

### Implied Price Calculation
```
Implied Price = (Comparable MCap / Target Circulating Supply)
```

Example: If NEAR has $5B MCap and AVAX has 400M circulating supply:
- Implied AVAX price at NEAR's MCap = $5B / 400M = $12.50

### Important Caveats for Comps
- Circulating supply differences can make direct MCap comps misleading
- FDV comparison often more useful for early-stage tokens
- TVL comparison only relevant for DeFi protocols
- Revenue/fees comparison emerging but data still inconsistent

## Risk Framework

### Risk Level Criteria
| Level | Criteria |
|-------|---------|
| **LOW** | Well-understood, mitigated, or not applicable |
| **MEDIUM** | Present but manageable, standard for the category |
| **HIGH** | Significant concern that could materially impact thesis |
| **CRITICAL** | Immediate threat, thesis-breaking risk |

### Risk Categories Deep Dive

#### Market Risk
- How correlated is the asset to BTC?
- What's the beta (does it amplify BTC moves)?
- Is it affected by specific sector rotation?

#### Liquidity Risk  
- Can you exit a meaningful position without 5%+ slippage?
- Is volume concentrated on 1-2 exchanges or distributed?
- DEX vs CEX liquidity split

#### Dilution Risk
- What % of supply is still locked?
- When are the next major unlocks?
- Are there ongoing emissions (inflation)?
- VC/team allocation that hasn't vested

#### Competition Risk
- How many direct competitors?
- Is there a clear moat (network effects, TVL, developer mindshare)?
- Can a fork or L2 capture the market?

#### Regulatory Risk
- Is it a security risk token?
- Has it been mentioned by regulators?
- Is the team doxxed/US-based?
- DeFi vs CeFi classification

#### Technical Risk
- Has the protocol been audited?
- Any history of exploits?
- Centralization vectors (admin keys, upgrade proxy)?
- Bridge dependencies

#### Narrative Risk
- Is the current narrative sustainable?
- How quickly can market attention shift?
- Is there real usage backing the narrative?

## Thesis Evolution Tracking

When re-analyzing a token, compare against previous thesis:

### What to Track
1. **Price delta** since last analysis
2. **Key metric changes** (MCap, volume, supply)
3. **Risk level changes** (any escalations or de-escalations?)
4. **Catalyst outcomes** (did predicted events happen?)
5. **New information** not present in last analysis

### Thesis Status Categories
| Status | Meaning |
|--------|---------|
| **INTACT** | Original thesis still valid, no material changes |
| **STRENGTHENED** | New evidence supports the thesis more strongly |
| **WEAKENED** | Assumptions challenged but thesis not broken |
| **INVALIDATED** | Core thesis no longer holds, reassess |
| **EVOLVED** | Thesis direction changed due to new information |
