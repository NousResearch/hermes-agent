---
name: crypto-thesis-engine
description: >
  Self-correcting crypto research engine. Fetches live market data, produces institutional-grade
  research reports with confidence scoring, learns from outcomes, detects its own biases, and
  adapts dynamically. Features bias detection, risk override warnings, self-critique, and
  research performance pattern analysis. NOT financial advice — a research tool that improves
  over time. Gets smarter with every analysis.
version: 2.1.0
author: frank
license: MIT
platforms: [macos, linux]
prerequisites:
  commands: [python3]
metadata:
  hermes:
    tags: [crypto, research, analysis, defi, market-data, learning, decision-engine]
    requires_toolsets: [terminal]
---

# Crypto Thesis Engine

A self-correcting cryptocurrency decision engine that transforms raw market data into
institutional-quality investment thesis reports. This is NOT a price ticker — it's a
research analyst that **thinks**, **learns from its mistakes**, and **questions itself**.

**What makes this different:**
- 🧠 **Learns** — Track whether analyses were correct or wrong. Accuracy improves over time.
- 📊 **Adaptive Confidence** — Confidence scores adjust dynamically based on historical
  performance, bias detection, and category-specific accuracy.
- 🔍 **Pattern Awareness** — Surfaces similar past analyses and their outcomes before making
  new predictions.
- 🔬 **Self-Critique** — Every report includes an honest assessment of its own assumptions,
  blind spots, and biggest uncertainties.
- ⚠️ **Risk Overrides** — Automatically warns when the current setup matches historically
  underperforming patterns.
- 🩺 **Bias Diagnosis** — Detects bullish/bearish skew, category blind spots, and
  overconfidence patterns. Recommends corrections.

## When to Use

- User asks to **analyze** a cryptocurrency or token
- User wants a **research report** on a crypto asset
- User wants to **compare** two or more crypto projects
- User asks about **bull/bear cases** for a token
- User says things like: "What do you think about ETH?", "Give me a thesis on SOL",
  "Compare AVAX vs NEAR", "Should I look into ARB?"
- User wants to **teach the system** whether a past analysis was right or wrong
- User wants to check the skill's **past performance** or **accuracy**
- Triggered via `/crypto-thesis-engine` slash command

## Quick Reference

| Command | Description |
|---------|-------------|
| `analyze <token>` | Full research report with confidence score |
| `compare <t1> <t2> [t3...]` | Side-by-side comparison (2-5 tokens) |
| `report <token> --format brief\|full\|executive` | Specific report format |
| `scan <category>` | Scan a category (layer1, defi, ai, etc.) |
| `learn <token> correct\|wrong [note]` | Record outcome for learning |
| `history <token>` | Past analyses for a token |
| `stats` | Performance dashboard |
| `diagnose` | Self-diagnosis: biases, blind spots |
| `strategy` | Research performance pattern analysis |

## Commands

### `analyze <token_id>`
Generate a full research report for a single token.
Automatically includes confidence score and similar past analyses.
```
/crypto-thesis-engine analyze ethereum
/crypto-thesis-engine analyze solana
/crypto-thesis-engine analyze arbitrum
```

### `compare <token_id_1> <token_id_2> [token_id_3...]`
Side-by-side comparative analysis of 2-5 tokens.
```
/crypto-thesis-engine compare ethereum solana
/crypto-thesis-engine compare arbitrum optimism base
```

### `report <token_id> [--format brief|full|executive]`
Generate a specific report format:
- `brief` — 1-page summary with key metrics and verdict
- `full` — Complete thesis with all sections (default)
- `executive` — Decision-maker focused, heavy on catalysts and risks
```
/crypto-thesis-engine report bitcoin --format executive
```

### `scan <category>`
Scan a category and surface the most interesting opportunities.
Categories: `layer1`, `layer2`, `defi`, `ai`, `gaming`, `rwa`, `meme`
```
/crypto-thesis-engine scan layer2
/crypto-thesis-engine scan defi
```

### `learn <token_id> <correct|wrong> [note]`
🆕 **Teach the system** whether a past analysis was accurate.
This is how the skill improves — feedback drives future confidence scores.
```
/crypto-thesis-engine learn solana correct
/crypto-thesis-engine learn arbitrum wrong "L2 narrative faded faster than expected"
/crypto-thesis-engine learn bitcoin correct "BTC indeed broke out as predicted"
```

### `history <token_id>`
Retrieve past analyses for this token, including outcomes and accuracy stats.
```
/crypto-thesis-engine history ethereum
```

### `stats`
View the skill's overall performance — accuracy by token, category, and global stats.
```
/crypto-thesis-engine stats
```

### `diagnose`
🆕 **Run a full self-diagnosis** — detects biases, blind spots, overconfidence, and
recommends corrective actions.
```
/crypto-thesis-engine diagnose
```

### `strategy`
🆕 **Analyze research performance patterns** — mines all historical data to show where
the system's analyses have been most/least accurate across categories, momentum signals,
and market cap tiers. This is a **self-assessment tool**, not investment advice.
```
/crypto-thesis-engine strategy
```

## Procedure

### Step 0: Load Memory & Compute Confidence (NEW — Run Before Every Analysis)

Before generating any analysis, check the learning memory for context:

**0a. Compute confidence score:**
```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" confidence \
    --token <token_id> \
    --categories <comma_separated_categories> \
    --data-completeness 0.8
```

This returns a weighted confidence score (20-95%) based on:
- Token-specific accuracy history (30% weight)
- Category accuracy history (25% weight)
- Bias detection penalty/boost (15% weight) — **adaptive adjustment**
- Global accuracy across all analyses (10% weight)
- Data completeness of current analysis (10% weight)
- Recency of similar correct predictions (10% weight)

**0b. Run risk override check:**
```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" risk-check \
    --token <token_id> \
    --categories <categories> \
    --sentiment <planned_sentiment> \
    --momentum <momentum_signal> \
    --mcap <market_cap> \
    --rank <rank>
```

This is the "pre-flight safety check". It returns warnings if:
- This **category** has historically low accuracy → confidence should drop
- This **sentiment** in this **category** has a poor track record
- The **sentiment conflicts with momentum** → historically risky
- **Similar token setups have failed** → pattern-based red flag

Risk levels: `CLEAR` (proceed) → `CAUTION` → `WARNING` → `DANGER` (reconsider)

**If risk_level is WARNING or DANGER**, you MUST:
1. Mention the risk overrides prominently in the report
2. Consider adjusting your sentiment toward neutral
3. Add extra weight to the bear case arguments

**0c. Find similar past analyses:**
```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" similar \
    --token <token_id> \
    --categories <categories> \
    --mcap <market_cap> \
    --rank <rank>
```

This surfaces past analyses of similar tokens (same category, similar market cap tier,
similar rank) and their outcomes. Use this to inform your current analysis.

**0c. Check token history:**
```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" history --token <token_id>
```

If previous analyses exist for this exact token, reference them. Note:
- How your last analysis performed (if outcome is recorded)
- How the price has moved since then
- Whether the thesis should be updated or maintained

### Step 1: Fetch Market Data

Run the data collection script to pull live data from CoinGecko API:

```bash
python3 "$(dirname "$0")/scripts/fetch_market_data.py" --token <token_id> --output json
```

For comparisons, fetch multiple tokens:
```bash
python3 "$(dirname "$0")/scripts/fetch_market_data.py" --tokens <id1>,<id2>,<id3> --output json
```

For category scans:
```bash
python3 "$(dirname "$0")/scripts/fetch_market_data.py" --category <category> --top 10 --output json
```

**If the script fails** (e.g., rate limit, network error), wait 10 seconds and retry once.
If it fails again, inform the user and offer to try with cached data if available.

### Step 2: Parse and Validate Data

The script outputs structured JSON. Validate these critical fields exist:
- `current_price`, `market_cap`, `total_volume`
- `price_change_percentage_24h`, `price_change_percentage_7d`, `price_change_percentage_30d`
- `ath`, `ath_change_percentage`, `ath_date`
- `atl`, `atl_change_percentage`, `atl_date`
- `circulating_supply`, `total_supply`, `max_supply`

If any field is `null`, note it as "Data unavailable" — do NOT fabricate numbers.
Count the available fields to determine data_completeness for confidence scoring.

### Step 3: Compute Derived Metrics

Using the raw data, compute these derived metrics:

1. **Valuation Metrics**
   - Market Cap / FDV ratio (circulating vs. total supply ratio)
   - Volume / Market Cap ratio (liquidity indicator)
   - Distance from ATH (%) — how far the asset has fallen from peak
   - Distance from ATL (%) — how far it's risen from bottom

2. **Momentum Signals**
   - 24h vs 7d vs 30d trend direction (accelerating/decelerating)
   - Volume trend (is volume confirming price moves?)
   - Relative strength vs BTC and ETH (if data available)

3. **Supply Dynamics**
   - Circulating/Total supply ratio — unlock risk indicator
   - If max_supply exists: inflation trajectory
   - If FDV >> Market Cap: significant dilution risk flag

### Step 4: Generate Analysis (The Thinking Part)

This is where you add value beyond the data. For each section below, reason through the data
like a senior research analyst. DO NOT simply restate numbers. Derive meaning.

**IMPORTANT — Integrate learning context into your analysis:**
- Reference the confidence score from Step 0a in your report
- If similar past analyses exist (Step 0b), mention them and their outcomes
- If this token was analyzed before (Step 0c), compare with the previous thesis
- Adjust your language based on confidence:
  - HIGH (80%+): "Based on strong historical accuracy..."
  - MODERATE (60-79%): "With moderate confidence..."
  - LOW (40-59%): "With limited historical data, cautiously..."
  - VERY LOW (<40%): "Note: this analysis type has been unreliable..."

#### 4a. Overview
- What is this project? (1-2 sentences from your knowledge, DO NOT use API for this)
- Where does it sit in the crypto ecosystem? (L1, L2, DeFi, infra, etc.)
- Current market position summary in plain language

#### 4b. Bull Case (Why it could go up)
Construct 3-5 arguments based on:
- Favorable valuation metrics (e.g., "trading 85% below ATH while fundamentals improved")
- Positive momentum signals
- Healthy supply dynamics
- Network/ecosystem growth you know about
- Upcoming catalysts from your knowledge base
- Cross-reference: if similar projects have higher valuations, there's a gap to close

#### 4c. Bear Case (Why it could go down)
Construct 3-5 counterarguments:
- Overvaluation signals (e.g., "Volume/MCap declining despite price increase = weak hands")
- Negative momentum (decelerating gains or accelerating losses)
- Supply unlock risks (low circ/total ratio = future selling pressure)
- Competitive threats you know about
- Macro/regulatory headwinds

#### 4d. Risk Matrix
Rate each risk category as LOW / MEDIUM / HIGH with a one-line justification:
- **Market Risk**: Correlation to BTC, beta estimation
- **Liquidity Risk**: Volume/MCap ratio assessment
- **Dilution Risk**: Supply dynamics
- **Competition Risk**: How crowded is the niche?
- **Regulatory Risk**: Is this sector under scrutiny?
- **Technical Risk**: Smart contract risk, centralization concerns
- **Narrative Risk**: Is the hype sustainable?

#### 4e. Catalysts & Upcoming Events
List known upcoming events that could move the price:
- Protocol upgrades
- Token unlocks (flag as negative catalyst)
- Partnership announcements
- Regulatory milestones
- Market structure events (ETF, listings, etc.)

Note: Use your training knowledge here. Flag which catalysts are confirmed vs. speculated.

#### 4f. Comparable Analysis
Identify 3-5 most similar projects and compare:
- Market cap ranking relative to peers
- Valuation gap analysis (is it cheap or expensive vs. comps?)
- What would the price be if it matched Comp X's valuation?

#### 4g. Similar Past Analyses (Pattern Awareness)
🆕 Include the results from Step 0b. Format as:

```markdown
## 🔁 Similar Past Analyses
| Token | Category | Outcome | Sentiment | Date |
|-------|----------|---------|-----------|------|
| SOL   | layer-1  | ✅ Correct | Bullish | 2026-03-15 |
| AVAX  | layer-1  | ❌ Wrong   | Bullish | 2026-03-01 |

**Pattern insight:** [summary from similar analysis output]
```

If no similar analyses exist, note: "No prior similar analyses — this is novel territory."

#### 4h. Time Horizon Commentary
- **Short-term (1-4 weeks)**: Momentum-driven outlook based on current trends
- **Medium-term (1-6 months)**: Catalyst-driven outlook
- **Long-term (6-24 months)**: Fundamental-driven thesis

#### 4i. Self-Critique (MANDATORY)
🆕 This section forces the system to question its own reasoning. For EVERY analysis,
include an honest self-assessment:

**Generate this section by answering these questions:**

1. **Key Assumptions**: What are the 2-3 biggest assumptions this thesis rests on?
   List them explicitly. If any assumption breaks, the thesis fails.

2. **Where I Could Be Wrong**: Identify the 1-2 most likely ways this analysis is
   wrong. Be specific — not generic. For example, not "market could crash" but
   "if BTC breaks below $60K, the correlation-adjusted downside for this token is 3x"

3. **Biggest Uncertainty**: What single unknown would most change this thesis if
   resolved? This is the one thing you wish you had data on.

4. **Confidence Justification**: Explain in plain language why the confidence score
   is what it is. Reference specific factors:
   - "I'm at 72% because L1 analyses have been 80% accurate, but this specific
     token has no history yet"
   - "Only 45% confidence because bullish L2 calls have a 40% failure rate"

5. **Risk Override Acknowledgment**: If Step 0b surfaced any risk overrides, address
   them directly:
   - "⚠️ WARNING: This setup has historically underperformed (3/5 similar failures).
     I've weighted the bear case more heavily as a result."
   - If no overrides: "No historical risk patterns triggered for this setup."

Format:
```markdown
## 🪞 Self-Critique

**Key Assumptions:**
1. [assumption 1]
2. [assumption 2]

**Where I Could Be Wrong:**
- [specific failure scenario]

**Biggest Uncertainty:**
- [the one thing that matters most]

**Confidence Justification:**
- [plain language explanation]

**Risk Override Status:** [CLEAR / CAUTION / WARNING / DANGER]
- [details if applicable]
```

### Step 5: Format the Report

Use the report template from `templates/thesis_report.md`. Output structure:

```markdown
# 🔬 Crypto Thesis: [TOKEN NAME] ([SYMBOL])
> Generated: [DATE] | Data Source: CoinGecko | Skill: crypto-thesis-engine v2.1.0

## 🎯 Confidence: X% (GRADE)
> [One-line reasoning for the confidence score]
> Similar past analyses: X/Y correct | Token history: Z analyses
> ⚠️ Risk Override: [CLEAR/CAUTION/WARNING/DANGER] — [one-line if not CLEAR]

## 📊 Market Snapshot
[market data tables]

## 📝 Overview
[narrative overview]

## 🐂 Bull Case
[numbered arguments]

## 🐻 Bear Case
[numbered arguments]

## ⚠️ Risk Matrix
[risk table]

## 🚀 Catalysts
[bullet list with confirmed ✅ / speculated 🔮 markers]

## 📐 Comparable Analysis
[peer comparison table]

## 🔁 Similar Past Analyses
[pattern awareness section]

## 🔭 Time Horizon
[short/medium/long-term outlook]

## 🪞 Self-Critique
[assumptions, failure scenarios, uncertainties, confidence justification]

## ⚖️ Verdict
[One-paragraph final synthesis. Not a recommendation — a structured opinion.]
[If risk overrides are active, explicitly state how they influenced the verdict.]

---
*⚠️ DISCLAIMER: This is an AI-generated research analysis, not financial advice.
Confidence score reflects historical accuracy, not prediction certainty.
Always do your own research (DYOR). Past performance does not guarantee future results.*
```

### Step 6: Save to Memory & Remember

After generating a report, record it in the learning memory:

**6a. Record the analysis:**
```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" record \
    --token <token_id> \
    --price <current_price> \
    --mcap <market_cap> \
    --momentum <momentum_signal> \
    --sentiment <bullish|cautiously_bullish|neutral|cautiously_bearish|bearish> \
    --categories <comma_separated_categories> \
    --rank <market_cap_rank> \
    --confidence <computed_confidence_score>
```

Determine `--sentiment` from your verdict:
- If verdict is clearly positive → `bullish`
- If positive with caveats → `cautiously_bullish`
- If balanced/mixed → `neutral`
- If negative with some upside → `cautiously_bearish`
- If clearly negative → `bearish`

**6b. Save the report to disk:**
```bash
mkdir -p ~/.hermes/thesis-reports
cat > ~/.hermes/thesis-reports/<token_id>_$(date +%Y%m%d_%H%M%S).md << 'REPORT'
[report content]
REPORT
```

**6c. Remind the user about feedback:**
After presenting the report, tell the user:
> "💡 To help me improve, tell me later whether this analysis was accurate:
> `/crypto-thesis-engine learn <token_id> correct` or
> `/crypto-thesis-engine learn <token_id> wrong`"

### Step 7: Handle Learn Command

When the user runs `learn <token_id> <outcome>`:

```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" learn \
    --token <token_id> \
    --outcome <correct|wrong> \
    --note "<optional user note>"
```

After recording, present a summary:
```markdown
## 📚 Learning Recorded
- **Token:** <token_id>
- **Outcome:** ✅ Correct / ❌ Wrong
- **Token accuracy:** X/Y (Z%)
- **Overall accuracy:** A/B (C%)
- **Note:** <user note if provided>
```

### Step 8: Handle Stats Command

When the user runs `stats`:

```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" stats
```

Present as a formatted dashboard:
```markdown
## 📊 Crypto Thesis Engine — Performance Dashboard

### Overall
| Metric | Value |
|--------|-------|
| Total Analyses | X |
| Outcomes Recorded | Y |
| Accuracy | Z% |

### Top Categories
| Category | Accuracy | Record |
|----------|----------|--------|
| layer-1 | 75% | 3/4 |
| defi | 60% | 3/5 |

### Top Tokens (by accuracy)
| Token | Accuracy | Record |
|-------|----------|--------|
| BTC | 80% | 4/5 |
| ETH | 66% | 2/3 |

### Most Analyzed
| Token | Count |
|-------|-------|
| BTC | 8 |
| ETH | 5 |
```

### Step 9: Handle Diagnose Command

🆕 When the user runs `diagnose`:

```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" diagnose
```

Present as a system health report:
```markdown
## 🩺 System Self-Diagnosis

### Health: [HEALTHY / NEEDS_ATTENTION / CRITICAL / INSUFFICIENT_DATA]

### Biases Detected
| Bias | Severity | Details |
|------|----------|---------|
| BULLISH_BIAS | HIGH | 80% of analyses are bullish, only 45% accurate |
| OVERCONFIDENCE | MEDIUM | High-confidence analyses only 55% accurate |

### Blind Spots (Weak Categories)
| Category | Accuracy | Record | Recommendation |
|----------|----------|--------|----------------|
| layer-2 | 33% | 1/3 | Lower confidence, be more conservative |

### Strengths
| Category | Accuracy | Record |
|----------|----------|--------|
| layer-1 | 85% | 6/7 |

### Confidence Calibration
| Level | Count | Actual Accuracy |
|-------|-------|-----------------|
| High (≥70%) | 5 | 60% |
| Mid (50-69%) | 8 | 75% |
| Low (<50%) | 2 | 50% |

### Recommendations
1. ⚠️ CRITICAL: Bullish predictions are wrong >50% of the time...
2. 📉 layer-2: Only 33% accuracy...
3. Consider being more selective about strong bullish verdicts
```

### Step 10: Handle Strategy Command

🆕 When the user runs `strategy`:

```bash
python3 "$(dirname "$0")/scripts/thesis_memory.py" strategy
```

Present as a **research performance self-assessment** (NOT investment advice):
```markdown
## 📊 Crypto Thesis Engine — Research Performance Analysis

> ⚠️ This is a self-assessment of analytical accuracy, NOT investment advice.
> Based on [N] analyses with outcomes | Overall accuracy: [X%]
> [N] patterns mined across momentum, sentiment, category, and mcap dimensions

### Executive Summary
- Strongest research areas: [categories] (historically accurate)
- Weakest research areas: [categories] (historically inaccurate — treat with skepticism)
- Most accurate momentum signal: [signal] ([X%] accuracy)
- Least accurate momentum signal: [signal] ([X%] accuracy)

### ✅ High-Accuracy Research Patterns
| Pattern | Accuracy | Record | Insight |
|---------|----------|--------|--------|
| layer-1 tokens + bullish verdict | 100% | 3/3 | Research is reliable here |
| mega-cap + accelerating momentum | 100% | 2/2 | Historically accurate |
| layer-1 tokens | 80% | 4/5 | Strong analytical track record |

### ❌ Low-Accuracy Research Patterns
| Pattern | Accuracy | Record | Insight |
|---------|----------|--------|--------|
| layer-2 tokens + bullish verdict | 0% | 0/3 | Research has been unreliable |
| mid-cap + bullish verdict | 0% | 0/3 | Analyses consistently inaccurate |
| cautiously_bullish verdict | 33% | 1/3 | Low analytical confidence |

### 🗒️ Research Quality Insights

**HIGH ACCURACY (reliable research areas):**
1. When: [condition] → [X%] accuracy over [N] analyses

**LOW ACCURACY (unreliable — extra skepticism needed):**
1. When: [condition] → only [X%] accuracy over [N] analyses

### 📊 Accuracy by Dimension

**Categories:**
| Category | Research Accuracy | Assessment |
|----------|-------------------|------------|
| layer-1 | 80% (4/5) | ✅ Reliable |
| layer-2 | 0% (0/4) | ❌ Unreliable |

**Momentum Signals:**
| Signal | Research Accuracy | Assessment |
|--------|-------------------|------------|
| ACCELERATING_UP | 67% (2/3) | ✅ Most accurate |
| STEADY_UP | 50% (1/2) | ⚠️ Inconclusive |

---
*⚠️ DISCLAIMER: This is a research performance self-assessment, not investment advice.
Patterns show where this system's analyses have been accurate or inaccurate.
Past analytical accuracy does not predict future market outcomes. Always DYOR.*
```

## Pitfalls

- **CoinGecko Rate Limits**: Free API allows ~10-30 calls/minute. The script handles
  this with exponential backoff, but if analyzing many tokens in a scan, pace yourself.
- **Stale Data**: CoinGecko data may be delayed by a few minutes. Note this in the report.
- **Token ID Mismatch**: If the user provides a ticker symbol (e.g., "ETH"), you need to
  resolve it to a CoinGecko ID (e.g., "ethereum"). The script has a built-in resolver,
  but ambiguous tickers (e.g., "LUNA") may need clarification.
- **Missing Fields**: Some smaller tokens won't have all data fields. Handle gracefully
  by noting "N/A" rather than crashing.
- **AI Hallucination Risk**: For the knowledge-based sections (catalysts, comparable
  analysis), be explicit about your confidence level. If you're unsure, say so.
- **Outcome Timing**: "Correct" vs "wrong" depends on timeframe. When learning, the user
  should specify what timeframe they're judging. Short-term analyses can be "wrong" even
  if the long-term thesis was right.
- **Cold Start**: When no history exists, confidence starts at 50% (baseline). It takes
  5-10 outcomes before confidence scores become meaningful. Tell the user this.
- **NOT Financial Advice**: Always include the disclaimer. This is research analysis,
  not a buy/sell recommendation.

## Verification

- Run `python3 scripts/fetch_market_data.py --token bitcoin --output json` and verify
  valid JSON output with all expected fields
- Run `python3 scripts/thesis_memory.py stats` and verify valid JSON output
- Run `python3 scripts/thesis_memory.py confidence --token bitcoin --categories layer-1`
  and verify confidence score between 20-95
- Confirm the report contains all sections including 🎯 Confidence and 🔁 Similar Past Analyses
- Verify derived metrics are mathematically correct (e.g., MCap/FDV ratio)
- Ensure no hallucinated price data — all numbers must come from the API output
- Check that the disclaimer is present
- Verify the learn command updates memory.json correctly
- Run `python3 scripts/thesis_memory.py diagnose` and verify JSON output with biases,
  blind_spots, strengths, and recommendations
- Run `python3 scripts/thesis_memory.py risk-check --token bitcoin --categories layer-1
  --sentiment bullish --momentum ACCELERATING_UP` and verify risk override output
- Confirm every report includes the 🪞 Self-Critique section with all 5 subsections
- Verify that risk_level DANGER/WARNING causes more conservative language in the verdict
- Run `python3 scripts/thesis_memory.py strategy` and verify best/worst patterns and
  actionable rules are generated from historical data

## Future Improvements

This skill is designed to evolve. Planned enhancements:
- **On-chain data integration**: TVL, active addresses, gas usage via DeFiLlama
- **Social sentiment**: Integrate LunarCrush or similar social metrics
- **Automatic outcome detection**: Poll price after N days and auto-evaluate
- **Portfolio thesis**: Analyze an entire portfolio as a cohesive investment thesis
- **Alert system**: Set up cron-based monitoring for thesis-breaking events
- **Auto-weight tuning**: Automatically adjust confidence weights based on calibration data
- **Category-specific models**: Different analysis frameworks for DeFi vs L1 vs Meme tokens
- **Contrarian mode**: When bias is detected, force-generate the opposite thesis as a check
