---
name: deep-research
description: Scientific research methodology for deep investigation with multi-agent debate, quality gates, evidence cards, forecast calibration, and auto-discovered capabilities. Plans research, discovers available tools, iteratively searches until saturation, enforces quality gates with rollback, performs structured debate analysis, and synthesizes verified reports.
version: 2.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research, analysis, synthesis, scientific-method, investigation, forecasting]
    related_skills: [duckduckgo-search, arxiv, polymarket, blogwatcher, scrapling]
---

# Deep Research v2.0 — Scientific Intelligence System

## What This Is

This is a complete research methodology skill. It teaches the agent HOW to think like a researcher AND enforces quality through structured processes.

**New in v2.0:**
- Multi-agent debate (FOR/AGAINST/ALTERNATIVE analysis)
- Quality gates with automatic rollback
- Evidence cards with bias detection
- Self-learning forecast calibration
- Auto-discovery of available tools
- PIVOT/REFINE decision loops

Use this for:
- Pre-writing research: "I want to write about X — what do I need to know?"
- Post-writing validation: "I wrote an article — what evidence supports my claims?"
- Competitive intelligence: "What's happening in X space right now?"
- Due diligence: "Is this claim true? What do primary sources say?"
- Forecasting: "Will X happen? What's the probability?"
- Deep dives: "I need to deeply understand X topic"

## Helper Scripts

This skill includes helper scripts in `scripts/`:

```bash
# Session manager and report generator
python3 scripts/deep_research.py init "Is AI replacing programmers?"

# Quality scoring with proof validation
python3 scripts/quality_score.py assess --proof proof.json

# Evidence cards with bias detection
python3 scripts/evidence_cards.py add --claim "X" --source "url"

# Quality gates with rollback
python3 scripts/gates.py all --proof proof.json

# Multi-agent debate
python3 scripts/debate.py init --hypothesis "X will happen"

# Forecast history and calibration
python3 scripts/forecast_history.py log --question "X?" --prob 65 --method polymarket

# Capability auto-discovery
python3 scripts/capability_map.py discover
```

State directory: `$HERMES_HOME/state/deep-research/` or `~/.hermes/state/deep-research/`

---

## Research Methodology

Real researchers don't search once and stop. They follow iterative cycles:

```
QUESTION → CAPABILITY_DISCOVER → PLAN → SEARCH → DEBATE → GATES → REFLECT → PIVOT/REFINE → VERIFY → SYNTHESIZE
                ↑                                                            ↓
                └────────────── (if gates fail, rollback to this point) ─────┘
```

---

## Phase 0: Capability Discovery (NEW)

MANDATORY. Before any research, discover what tools are available.

### Auto-Discovery

```bash
python3 scripts/capability_map.py discover
```

This returns:
- Available skills (deep-research, duckduckgo-search, polymarket, etc.)
- Available tools (ddgs, scrapling, curl, etc.)
- Built-in tools (browser_navigate, terminal, etc.)
- Capability index (what can be done with available tools)

### Build Capability Map

From the discovery output, build a capability map:

| Research Need | Available Tool | How to Use |
|---------------|----------------|------------|
| General search | [discovered] | [from skill docs] |
| News search | [discovered] | [from skill docs] |
| Prediction markets | [discovered] | [from skill docs] |
| Academic papers | [discovered] | [from skill docs] |
| Article extraction | [discovered] | [from skill docs] |

### Fallback Priority

When primary tools aren't available:
1. Skill-based tools (loaded via skill_view) — preferred
2. Built-in tools (browser_navigate, terminal) — reliable fallback
3. Python packages (scrapling, ddgs) — if installed
4. curl (always available) — last resort

### Paywall/Access Escalation Ladder

1. Try scrapling extract get (basic fetch)
2. Try scrapling extract stealthy-fetch (anti-bot bypass)
3. Try browser_navigate + browser_snapshot (interactive)
4. Search for same story from different source
5. Search for social media discussion
6. If all fail, note as gap and move on

---

## Phase 1: Question Definition and Hypothesis

Before searching, define clearly:

1. **Core question**: What exactly am I trying to answer?
2. **Sub-questions**: What smaller questions need answering first?
3. **Hypothesis**: Form an initial hypothesis. "I think X is true because Y."
4. **Claims to verify**: What specific claims need evidence?
5. **Success criteria**: How will I know research is complete?

### Initialize Debate

```bash
python3 scripts/debate.py init --hypothesis "I believe X will happen because Y"
```

This sets up the multi-agent debate framework.

---

## Phase 2: Initialize Evidence Cards

Create an evidence card deck:

```bash
python3 scripts/evidence_cards.py init --session "session-id"
```

Every piece of evidence will become a structured card with:
- Claim text
- Source URL and tier
- Verification status
- Bias flags (auto-detected)
- Debate analysis

---

## Phase 3: Parallel Search Execution

Using the capability map from Phase 0, execute searches across multiple sources.

### Execution Pattern

**Strategy A: Same tool, multiple queries — use execute_code batching**

```python
from hermes_tools import terminal

# Batch all searches into one execute_code block
queries = {
    "topic aspect 1": "query text 1",
    "topic aspect 2": "query text 2",
}

for label, query in queries.items():
    print(f"=== {label} ===")
    # Use discovered tool from capability map
```

**Strategy B: Different tools needed — use delegate_task batch mode**

When sub-questions need different tools, use parallel delegation.

### Add Evidence Cards

After each search, add evidence cards:

```bash
python3 scripts/evidence_cards.py add \
  --claim "X costs Y" \
  --source "https://..." \
  --type "primary" \
  --credibility 4
```

Bias flags are auto-detected based on source type.

---

## Phase 4: Multi-Agent Debate (NEW)

This is what separates research from searching. After initial evidence collection:

### Run Structured Debate

```bash
# Add arguments from each perspective
python3 scripts/debate.py argue --side proponent --argument "Evidence shows X because..." --evidence "url"

python3 scripts/debate.py argue --side skeptic --argument "However, Y contradicts this because..." --evidence "url"

python3 scripts/debate.py argue --side alternative --argument "Actually, Z might be the real cause..." --alternative-hypothesis "Z is happening because W"
```

### Debate Requirements

- Proponent must cite specific evidence
- Skeptic must search for counter-evidence
- Alternative must propose different explanation
- Minimum 2 arguments per side
- Evidence must be cited (not just claimed)

### Synthesize Debate

```bash
python3 scripts/debate.py synthesize
```

The agent must fill in:
- Strongest argument for hypothesis
- Strongest argument against hypothesis
- Key evidence on both sides
- Unresolved questions
- Whether hypothesis should be revised

---

## Phase 5: Quality Gates (NEW)

Gates enforce minimum standards. If a gate fails, the pipeline rolls back.

### Run All Gates

```bash
python3 scripts/gates.py all --proof proof.json
```

### Gate Definitions

| Gate | Pass Criteria | Rollback Target |
|------|--------------|-----------------|
| hypothesis_stated | Hypothesis explicitly stated | planning |
| evidence_diversity | 2+ source types used | search |
| source_quality | At least 1 Tier 1-2 source | search |
| claim_verification | 50%+ claims verified | gap_filling |
| contrarian_evidence | Searched against hypothesis | reflection |
| probability_grounding | Forecast tied to market/base rate | search |
| saturation | 80%+ critical gaps filled | gap_filling |
| freshness | Recent sources for time-sensitive topics | search |

### Gate Failure Handling

If a CRITICAL gate fails:
1. Note the failure and rollback target
2. Execute rollback (go back to earlier phase)
3. Fix the issue
4. Re-run gates

Maximum 2 rollback cycles to prevent infinite loops.

---

## Phase 6: PIVOT/REFINE Decision (NEW)

After debate and gates, make a decision:

### Decision Types

| Decision | When | Action |
|----------|------|--------|
| PROCEED | Gates passed, hypothesis supported | Continue to verification |
| REFINE | Minor gaps, hypothesis mostly supported | Fill gaps, strengthen evidence |
| PIVOT | Hypothesis contradicted by evidence | Formulate new hypothesis, restart debate |

### Record Decision

```bash
python3 scripts/debate.py verdict --findings "finding1|finding2|finding3"
```

### PIVOT Process

If pivoting:
```bash
python3 scripts/debate.py pivot --new-hypothesis "New hypothesis" --reason "Old hypothesis contradicted by X"
```

Maximum 2 pivots per research session.

---

## Phase 7: Verification Pass

Before synthesizing, verify each key claim:

1. List every specific claim in your draft
2. Trace each claim back to its source
3. Rate each claim: Verified (2+ sources), Supported (1 source), AI inference (your analysis), Unverified (remove)
4. Update evidence card verification status:

```bash
python3 scripts/evidence_cards.py verify --card-id "card-xxx" --status verified --notes "Cross-referenced with source Y"
```

---

## Phase 8: Synthesis

Structure findings into output.

### Briefing Format (Default)

```
SITREP: [TOPIC] — [TIMEFRAME]
[Date] | Confidence: [High/Medium/Low]

CURRENT STATE
- [What's happening with numbers]
- [Key context]

KEY INTEL
1. [Finding with source]
2. [Finding with source]
3. [Finding with source]

ESTIMATE / FORECAST (if applicable)
[Scenarios with probabilities]
MOST LIKELY: [single line]

KEY VARIABLE
[The one thing that determines outcome]

BOTTOM LINE
[Actionable recommendation]
```

### Telegram Rules

- NO markdown tables (use code blocks or lists)
- NO markdown headers (use ALL CAPS)
- Bold, italic, code blocks OK

---

## Forecasting-Specific Methodology

When the research involves estimating probabilities:

### Step 1: Check Prediction Markets

```bash
python3 scripts/capability_map.py recommend --type forecast
```

Use Polymarket, Metaculus, or similar for crowd-sourced probabilities.

### Step 2: Find Analyst Forecasts

Search for:
- Investment bank reports (Goldman, Morgan Stanley, JPMorgan)
- Think tank analyses (RAND, CFR, IISS)
- Research firm reports

### Step 3: Use Historical Base Rates

"What happened in similar situations?"

### Step 4: Apply Bayesian Framework

Start with base rate, adjust for specific evidence.

### Step 5: Log Forecast for Calibration

```bash
python3 scripts/forecast_history.py log \
  --question "Will X happen by Y date?" \
  --prob 65 \
  --method polymarket \
  --methodology "Base rate 40% + new evidence adjustment +25%" \
  --category "geopolitical"
```

### Step 6: Later, Resolve and Learn

```bash
python3 scripts/forecast_history.py resolve --id "f-xxx" --outcome true --lesson "Underestimated speed of escalation"
```

### Calibration Tracking

```bash
python3 scripts/forecast_history.py calibration
```

Returns:
- Brier score (lower = better calibrated)
- Calibration curve (predicted vs actual)
- Topic-specific biases
- Recommendations for improvement

---

## Quality Score System

The quality_score.py script enforces quality through proof objects.

### Proof Object Structure

```json
{
  "sub_questions": [
    {"question": "...", "answered": true, "sources": ["url1", "url2"]}
  ],
  "source_types_used": ["web_search", "news", "prediction_market"],
  "source_tiers": {"tier1": 2, "tier2": 3, "tier3": 1, "tier4": 0, "tier5": 0},
  "contrarian_searches": [
    {"query": "why X is wrong", "found_evidence": true, "result_summary": "..."}
  ],
  "claims": [
    {"claim": "X costs Y", "status": "verified", "sources": ["url1", "url2"]}
  ],
  "hypothesis_revised": true,
  "hypothesis_original": "I think X...",
  "hypothesis_revised_text": "I now think Y...",
  "contradictions": [
    {"contradiction": "Source A says X, Source B says Y", "resolved": true, "resolution": "..."}
  ],
  "gaps_identified": [
    {"gap": "...", "rank": "critical", "filled": true}
  ],
  "search_rounds": 2,
  "ai_inference_labeled": true
}
```

### Score Thresholds

- >= 0.90: SATURATED — proceed to output
- 0.70-0.89: GOOD — targeted gap-filling needed
- 0.50-0.69: ADEQUATE — significant gaps
- < 0.50: INSUFFICIENT — major improvements needed

---

## Pitfalls

- Don't confuse searching with researching
- Don't stop at first result
- Don't ignore contradictions
- Don't cherry-pick evidence
- Don't present correlation as causation
- Don't skip the debate phase
- Don't skip quality gates
- Don't present AI inference as sourced fact
- Don't guess at CLI syntax — read skill docs
- Don't present probabilities without grounding
- Don't pivot endlessly — max 2 pivots
- Don't skip calibration tracking for forecasts

---

## Verification Checklist

A good research session produces:

1. Clear answer with confidence level
2. Evidence cards with verification status
3. Debate synthesis (FOR/AGAINST/ALTERNATIVE)
4. Gates passed (or explicit gaps noted)
5. Quality score >= 0.70
6. Hypothesis stated and either supported or revised
7. Contrarian evidence addressed
8. AI inference clearly separated from fact
9. For forecasts: probability logged with methodology
10. Calibration recommendations if applicable