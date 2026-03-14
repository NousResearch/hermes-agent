---
name: deep-research
description: Scientific research methodology for deep investigation. Plans research, auto-discovers available tools and skills, iteratively searches until saturation, evaluates credibility, performs contrarian analysis, verifies claims, and synthesizes structured reports.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research, analysis, synthesis, scientific-method, investigation]
    related_skills: [duckduckgo-search, arxiv, polymarket, blogwatcher]
---

# Deep Research — Scientific Investigation Skill

## What This Is

This is a research methodology skill. It teaches the agent HOW to think like a researcher — not just search and dump results, but plan, investigate, evaluate, reflect, and synthesize.

**Critical: The user's initial message is a starting point, not the complete research.** When the user provides info (a tweet, an article, a claim), use it as the seed for deep independent research. Verify their claims, find the original sources, look for additional context, find contradicting evidence, and bring back MORE than what they gave you. You are the expert researcher — they gave you a lead, now go investigate it properly before coming back with findings.

Use this for:
- Pre-writing research: "I want to write about X — what do I need to know?"
- Post-writing validation: "I wrote an article — what evidence supports my claims?"
- Competitive intelligence: "What's happening in X space right now?"
- Due diligence: "Is this claim true? What do primary sources say?"
- Deep dives: "I need to deeply understand X topic"

## Helper Scripts

This skill includes two helper scripts:

```bash
# Session manager and report generator
python3 SKILL_DIR/scripts/deep_research.py init "Is AI replacing programmers?"

# Quality scorer for proof-based saturation checks
python3 SKILL_DIR/scripts/quality_score.py assess --proof proof.json
python3 SKILL_DIR/scripts/quality_score.py checklist
```

`SKILL_DIR` is the directory containing this SKILL.md file.

Deep research sessions are stored in a writable state directory by default:
- `$HERMES_DEEP_RESEARCH_STATE_DIR` if set
- otherwise `$HERMES_HOME/state/deep-research`
- otherwise `~/.hermes/state/deep-research`

## Research Methodology

Real researchers don't search once and stop. They follow iterative cycles:

```
QUESTION → PLAN → SEARCH → EVALUATE → REFLECT → IDENTIFY GAPS → FILL GAPS → VERIFY → SYNTHESIZE
                                      ↑                                    |
                                      └────────── (if not saturated) ─────┘
```

Each cycle gets you closer to truth. You stop when you reach saturation — when new searches stop yielding new relevant information and all critical gaps are filled. Maximum 3 cycles to prevent bloat.

## Phase 1: Question Definition and Hypothesis

Before searching anything, define clearly:

1. **Core question**: What exactly am I trying to answer?
2. **Sub-questions**: What smaller questions need answering first?
3. **Hypothesis**: Form an initial hypothesis before searching. "I think X is true because Y." This is NOT a conclusion — it is a lens that makes your searches more targeted. You will revise or discard it as evidence accumulates.
4. **Claims to verify**: What specific claims need evidence?
5. **Contradictions to resolve**: What conflicts exist in my understanding?
6. **Success criteria**: How will I know the research is complete?

Output: A research plan with numbered questions, a hypothesis, and success criteria.

## Phase 2: Tool Discovery and Capability Mapping

MANDATORY. Never skip this phase because you "already know" the tools. Last time you did, you missed Polymarket and used the wrong search syntax.

### Step 1: Discover Skills

Call `skills_list()` to get all available skills. Filter for research-relevant ones:

- Any skill with tags: `research`, `news`, `web`, `scraping`, `search`, `social-media`, `browser`, `market-data`
- Any skill whose description mentions: search, fetch, scrape, extract, browser, social, news, prediction, market

For each relevant skill, call `skill_view(name)` to load its documentation and understand:
- What it does
- How to invoke it (exact commands/arguments)
- What inputs it needs
- Any environment requirements

Skills know their own CLI syntax. Read the SKILL.md and use it. Do not guess at flags.

### Step 2: Discover Built-in Tools

Check what built-in tools you have access to. For research, these are relevant:

- `browser_navigate` / `browser_snapshot` / `browser_click` — for interactive browsing of JS-rendered pages, paywalled content, or sites that block scraping
- `web_search` — built-in web search if available

### Step 3: Verify Optional Tools

```bash
# Check tools referenced by discovered skills (use names from SKILL.md)
for cmd in ddgs twitter scrapling curl python3 node; do
    command -v $cmd >/dev/null 2>&1 && echo "$cmd: available" || echo "$cmd: not found"
done
```

### Step 4: Build Capability Map

Create a capability map from what you discovered. Include EVERY research-relevant skill, even ones you think you won't need:

| Research Need | Available Tool/Skill | How to Use (from SKILL.md) |
|---------------|---------------------|---------------------------|
| General web search | [discovered skill] | [exact syntax from its docs] |
| News search | [discovered skill] | [exact syntax from its docs] |
| Social media / X | [discovered skill] | [exact syntax from its docs] |
| Article extraction | [discovered skill] | [exact syntax from its docs] |
| Prediction markets | [discovered skill] | [exact syntax from its docs] |
| Interactive browsing | [built-in tools] | browser_navigate, snapshot, click |
| Academic papers | [discovered skill or web fallback] | [exact syntax or site: filter] |

### Step 5: Identify Research-Type Specific Tools

Based on the research topic, identify which tools from the capability map are especially relevant:

- **Forecasting questions** -> prediction markets (Polymarket), analyst reports, historical base rates
- **Geopolitical/military** -> Wikipedia (often best aggregate for active conflicts), news search, expert commentary
- **Technical/academic** -> arxiv, academic search, primary papers
- **Company/market** -> financial news, SEC filings, analyst reports
- **Fact-checking** -> primary sources, official statements, data verification

### Fallback Priority

When primary search tools aren't available or fail:

1. **Skill-based tools** (loaded via skill_view) — always preferred
2. **Built-in tools** (browser_navigate, read_file, etc.) — reliable fallback
3. **Scrapling** (if installed) — for direct URL fetching
4. **curl** (always available) — last resort for HTTP requests

### Paywall/Access Escalation Ladder

When a key source is inaccessible:
1. Try `scrapling extract get` (basic fetch)
2. Try `scrapling extract fetch` with `--headless --disable-resources` (JS rendering)
3. Try `scrapling extract stealthy-fetch` with `--solve-cloudflare` (anti-bot bypass)
4. Try `browser_navigate` + `browser_snapshot` (interactive browser)
5. Search for the same story/topic from a different accessible source
6. Search for social media discussion about the article (often quotes key content)
7. If all fail, note as gap and move on — do not spend more than 2 attempts on one URL

For 404/dead links: skip immediately at step 1 and go to step 5 (find alternative source).

## Phase 3: Parallel Search Execution

Using the capability map from Phase 2, execute searches across multiple sources.

### Execution Pattern

For each research question, run multiple searches. Use two strategies depending on what you're searching with:

**Strategy A: Same tool, multiple queries — use execute_code batching**

When the same tool (e.g., your web search tool from the capability map) can answer multiple sub-questions, batch all queries into a single `execute_code` block. This is faster than multiple calls, keeps all results in one context, and lets you process/compare immediately.

Read the tool's SKILL.md for exact import/syntax, then batch like:

```python
# Example pattern — substitute YOUR tool's Python API from its SKILL.md
# (Shown here using ddgs as illustration; use whatever tool you discovered)
from ddgs import DDGS

queries = {
    "topic aspect 1": "query text 1",
    "topic aspect 2": "query text 2",
    "topic aspect 3": "query text 3",
}

for label, query in queries.items():
    print(f"=== {label} ===")
    with DDGS() as d:
        for r in d.text(query, max_results=5):
            print(f"  {r['title']}")
            print(f"  {r.get('href', 'N/A')}")
            print(f"  {r.get('body', '')[:200]}")
            print()
```

**Strategy B: Different tools needed — use delegate_task batch mode**

When sub-questions need different tools (one needs search, another needs scraping, another needs browser), use `delegate_task` with the `tasks` array for parallel execution.

**Strategy C: Deep article reading — use execute_code with your extraction tool**

For extracting full content from key URLs found during search, batch multiple URLs into one execute_code block using your extraction tool's API (loaded from its SKILL.md).

### Search Angles to Cover

For a typical research topic, ensure you search from at least these angles:

1. **Broad web search** — general coverage, multiple viewpoints
2. **News search** — recent developments, breaking news, timelines
3. **Social/sentiment** — expert opinions, prediction markets (if tools available)
4. **Deep article read** — extract full content from 3-5 most important URLs found
5. **Contrarian/expert depth** — analyst reports, primary data, alternative viewpoints

### Source Diversity Rule

For each sub-question, use at least 2 different source types. Never rely on a single search tool or a single source. Cross-reference everything.

### Handling Failures

- Tool returns empty results -> try alternative tool from your capability map
- Tool errors out -> fall back down the priority chain (skill -> built-in -> scrapling -> curl)
- Paywalled content -> follow the Paywall/Access Escalation Ladder from Phase 2
- 404/dead link -> skip immediately and find alternative, do not debug dead URLs
- Need to read specific URL that scraping cannot handle -> use browser_navigate

### Incremental Proof Building (Critical)

Do NOT wait until the end to build the proof object. Build it AS YOU RESEARCH. After each search round, update the proof:

1. **After each search batch**: add new sub-questions answered, new sources found, new source tiers
2. **After contrarian searches**: immediately add contrarian_searches entries with query/found_evidence/result_summary
3. **After reading key articles**: immediately rate claims as verified/supported/unverified
4. **After finding contradictions**: add to contradictions list with resolved status

Save proof to a local JSON file after each major search round, for example `proof.json`. This prevents memory loss between rounds and makes the final scoring step a formality.

Initial proof template to create at start of Phase 3:
```json
{
  "sub_questions": [{"question": "...", "answered": false, "sources": []}],
  "source_types_used": [],
  "source_tiers": {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0, "tier5": 0},
  "contrarian_searches": [],
  "claims": [],
  "hypothesis_revised": false,
  "hypothesis_original": "...",
  "contradictions": [],
  "gaps_identified": [],
  "search_rounds": 0,
  "ai_inference_labeled": true
}
```

## Phase 4: Critical Evaluation

Not all sources are equal. Evaluate each finding:

### Source Hierarchy
- **Tier 1 — Primary**: Original data, official reports, direct quotes, government statistics, company filings
- **Tier 2 — Expert analysis**: Peer-reviewed papers, recognized analysts (Goldman Sachs, ING, etc.), research firms (Windward, etc.)
- **Tier 3 — Credible reporting**: Reputable news outlets (Reuters, Bloomberg, AP, BBC), industry publications
- **Tier 4 — Secondary**: Blog posts, opinion pieces, regional news, think tank reports
- **Tier 5 — Unverified**: Social media, anonymous claims, user-generated content

### Red Flags to Watch For
- Claims without citations
- Single-source reporting
- Conflicts of interest (who benefits from this narrative?)
- Outdated information (check dates!)
- Emotional language over factual language
- Correlation presented as causation
- Survivorship bias

### Cross-Reference Rule
- 1 source saying something = interesting, needs verification
- 2 independent sources = credible claim
- 3+ independent sources = established fact
- Primary source contradicts secondary = trust primary
- Prediction markets (Polymarket, Metaculus) = crowd-sourced probability, valuable for forecasting

## Phase 5: Self-Reflection and Contrarian Search (Critical)

This is what separates research from searching. After initial search results are in, BEFORE synthesis:

### Mandatory Reflection Questions

1. **State your hypothesis explicitly.** Write it out: "Based on what I've found so far, I believe X because Y."

2. **What contradicts your hypothesis?** Look at your findings — what evidence pushes against what you initially believed? Flag contradictions explicitly.

3. **What would a skeptic say?** Write out the counter-argument to your thesis.

4. **Search for the counter-argument.** This is NOT optional. Run at least one search specifically designed to find evidence AGAINST your hypothesis:
   - "[topic] bear case"
   - "[topic] why wrong"
   - "[topic] alternative explanation"
   - "[topic] criticism"
   - "[forecast number] why will not happen"

5. **Assess confidence honestly:**
   - High: multiple primary sources, consistent data, prediction markets align
   - Medium: some supporting evidence, gaps remain, expert disagreement
   - Low: speculative, few sources, contradictions unresolved

6. **Note surprises.** What did you find that you did NOT expect? These are often where the real insight lives.

### Hypothesis Revision

After reflection and contrarian search, explicitly state:
- Original hypothesis
- Evidence for
- Evidence against
- Revised hypothesis (if changed)
- Confidence level

## Phase 6: Gap Analysis and Iterative Loop with Quality Scoring

This is where most AI research fails. It stops after one round. We force iteration using a proof-based quality score that the agent CANNOT fake — the score is computed from actual evidence, not vibes.

### The Scoring System

A script at `SKILL_DIR/scripts/quality_score.py` computes quality from a **proof object** (JSON). The agent provides evidence, the script computes the score. You cannot advance until the score hits 0.90 (SATURATED).

**8 dimensions, weighted:**
- Sub-Question Coverage (20%) — all questions answered with sources
- Contrarian Evidence (15%) — searched against your hypothesis
- Claim Verification (15%) — claims traced to sources, AI inference labeled
- Source Diversity (10%) — multiple source types used
- Source Quality (10%) — primary/analyst sources outweigh blogs
- Hypothesis Revision (10%) — updated based on evidence
- Contradiction Resolution (10%) — contradictions found and resolved
- Gap Analysis & Saturation (10%) — gaps identified, ranked, critical ones filled

### Step 1: Identify Specific Gaps

After Phase 5, write out every gap explicitly:

- **Unanswered sub-questions**: Which of your original questions still lack sufficient evidence?
- **Weak evidence areas**: Where do you only have Tier 4-5 sources?
- **Contradictions unresolved**: Where do credible sources disagree and you haven't found the resolution?
- **Missing data points**: Numbers, dates, quantities you need but don't have
- **Unverified claims**: Things that seem true but only have single-source support

### Step 2: Rank Gaps by Impact

Not all gaps matter equally. Rank each gap:
- **Critical**: Without this, the core answer is unreliable
- **Important**: Would significantly strengthen the conclusion
- **Nice to have**: Adds depth but doesn't change the answer

### Step 3: Targeted Gap-Filling Searches

For each Critical and Important gap, run a targeted search designed SPECIFICALLY to fill that gap:
- Data gaps -> search for the specific number/term
- Source quality gaps -> search for primary sources (official reports, government data)
- Contradiction gaps -> search for the resolution
- Expert opinion gaps -> search for analyst commentary or academic analysis

### Step 4: Build Proof Object and Score

After each round, build a proof JSON with:
- sub_questions: list with question/answered/sources for each
- source_types_used: list of tool types used
- source_tiers: count by tier (tier1=primary, tier2=analyst, tier3=reporting, tier4=secondary, tier5=unverified)
- contrarian_searches: list with query/found_evidence/result_summary
- claims: list with claim/status/sources (status: verified/supported/ai_inference/unverified)
- hypothesis_revised: true/false + original and revised text
- contradictions: list with description/resolved/resolution
- gaps_identified: list with gap/rank(filled=true/false)
- search_rounds: how many iterations performed
- ai_inference_labeled: true/false

Save to file, run:
```bash
python3 SKILL_DIR/scripts/quality_score.py assess --proof proof.json
```

Run `python3 SKILL_DIR/scripts/quality_score.py checklist` to see the full proof requirements.

### Step 5: Loop Until Saturated

**Score thresholds:**
- **>= 0.90 SATURATED**: Proceed to Phase 7 (Verification Pass)
- **0.70-0.89 GOOD**: Identify weak dimensions, run targeted gap-filling, rescore
- **0.50-0.69 ADEQUATE**: Significant gaps. Run full gap analysis round, rescore
- **< 0.50 INSUFFICIENT**: Major problems. Go back to Phase 3-5 with focus on weakest dimensions

**Maximum 3 rounds** to prevent bloat. If still below 0.90 after 3 rounds, proceed anyway but flag confidence as LOW and list specific deficiencies in final output.

**The score can go down.** If a new search round reveals problems (new contradictions, claims turn out false, source quality drops), the score decreases. This prevents the agent from just going through the motions.

## Phase 7: Verification Pass

Before synthesizing the final output, verify each key claim you plan to make:

1. **List every specific claim** in your draft answer (numbers, dates, names, causal relationships)
2. **Trace each claim** back to its source
3. **Rate each claim**: Verified (2+ independent sources), Supported (1 good source), Unverified (no direct source)
4. **Flag any claims that are AI inference** vs. directly sourced — label these as your analysis, not fact

This step prevents hallucination and overconfident assertions. If any key claim is Unverified, either search for a source or remove the claim.

## Phase 8: Synthesis

Structure the findings into a clear, authoritative output.

### Internal Report Structure (for your reasoning)

Use this structure internally. Adapt the output to the user's platform and request type (see Output Format Adaptation below).

```
# Research Report: [Topic]
Date: [timestamp]
Core Question: [original question]
Hypothesis: [initial -> revised]
Rounds: [number of search iterations performed]

## Executive Summary
[3-5 sentence summary of key findings]

## Key Findings
1. [Finding with source]
2. [Finding with source]

## Evidence by Sub-Question
[Sub-question 1]: Conclusion, evidence, confidence, remaining gaps
[Sub-question 2]: ...

## Hypothesis Evaluation
- Evidence supporting: [...]
- Evidence contradicting: [...]
- Contrarian search results: [...]
- Revised hypothesis: [...]

## Claim Verification
- Verified (2+ sources): [list]
- Supported (1 good source): [list]
- AI inference (labelled as analysis): [list]
- Unverified (removed from output): [list]

## Contradictions Found
- [Contradiction 1: Source A says X, Source B says Y]
- [Resolution or remaining uncertainty]

## Gap Iteration Log
Round 1 gaps identified: [...]
Round 2 gaps filled: [...]
Round 2 remaining: [...]
Saturation achieved: [Yes/No, why]

## Gaps and Limitations
- [What we couldn't verify and why we stopped looking]
- [What's outside our knowledge]

## Sources
[List with tier classification]
```

## Chain of Thought (Internal Process — Do Not Output)

Throughout research, maintain internal reasoning discipline. The user never sees this — it's your thinking process:

1. **Form hypothesis** before searching
2. **Search with intent** — each query should test or expand understanding
3. **Note surprises** — when results contradict expectations, that's where insight lives
4. **Cross-reference** — never rely on single sources
5. **Contrarian search** — actively find evidence AGAINST your hypothesis
6. **Revise hypothesis** — update as evidence accumulates
7. **Assess confidence** honestly

### Probability Estimation Methodology (for forecast/scenario questions)

When the research involves estimating probabilities (scenarios, forecasts, outcomes), do NOT just guess. Use this methodology:

1. **Check prediction markets** (Polymarket, Metaculus, Manifold) for crowd-sourced probabilities. These are real-money bets and often more accurate than expert opinions.

2. **Find analyst forecasts** — investment banks (Goldman, Morgan Stanley, JPMorgan), think tanks (RAND, CFR, IISS), and research firms publish probability-weighted scenarios.

3. **Use historical base rates** — what happened in similar situations? (e.g., "of the last 10 wars involving a great power, X ended in negotiation, Y ended in frozen conflict, Z ended in escalation")

4. **Apply the Bayesian framework** — start with base rate, adjust for specific evidence. State your adjustment reasoning.

5. **Label probability sources clearly**:
   - "X% per [Polymarket as of date]" = crowd-sourced real-money estimate
   - "X% per [Goldman Sachs]" = expert analyst estimate
   - "X% estimated based on [reasoning]" = your analysis (lower confidence)
   - "X% historical base rate from [source]" = reference class estimate

6. **Never present probabilities without methodology**. The user needs to know if it's a Polymarket number or a guess.

7. **State uncertainty ranges, not point estimates** — "35-45%" is more honest than "40%" when you're uncertain.

The deep research script (`SKILL_DIR/scripts/deep_research.py`) can maintain a lightweight research activity log for archival if useful, but the user-facing output should be the research report or briefing — never raw internal reasoning.

## Output Format Adaptation

Adapt output to the user's platform and request type:

### Military Briefing Format (default for most requests)

Use this format unless the user explicitly asks for something else. Inspired by military SITREPs — scannable, no fluff, right to the point:

```
SITREP: [TOPIC] — [TIMEFRAME]
[Date] | Confidence: [High/Medium/Low]

CURRENT STATE
- [What's happening right now with numbers]
- [Current price/data/status]
- [Key context, 1-2 sentences max per point]

KEY INTEL
1. [Most important finding with source]
2. [Second finding with source]
3. [Third finding with source]
4. [Fourth finding with source]

ESTIMATE / FORECAST (if applicable)
[Simple table or list of scenarios]
MOST LIKELY: [single line with number]

KEY VARIABLE
[The one thing that determines which scenario happens]

BOTTOM LINE
[What the user should do/expect in 1-2 sentences]
```

Rules for briefing format:
- No executive summary sections — just lead with the answer
- No "research methodology" explanation — user doesn't care
- No credibility score tables — weave sources into the text
- Use all-caps section headers (CURRENT STATE, KEY INTEL, etc.) for scannability
- Numbers inline, not footnoted
- 3-5 bullets per section max
- End with BOTTOM LINE, not "further research needed"

### Formal Report Mode

Full structured report with executive summary, findings by sub-question, contradictions, gaps, and source list. Use ONLY when:
- User explicitly asks for a report
- Research supports an article/whitepaper the user is writing
- Save to file and provide file path

### Quick Answer

Direct answer with key evidence, no report structure. Use for fact-checks and single-question lookups. Still be specific — name sources, cite numbers, state confidence.

Regardless of format: always be credible and authoritative. State what you know, what you're uncertain about, and what evidence supports each claim. Never hedge with "some sources say" without naming which sources and what they actually say.

### Platform Detection
- Telegram: briefing format, no tables (Telegram Bot API doesn't render markdown tables), no markdown headers
- Discord: briefing format, limited markdown ok
- CLI: any format, can save files
- Unknown: briefing format by default

### Telegram Table Workaround
Telegram doesn't render markdown tables. Use one of these instead:
- Simple lists with aligned spacing
- Pipe-separated lines (manual alignment)
- Dash-separated key-value pairs
- For complex data, generate an image file and send as media

### Streaming vs Monolithic Delivery

For complex multi-topic research, consider splitting into multiple messages:

**When to split:**
- Answer has 3+ distinct sections (like pathway analysis)
- Total output would be 500+ words
- User is on Telegram (hard to read walls of text on mobile)

**When to keep together:**
- Simple fact-check or quick answer
- User explicitly asks for a report
- Content is tightly interconnected and splitting would lose coherence

**Splitting strategy:**
1. Message 1: BOTTOM LINE + KEY INTEL (the answer, up front)
2. Message 2+: Supporting details (if user wants more)
3. Or: SITREP header + sections, keep under 400 words per message

The goal: user gets the answer immediately, details are available if they want them.

## Example Workflows

### Workflow 1: Pre-Article Research
User: "I want to write about why most AI agents fail in production"

1. Define question and hypothesis
2. Discover tools (skills_list, skill_view for relevant skills)
3. Batch search: web search + news search in one execute_code block using discovered tools
4. Social angle: search social media tool for expert opinions
5. Deep read: extract full content from 3-5 key articles using discovered extraction tool
6. Contrarian search: "AI agents succeed in production" / "why AI agents will work"
7. Synthesize into briefing

### Workflow 2: Fact-Check / Due Diligence
User: "I read on X that [claim]. Is this true?"

1. Parse the claim into verifiable sub-claims
2. Hypothesis: "This claim is likely [true/false] because [reasoning]"
3. Search for original source (not just people talking about the claim)
4. Search for supporting evidence
5. Search for contradicting evidence (mandatory)
6. Quick answer with confidence level and best evidence

### Workflow 3: Forecast / Estimation
User: "What will [X] cost/look like in [future date]?"

1. Current state: find current prices/data
2. Driving factors: find what influences the variable
3. Expert forecasts: search for analyst predictions
4. Prediction markets: check Polymarket or similar for crowd probabilities
5. Scenario analysis: present base/bull/bear cases with probability estimates
6. Contrarian: search for "why [forecast] is wrong"
7. Synthesize with explicit confidence levels per scenario

## Pitfalls

- **Don't confuse searching with researching** — searching finds information, researching finds truth
- **Don't stop at first result** — first page of Google isn't research
- **Don't ignore contradictions** — they're where the insight lives
- **Don't cherry-pick** — actively search for evidence against your hypothesis
- **Don't present correlation as causation** — flag it explicitly
- **Don't trust secondary sources over primary** — always trace back
- **Don't skip the reflection/contrarian phase** — it's the most important part
- **Don't over-research** — know when you have enough to answer the question
- **Don't guess at CLI syntax** — read the skill docs, use what they say
- **Don't write more tool calls when fewer will do** — batch queries in execute_code instead of separate calls
- **Don't stop at one search round** — run at least one gap analysis iteration before synthesis
- **Don't present AI inference as sourced fact** — if you're drawing a conclusion from multiple data points, label it as your analysis
- **Don't conflate "I searched" with "I found"** — a search returning empty results is itself a finding (this data may not exist publicly)

## Verification

A good research session produces:
1. Clear answer to the original question (with confidence level)
2. Supporting evidence with sources cited inline
3. Contradictions found (not hidden) and resolved where possible
4. Hypothesis stated and either supported or revised based on evidence
5. Contrarian evidence explicitly addressed
6. Gap analysis performed with at least one iteration loop back
7. Saturation achieved (Critical gaps filled with 2+ independent sources)
8. Key claims verified (traced back to sources, rated Verified/Supported/Unverified)
9. AI inference clearly separated from sourced fact
10. Gaps identified (what we still don't know, with justification for stopping)
11. Scenario analysis with probability estimates (for forecast-type questions)
12. Full source list with tier classification
