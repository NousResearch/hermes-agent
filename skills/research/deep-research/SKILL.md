---
name: deep-research
description: |
  Autonomous deep research agent with multi-source intelligence. Takes any question and
  produces a comprehensive, citation-backed research report. Decomposes complex topics into
  sub-questions, searches the web and arXiv, extracts and cross-validates content from 10+
  sources, scores confidence levels, identifies contradictions, and synthesizes everything
  into a publication-ready report. Supports quick scans (2 min), standard research (8 min),
  and deep dives (15 min). Includes comparison mode, timeline analysis, and multi-format
  export (Markdown, JSON, HTML). Use for: market analysis, tech evaluation, literature
  review, competitive intelligence, due diligence, policy research, investment thesis.
version: 2.0.0
author: vominh1919
license: MIT
metadata:
  hermes:
    tags: [Research, Analysis, Web, arXiv, Reports, Citations, Intelligence]
    related_skills: [arxiv, blogwatcher, llm-wiki, github-contributions]
    prerequisites:
      commands: [python3]
      tools: [web_search, web_extract, execute_code, write_file]
---

# Deep Research Agent v2.0

> **Turn any question into a research-grade report in minutes.**

An autonomous research agent that decomposes complex questions, searches multiple sources,
cross-validates findings, and produces publication-ready reports with citations.

## Quick Start

```
/deep-research What are the best open-source AI agent frameworks in 2026?
```

**Output**: A structured 2000-word report with 15+ citations, confidence ratings, and recommendations.

## Research Modes

| Mode | Duration | Sources | Output | Use When |
|---|---|---|---|---|
| quick | 2-3 min | 5-8 sources | 800 words | Quick overview, initial screening |
| standard | 5-8 min | 10-15 sources | 2000 words | Most research needs |
| deep | 12-15 min | 20-30 sources | 4000 words | Investment decisions, academic work |
| compare | 5-8 min | 10-15 sources | Comparison matrix | Choosing between options |

Specify mode: `/deep-research [mode:deep] Your question here`

## Core Workflow

### Phase 1: INTELLIGENT DECOMPOSITION

Analyze the question and create a structured research plan:

```
INPUT: "Should we adopt Rust for our backend microservices?"

OUTPUT PLAN:
├── Q1: What are Rust's performance characteristics for web services?
├── Q2: How does Rust's ecosystem compare for backend development?
├── Q3: What is the learning curve and hiring difficulty?
├── Q4: Case studies: companies using Rust in production
├── Q5: Migration cost analysis from current stack
└── Q6: Risk assessment and mitigation strategies
```

Decomposition strategies:
- Technology: Performance → Ecosystem → Adoption → Migration → Risk
- Market: Size → Growth → Players → Trends → Regulation
- Academic: Background → Methods → Results → Limitations → Future
- Competitive: Features → Pricing → Market share → Strengths → Weaknesses

### Phase 2: MULTI-SOURCE RESEARCH

For each sub-question, execute:

```
1. web_search(sub_question, limit=5) — find authoritative sources
2. web_extract(top_3_urls) — extract key claims and data
3. Cross-validate: Do multiple sources agree?
4. Score: HIGH (3+ sources agree) / MEDIUM / LOW
5. Track source type: ACADEMIC | INDUSTRY | COMMUNITY | NEWS | OFFICIAL
```

### Phase 3: INTELLIGENCE SYNTHESIS

Don't just collect facts — analyze:
- CONSENSUS: All sources agree on X
- CONTRADICTION: Source A claims Y, Source B claims Z
- TREND: Pattern emerging since DATE
- GAP: No recent data on TOPIC
- IMPLICATION: This means X for the user

### Phase 4: REPORT GENERATION

Produce structured report. See templates/report.md for format.

## Execution Strategy

### Recommended: Batch Research via execute_code

Use execute_code to run multiple searches in one tool call:

```python
from hermes_tools import web_search, web_extract, write_file
import json

questions = [
    "Rust web framework performance benchmarks 2026",
    "Rust vs Go vs Java backend comparison",
    "Rust backend adoption case studies",
]

results = {}
for q in questions:
    try:
        search = web_search(query=q, limit=5)
        urls = [r.get("url", "") for r in search[:3] if r.get("url")]
        if urls:
            content = web_extract(urls=urls)
            results[q] = {"search": search, "content": content}
    except Exception as e:
        results[q] = {"error": str(e)}

write_file(path="research_raw.json", content=json.dumps(results, indent=2, default=str))
print(f"Researched {len(results)} questions")
```

### For Deep Research: Parallel Sub-agents

Use delegate_task to research sub-questions in parallel:

```python
delegate_task(
    tasks=[
        {"goal": "Research Rust performance for web", "toolsets": ["web", "file"]},
        {"goal": "Research Rust ecosystem and libs", "toolsets": ["web", "file"]},
        {"goal": "Research Rust adoption cases", "toolsets": ["web", "file"]},
    ]
)
```

## Quality Framework

### Source Quality Rating

| Grade | Criteria | Examples |
|---|---|---|
| A+ | Peer-reviewed, primary data | Nature, arXiv, official benchmarks |
| A | Expert analysis, official docs | Company engineering blogs, RFCs |
| B | Reputable publication, recent | TechCrunch, IEEE Spectrum |
| C | Community knowledge, opinion | Reddit, HN, Stack Overflow |
| D | Marketing, unverified | Vendor landing pages, ads |

### Confidence Scoring

| Level | Criteria |
|---|---|
| HIGH | 3+ A/B sources agree, no contradictions |
| MEDIUM | 2+ sources, mostly consistent |
| LOW | 1 source or significant disagreement |

### Claim Classification

Every key finding is tagged:
- [FACT] — Verifiable data point
- [OPINION] — Expert judgment or analysis
- [PREDICTION] — Future projection
- [CONSENSUS] — Multiple sources agree
- [DISPUTED] — Sources contradict

## Report Templates

### Standard Report (Default)
See templates/report.md

### Comparison Report
When comparing 2+ options:
```
/deep-research [mode:compare] Compare LangChain vs CrewAI vs AutoGen
```
Output includes: feature comparison matrix, scoring table, use-case fit analysis, recommendation.

### Timeline Report
For trend analysis:
```
/deep-research [mode:deep] Timeline of AI agent framework evolution 2023-2026
```
Output includes: chronological milestones, inflection points, trend extrapolation.

## Advanced Features

### 1. Continuous Research (with cronjob)
```
cronjob(name="weekly-ai-research", schedule="0 9 * * 1",
    prompt="Deep research: What were the most significant AI agent developments this week?")
```

### 2. Research to Action Pipeline
```
/deep-research Find the best open-source projects to contribute to
# Then use github-contributions skill to create PRs
```

### 3. Multi-session Research
```
Session 1: /deep-research Part 1: Market size and growth
Session 2: /deep-research Part 2: Competitive landscape
Session 3: /deep-research Part 3: Technology trends
Session 4: session_search → combine all parts → final report
```

## Execution Checklist

- [ ] Understand: Read question, identify scope and depth
- [ ] Plan: Create 5-8 sub-questions using decomposition strategy
- [ ] Search: web_search for each sub-question (use execute_code for batching)
- [ ] Extract: web_extract on top 3 URLs per sub-question
- [ ] Validate: Cross-check findings, note contradictions
- [ ] Score: Assign confidence levels and source grades
- [ ] Synthesize: Identify patterns, trends, gaps
- [ ] Write: Generate structured report with citations
- [ ] Save: write_file to save report
- [ ] Present: Show executive summary to user

## Limitations

| Limitation | Mitigation |
|---|---|
| Cannot access paywalled content | Search for open alternatives, note gaps |
| Web content may be outdated | Prefer recent sources (< 1 year) |
| Source quality varies | Use quality rating system, cross-validate |
| Language bias (English-heavy) | Note when non-English sources needed |
