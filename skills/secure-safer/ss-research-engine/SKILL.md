---
name: ss-research-engine
description: "Secure Safer Insurance Research Engine — multi-tier market intelligence powered by You.com APIs (Search, Research, Contents, Live News) that discovers consumer problems, compliance changes, SEO/AEO opportunities, and competitor gaps for Secure Safer Insurance."
version: 2.1.0
author: Hermes Agent via Rafiul
license: MIT
metadata:
  hermes:
    tags: [secure-safer, research, insurance, marketing, seo, youcom]
    related_skills: [ss-compliance-scraper, ss-seo-aeo-research, ss-content-generation]
---

# Secure Safer Research Engine (You.com Powered)

## When to Use This Skill

Load this skill when:
- Running market intelligence research for Secure Safer
- Compiling a Weekly Market Intelligence Report
- Researching a specific insurance topic or consumer problem
- Identifying content opportunities for lead generation
- Investigating compliance changes or regulatory updates
- Scoping competitor positioning and content gaps

**Output MUST use the Compact Research Brief format** (see Output Format section below). This is the feed for the content generation pipeline.

## Markets Covered

| State | Status |
|-------|--------|
| New York | Primary |
| New Jersey | Primary |
| Pennsylvania | Primary |
| Michigan | Primary |
| Florida | Expansion |
| Texas | Expansion |
| Ohio | Expansion |

## Lines of Business
Personal Auto | Homeowners | Landlord | Commercial Property | Commercial Auto
Workers Comp | General Liability | Professional Liability | Home Care Agency
OPWDD Services | Transportation | Limousine | TLC | Small Business

## Model Routing & Cost Optimization

This profile uses **OpenRouter multi-model routing** — different models for different task types to maximize quality per dollar.

| Role | Model | Cost/Run | When to Use |
|------|-------|----------|-------------|
| **Main conversation** | deepseek/deepseek-v4-flash | ~$0.0005 | Tier 1-2 research, simple queries, Reddit mining, search execution |
| **Fallback (auto)** | qwen/qwen3-next-80b-a3b-instruct | ~$0.003 | Main model fails or can't handle complexity (auto-escalates) |
| **Delegation (subagents)** | qwen/qwen3-next-80b-a3b-instruct | ~$0.003 | Deep research synthesis, regulatory analysis, multi-source cross-referencing |
| **Auxiliary tasks** | auto (cheapest capable) | varies | Vision, compression, session search — routed automatically |

### Tier-to-Model Mapping

| Research Tier | Recommended Model | Rationale |
|--------------|------------------|-----------|
| Tier 1 — Signal Detection | deepseek-v4-flash | Simple keyword queries, trend scanning. Cheap model is sufficient |
| Tier 2 — Sentiment Mining | deepseek-v4-flash | Reading Reddit threads, extracting complaints. No synthesis needed |
| Tier 3 — Industry/Regulatory | **delegate to qwen** | Cross-referencing multiple regulatory sources needs reasoning |
| Tier 4 — Competitor Intel | deepseek-v4-flash | Site: searches and gap analysis. Pattern matching, not deep reasoning |
| Weekly Report Compilation | **delegate to qwen** | Synthesizing 10+ findings into structured report needs stronger model |
| Compliance Deep Dive | **delegate to qwen** | Regulatory text analysis requires careful reasoning and citation accuracy |
| SEO/AEO Analysis | **delegate to qwen** | SERP interpretation, snippet format analysis benefits from stronger model |

### Cost Comparison

| Strategy | 22-day monthly cost |
|----------|-------------------|
| Everything on cheap model | ~$0.28 |
| Everything on strong model | ~$1.65 |
| **Multi-model routing (this profile)** | **~$0.52** |

Cost advantage: **~68% savings** vs running everything on the strong model.

## You.com API Integration

The research engine uses **You.com** as its primary data source via the You.com MCP server.

| API | Cost | What It Does |
|-----|------|-------------|
| **Search** | $5/1k calls | Real-time web + news, 1-100 results, LLM-ready JSON |
| **Contents** | $1/1k pages | Clean HTML/Markdown from any URL |
| **Research** | $12/1k calls (Lite+) | Multi-step reasoning, cross-references, cites sources |
| **Live News** | Included w/ Search | News-specific results |

## Research Methodology

### Tier 1 — Signal Detection (trending demand)

Run these FIRST to identify what people are searching for:

1. **You.com Search** — web search with LLM-optimized results:
   - Query: "why is [insurance type] so expensive [state]"
   - Query: "do I need [insurance type] for [situation]"
   - Query: "[state] insurance requirements [business/home/auto]"
   - Use `count=20`, `freshness=week` for trending topics

2. **You.com Live News** — news-specific:
   - Query: "[state] insurance regulation changes [year]"
   - Query: "[state] department of insurance [current year]"
   - Query: "insurance rate increase [state] [year]"
   - Use `freshness=day` for breaking news

### Tier 2 — Consumer Sentiment Mining

Find real questions, complaints, and pain points:

1. **You.com Search** with site:reddit.com operator:
   - "r/insurance [state] [line] help"
   - "r/landlord [state] insurance"
   - "r/truckers insurance"
   - "r/homeowners insurance nightmare"
   - "r/smallbusiness insurance [state]"

2. **You.com Contents API** — full thread content for high-engagement Reddit posts

3. **You.com Search** for consumer complaints:
   - "[insurance type] complaints [state]"
   - "worst insurance companies [state]"
   - "[state] insurance rate increase [year]"

### Tier 3 — Industry & Regulatory

Cross-reference findings with authoritative sources:

1. **You.com Search** with site: operators:
   - "site:insurancejournal.com [topic] [state]"
   - "site:propertycasualty360.com [topic]"
   - "site:insurancebusinessamerica.com [topic]"
   - "site:dfs.ny.gov insurance [topic]" (NY)
   - "site:nj.gov insurance [topic]" (NJ)
   - "site:insurance.pa.gov [topic]" (PA)
   - "site:michigan.gov insurance [topic]" (MI)
   - "site:naic.org [topic]"

2. **You.com Contents API** — full text of DOI bulletins, NAIC model acts, industry articles

3. **You.com Research API** (Lite) — for complex regulatory synthesis

### Tier 4 — Competitor Intelligence

1. **You.com Search** with site: operators:
   - "site:policygenius.com [topic]"
   - "site:goosehead.com [topic]"
   - "site:brightway.com [topic]"
   - "site:thezebra.com [topic]"
   - "site:selectquote.com [topic]"
   - "site:coverhound.com [topic]"

2. **You.com Contents API** — full competitor page content, compare structure and readability

### Deep Research Mode (You.com Research API)

For high-value topics:
1. Submit comprehensive question
2. API autonomously searches → reads → cross-references → synthesizes
3. Returns cited Markdown answer
4. Save directly to vault

## Output Format — Compact Research Brief

**This is the only output format for research.** Every research finding MUST be condensed into this single block. The compact brief is designed to feed directly into the content writing pipeline (ss-content-generation skill).

```
TOPIC: [One-line topic]
AUDIENCE: [Who this affects]
CONCERN: [The actual fear/pain point in plain English — 1-2 sentences max]
SEARCH INTENT: [informational / commercial / transactional / navigational]
SOURCE: [URL of the key source]
ADDITIONAL SOURCES: [URL2, URL3]

--- KEYWORDS ---
PRIMARY: [main keyword]
SECONDARY: [kw2, kw3, kw4]
LONG-TAIL: ["question phrase", "question phrase"]
AEO TARGET: [Featured snippet question to answer]

--- WORD TEMPERATURE ---
HOT WORDS (urgent/scary): word1, word2, word3
WARM WORDS (concerned): word1, word2, word3
NEUTRAL WORDS (informational): word1, word2, word3
COLD WORDS (reassuring): word1, word2, word3

--- COMPETITOR ANGLE ---
[1-2 sentences on how competitors cover this or don't]
CONTENT GAP: [What Secure Safer can say that competitors miss]

--- SELLING POINTS ---
• [One benefit per bullet]
• [Second benefit]
• [Third benefit]

--- RISKS OF NOT ACTING ---
[One sentence — worst case]
```

### Rules for Compact Briefs

- **TOPIC** is one line, no more
- **CONCERN** is 1-2 sentences in plain English — this is what the content will lead with
- **KEYWORDS**: primary goes in H1+first 60 words, AEO target drives the featured snippet answer
- **WORD TEMPERATURE**: HOT = 1-2 words used sparingly, WARM = main emotional driver (60%), COLD = solution zone
- **COMPETITOR ANGLE**: If no competitor covers it, note as gap
- **SELLING POINTS**: 3 bullets max, tight
- **RISKS**: One sentence, specific, actionable

### Example

```
TOPIC: NY commercial auto insurance minimum limits increase
AUDIENCE: NY-based businesses with commercial vehicles
CONCERN: "My premiums are going up and I don't know if my
          current coverage still meets the new minimums."
SEARCH INTENT: commercial
SOURCE: https://dfs.ny.gov/industry_guidance/circular_letters
ADDITIONAL SOURCES: https://insurancejournal.com/ny-commercial-auto-2026

--- KEYWORDS ---
PRIMARY: NY commercial auto insurance requirements
SECONDARY: NY commercial auto minimum limits, business auto insurance NY
LONG-TAIL: "what are the new commercial auto insurance limits in NY"
AEO TARGET: "What are the new commercial auto insurance limits in New York?"

--- WORD TEMPERATURE ---
HOT: fined, non-compliant, lapsed, lawsuit
WARM: confused, unexpected, increase, new requirement
NEUTRAL: minimum limits, bodily injury, property damage
COLD: compliant, covered, protected, reviewed

--- COMPETITOR ANGLE ---
Policygenius covers nationally — no NY-specific detail.
Local agents not publishing on this topic.
CONTENT GAP: No one explains what this means for specific NY business types.

--- SELLING POINTS ---
• Free review of existing policy against new limits
• Cover all NY commercial auto classes, not just standard fleets
• 20+ years in the NY market — saw this coming

--- RISKS OF NOT ACTING ---
Operating with insufficient limits means personal asset exposure
if you're in an accident.
```

## Scoring System

Every opportunity gets 5 scores (each 1-10):

| Score | How to Determine |
|-------|-----------------|
| Demand | You.com Search result volume, Reddit post frequency |
| Urgency | Emotional language in consumer posts, regulatory deadlines |
| Purchase Intent | Legally required? Risk of loss? Penalty? |
| SEO Opportunity | Competitor content quality, featured snippet gaps |
| Local Relevance | State-specific laws, local news coverage |

## Vault Integration

Save all research to:
```
/Users/rafiul/Documents/Social Media/Social Media App/_research/
```

Naming convention:
- Compact briefs: `_research/compact-briefs/[topic]-[date].md`
- Compliance briefs: `_research/compliance-briefs/[topic]-[date].md`
- SEO analyses: `_research/seo-analysis/[keyword]-[date].md`
- Competitor intel: `_research/competitor-intel/[competitor]-[date].md`
- News digests: `_research/news-digests/[topic]-[date].md`

## Weekly Report Production

1. Run You.com Search + Live News for each tier (focused scan)
2. Use You.com Research API for deep-dive on top 3 topics
3. Score top findings using 5-dimension system
4. Compile using `_templates/Weekly Market Intelligence Report.md` template
5. Save to `_research/` with date stamp
6. Present executive summary first

## Cost Estimates

~$2.84/month for 22 business days. $100 free You.com credit covers ~35 months.

## Pitfalls

- **Don't report news** — report what PEOPLE are worried ABOUT
- **Source everything** — .gov > .edu > trade > blog > commercial
- **State-specific context matters** — California problem ≠ NY problem
- **Never skip Tier 2** — best content ideas come from real consumer pain
- **Score honestly** — 1 means "nobody cares," don't inflate
- **Compact brief is the ONLY output format** — the content pipeline depends on this structure. Never return raw search results or verbose analysis.
- **You.com MCP `env:X` in headers doesn't resolve during sessions** — put the raw Bearer token in the MCP headers config, not the `env:YOUCOM_API_KEY` syntax.
- **Stale session files break file tools** — if `read_file`/`write_file`/`search_files` all fail with a stale path, run a search-and-replace across `~/.hermes/sessions/*.json` and `~/.hermes/processes.json` to fix the cwd references.
