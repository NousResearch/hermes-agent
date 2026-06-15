---
name: ss-compliance-scraper
description: "Specialized regulatory compliance research for Secure Safer Insurance — powered by You.com APIs to scrape state DOI sites (NY, NJ, PA, MI), NAIC model acts, federal regulatory changes (HHS, DOL, FEMA, IRS), and industry compliance news for LAH and P&C lines."
version: 2.0.0
author: Hermes Agent via Rafiul
license: MIT
metadata:
  hermes:
    tags: [secure-safer, compliance, regulatory, insurance, DOI, NAIC, youcom]
    related_skills: [ss-research-engine, ss-seo-aeo-research]
---

# Secure Safer Compliance Scraper (You.com Powered)

## When to Use This Skill

Load this skill when:
- A regulatory or compliance question comes up about any Secure Safer market
- You need to check state DOI bulletins for recent changes
- Researching NAIC model acts and adoption status
- Investigating federal changes (HHS, DOL, FEMA, IRS) affecting insurance
- Preparing a compliance brief for internal use or client education
- Fact-checking a claim against actual regulatory sources

## You.com API Integration

Uses the same You.com stack as the research engine:

| API | Usage |
|-----|-------|
| **Search API** | Find DOI bulletins, NAIC docs, industry news via site: operators |
| **Contents API** | Fetch full regulatory text as Markdown for deep analysis |
| **Live News** | Daily compliance news alerts for all 4 states |
| **Research API** | Complex regulatory synthesis (e.g., "Compare NY and NJ workers comp requirements") |

## Regulatory Source List

### State DOI (Primary Markets)

**New York** — You.com Search: `site:dfs.ny.gov [topic] [year]`
- NY DFS Insurance Division circular letters
- NY Insurance Regulations
- Market conduct exam reports

**New Jersey** — You.com Search: `site:nj.gov insurance [topic] [year]`
- NJ DOBI bulletins and orders

**Pennsylvania** — You.com Search: `site:insurance.pa.gov [topic] [year]`
- PA Insurance Department bulletins

**Michigan** — You.com Search: `site:michigan.gov insurance [topic] [year]`
- MI DIFS regulatory actions

### Federal Regulatory

**HHS (Health — LAH side)**
- You.com Search: `site:cms.gov [topic] [year]`
- ACA marketplace updates, MLR rebates, essential health benefits

**DOL (Benefits — LAH side)**
- You.com Search: `site:dol.gov [topic] [year]`
- ERISA, fiduciary rule updates, COBRA, disability benefits

**FEMA (Property — P&C side)**
- You.com Search: `site:fema.gov flood insurance [topic] [year]`
- NFIP updates, flood zone map changes

**IRS (Tax-advantaged products — LAH side)**
- You.com Search: `site:irs.gov HSA [topic] [year]`
- HSA contribution limits, FSA rules

### Industry Standards

**NAIC** — You.com Search: `site:naic.org [topic] model act`
- Model acts and model regulations
- Committee activities and adoption tracking

**AM Best** — You.com Search: `site:ambest.com [segment] outlook [year]`
- Market segment reports
- Financial rating changes

### Industry Publications (Compliance Focus)

- `site:insurancejournal.com compliance [state]`
- `site:propertycasualty360.com regulation [state]`
- `site:insurancebusinessamerica.com regulatory [state]`
- `site:thinkadvisor.com compliance LAH`

## Research Workflow

### Daily Compliance Scan (You.com Live News — 5 min)
1. Run You.com Live News for each state: `"[state] insurance regulation [year]" freshness=day`
2. Run: `"[state] department of insurance bulletin" freshness=day`
3. Flag anything urgent to a compliance brief

### Deep Dive (30-60 min — triggered by event or request)
1. Identify all relevant sources (state + federal + industry)
2. You.com Search each source with site: operators
3. You.com Contents API for full regulatory text as Markdown
4. Optional: You.com Research API for cross-jurisdiction synthesis
5. Write structured Compliance Brief
6. Save to vault: `_research/compliance-briefs/[topic]-YYYY-MM-DD.md`

### Complex Regulatory Research (You.com Research API)
For questions like:
- "What are the NAIC model acts adopted by NY in 2026?"
- "Compare workers compensation requirements across NY, NJ, PA, and MI"
- "What federal HHS changes affect home care agency insurance in 2026?"

Use You.com Research API (Standard effort) — it autonomously searches, reads, cross-references, and returns a cited answer.

## Compliance Brief Template

```markdown
## Compliance Brief: [Topic]
**Date:** YYYY-MM-DD
**Source:** [URL]
**Jurisdiction:** [NY / NJ / PA / MI / Federal / Multi-state]
**Line of Business:** [LAH / P&C / Both]

### Key Change
[1-2 sentence summary]

### Effective Date
[When does this take effect]

### Impact on Secure Safer
[What this means for the agency]

### Action Required
[What needs to happen]

### Source URL
[Link]
```

## Priority Topics by Market (2026 Context)

### NY
- NY DFS climate risk and flood insurance circulars
- NY auto insurance reform updates
- NY workers comp rate filings
- NY Paid Family Leave updates
- NY commercial auto insurance availability

### NJ
- NJ auto insurance rate filings and market conduct exams
- NJ workers comp classification updates
- NJ home care licensing requirements

### PA
- PA auto insurance medical benefit changes
- PA workers comp hearing calendar updates
- PA home care insurance requirements

### MI
- MI auto reform continued implementation
- MI workers comp dispute resolution updates
- MI No-Fault fee schedule changes

## Cost

Part of the You.com research budget (~$2.84/month for all research).

## Pitfalls

- **Don't confuse states** — always verify jurisdiction before applying a regulation
- **Date matters more than anything** — regulatory info is time-sensitive; always include effective dates
- **Don't quote secondary sources** — always trace back to the .gov or official document
- **Compliance ≠ marketing** — compliance briefs are factual, neutral, precise
- **Not all DOI bulletins affect agents** — some are consumer-facing only; distinguish carefully
- **NAIC model acts are NOT law** — only adopted versions have force. Check adoption status per state
- **You.com Contents API is your best friend for regulatory text** — it returns clean Markdown, perfect for analysis
