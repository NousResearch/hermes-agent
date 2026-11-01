---
name: unified-digest-themes
description: "Canonical cross-platform theme taxonomy for all digest pipelines (news, social, papers)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Digest, Taxonomy, Classification, News, Social, Papers, Categorization]
    related_skills: [arxiv, blogwatcher, hn-brief-digest, jargon, youtube-content]
---

# Unified Digest Themes

A **single, portable taxonomy** for categorizing content across any digest pipeline — news digests, social-media summaries, paper roundups, blog monitoring, or mixed-source briefings. This is the canonical theme set used for classification, deduplication, and downstream routing.

No system paths, no agent-specific configuration, no hard-coded infrastructure references. Drop this taxonomy into any project that needs consistent content categorization.

## Theme Table

| # | Theme | Description | Typical Sources |
|---|-------|-------------|-----------------|
| 1 | **AI & Machine Learning** | Model releases, training advances, benchmarks, alignment, inference optimization, agent frameworks, LLM tools | Papers, tech news, X posts, blog posts |
| 2 | **Security & Privacy** | Vulnerabilities, breaches, zero-days, cryptography, surveillance, data protection policy | Security blogs, news, CVE feeds, government advisories |
| 3 | **Policy & Regulation** | Government actions, legislation, court rulings, regulatory frameworks, compliance requirements | News, government sites, think-tank reports |
| 4 | **Science & Health** | Biomedicine, climate science, physics, astronomy, public health, epidemiology, drug development | Academic papers, science news, CDC/WHO, journals |
| 5 | **Business & Markets** | Company earnings, M&A, startups, funding rounds, stock/commodity markets, supply chains | Financial news, SEC filings, Crunchbase, company blogs |
| 6 | **Culture & Society** | Arts, entertainment, sports, social trends, education, lifestyle, diversity & inclusion | General news, social media, opinion pieces, cultural criticism |
| 7 | **Technology & Infrastructure** | Hardware launches, networking, cloud platforms, dev tools, programming languages, databases, DevOps | Tech blogs, vendor announcements, GitHub releases, Stack Overflow |
| 8 | **Geopolitics & World Affairs** | International relations, conflicts, diplomacy, trade wars, elections, sanctions, foreign policy | News wire services, government briefings, think tanks, embassy statements |
| 9 | **Environment & Energy** | Climate policy, renewable energy, natural disasters, biodiversity, pollution, sustainability reports | Environmental news, IPCC/UNEP, energy industry reports |
| 10 | **Open Source & Community** | Project releases, governance debates, foundation news, licensing changes, community health, contribution trends | GitHub Trends, foundation blogs, mailing lists, developer forums |

## Overlap-Resolution Guide

Content often spans multiple themes. Use these tiebreakers when a single item could reasonably fit two or more categories:

### AI × Technology & Infrastructure
- **Clarifying question**: Is the item about the *model/capability itself* (→ AI) or the *platform/tooling around it* (→ Technology)?
- If it announces a new model, training technique, or benchmark → **AI & Machine Learning**
- If it discusses deployment, hosting, MLOps, or agent frameworks (e.g., Docker Compose for LLMs) → **Technology & Infrastructure**
- *Example*: "Llama 3.1 405B release" → AI. "Kubernetes operators for model serving" → Technology.

### Security × Policy & Regulation
- **Clarifying question**: Is the focus on the *technical vulnerability* (→ Security) or the *government response* (→ Policy)?
- Technical disclosure (CVE, exploit PoC, breach postmortem) → **Security & Privacy**
- Legislation, regulatory fine, or court ruling (e.g., GDPR enforcement, CFAA ruling) → **Policy & Regulation**
- *Example*: "SSH vulnerability disclosed" → Security. "EU Digital Services Act enforcement" → Policy.

### Business × Culture & Society
- **Clarifying question**: Is the primary frame *financial/corporate* or *societal/cultural*?
- Earnings reports, acquisitions, funding rounds → **Business & Markets**
- Cultural impact of a technology, workplace diversity studies, social-media trends → **Culture & Society**
- *Example*: "OpenAI raises $6.6B" → Business. "How social media affects teen mental health" → Culture.

### Policy × Geopolitics
- **Clarifying question**: Is the actor a *domestic regulator/legislature* or a *foreign state/coalition*?
- Domestic lawmaking, agency rulemaking, local regulation → **Policy & Regulation**
- International treaties, cross-border conflicts, sanctions between nations → **Geopolitics & World Affairs**
- *Example*: "US Senate passes AI bill" → Policy. "US-China chip export controls" → Geopolitics.

### Science × Health × Environment
- **Clarifying question**: Which domain does the *primary finding* live in?
- Biomedical discovery, clinical trial results, epidemiology → **Science & Health**
- Climate-science paper, energy policy research, conservation study → **Environment & Energy**
- When both are equally strong, prefer **Science & Health** (the broader umbrella).
- *Example*: "New mRNA vaccine trial results" → Science & Health. "IPCC climate report" → Environment.

### Open Source × Technology
- **Clarifying question**: Is the focus on the *project's community/governance* or the *technical artifact itself*?
- New project announcement, license change debate, foundation news, contributor metrics → **Open Source & Community**
- Technical deep-dive, performance benchmarks, architecture comparison → **Technology & Infrastructure**
- *Example*: "Redis changes license to SSPL" → Open Source. "Rust 1.80 compiler optimizations" → Technology.

### Fallback rule
When an item genuinely touches 3+ themes or the clarifying question doesn't help:
1. Assign the **single theme** that best describes the item's *headline/title*.
2. Tag secondary themes as comma-separated labels in a free-text field (not as an additional primary assignment).
3. If the item is truly unclassifiable (e.g., a meta item about the digest itself), use **"General"** as a catch-all.

## Usage

```python
# Pseudocode — adapt to your pipeline's language and data model.
# The taxonomy is language-agnostic; no imports, no dependencies.

THEMES = {
    1:  "AI & Machine Learning",
    2:  "Security & Privacy",
    3:  "Policy & Regulation",
    4:  "Science & Health",
    5:  "Business & Markets",
    6:  "Culture & Society",
    7:  "Technology & Infrastructure",
    8:  "Geopolitics & World Affairs",
    9:  "Environment & Energy",
    10: "Open Source & Community",
}

def classify(item_title, item_body, item_source):
    \"\"\"Return a primary theme ID and optional secondary theme IDs.\"\"\"
    # Implementation is pipeline-specific:
    # - keyword matching
    # - LLM classification
    # - source-based routing (e.g., all arXiv papers → AI & ML)
    # - hybrid approach
    pass
```

## Integration Notes

- **This taxonomy replaces ad-hoc tag sets.** If a pipeline uses custom categories (e.g., "Tech News" or "Research"), map those into this 10-theme set for consistency across sources.
- **Sub-themes are per-pipeline.** A pipeline can define sub-categories (e.g., under "AI & Machine Learning": "LLMs", "Computer Vision", "Robotics") without changing the top-level theme IDs.
- **Backward compatibility.** Adding a new theme (IDs 11+) is fine. Never rename, renumber, or delete an existing theme — that breaks cross-pipeline aggregation.
- **The overlap guide is the single source of truth** for ambiguous items. Don't establish pipeline-specific tiebreaker rules; use the ones here.
