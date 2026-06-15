---
name: ss-seo-aeo-research
description: "SEO and AEO (Answer Engine Optimization) research for Secure Safer Insurance — SERP analysis, featured snippet opportunities, competitor content audits, EEAT requirements, and local market keyword research across NY, NJ, PA, MI."
version: 1.0.0
author: Hermes Agent via Rafiul
license: MIT
metadata:
  hermes:
    tags: [secure-safer, seo, aeo, content, serp, featured-snippets, eeat]
    related_skills: [ss-research-engine, ss-compliance-scraper]
---

# Secure Safer SEO/AEO Research Skill

## When to Use This Skill

Load this skill when:
- Researching keywords and search intent for a content topic
- Analyzing SERP features (featured snippets, people also ask, knowledge panels)
- Auditing competitor content for SEO structure and quality
- Planning content that targets featured snippets (AEO)
- Determining EEAT requirements for YMYL insurance content
- Looking for local SEO opportunities in NY, NJ, PA, MI markets

## Research Framework

### Step 1: Keyword Cluster Identification

web_search for:
```
site:google.com [insurance topic] [state]
"best [insurance type]" [state]
"how much does [insurance type] cost" [state]
"[state] [insurance type] requirements"
```

Identify:
- Primary keyword (high volume, commercial intent)
- Secondary keywords (long-tail, informational intent)
- Question-based keywords (PAA targets)
- Local modifiers (city, county, state names)

### Step 2: SERP Landscape Analysis

For each primary keyword, web_search to determine:

**SERP Feature Detection:**
- Featured snippet present? Type: paragraph / list / table / video
- "People also ask" questions (capture all)
- Local pack showing? (3-pack for location-based queries)
- Knowledge panel present?

**Competitor Analysis:**
- Top 3 ranking URLs (titles, domains, page types)
- Content format (blog post, guide, landing page, tool/calculator)
- Word count estimate
- Readability assessment
- Internal linking patterns

### Step 3: AEO Opportunity Mapping

For featured snippet targeting:

**Snippet Formats by Intent:**
| Intent | Best Snippet Format | Example |
|--------|-------------------|---------|
| Definition / What is X | Paragraph (40-60 words) | "What is commercial auto insurance?" |
| How-to / Steps | Ordered list (3-7 steps) | "How to get workers comp in NY" |
| Comparison | Table (2-4 columns) | "NY vs NJ auto insurance requirements" |
| Requirements / Checklist | Bullet list | "NY landlord insurance requirements" |
| Cost / Price | Table with ranges | "Average cost of homeowners insurance in NY" |

**Capture PAA questions** that:
1. Have NO good answer (competitor page doesn't address directly)
2. Secure Safer has unique authority on (NY-specific, niche LoB)
3. Have high search volume potential (Google Trends data)

### Step 4: EEAT Requirements Analysis

For YMYL (Your Money Your Life) insurance content:

**Experience:**
- Does agent/agency have hands-on experience with this LoB?
- Case studies, client testimonials, real examples needed
- Source: agency expertise, claims handling experience

**Expertise:**
- Author credentials needed: CIC, CPCU, CRM, CRM, agency principal
- Industry certifications and licenses
- Source: LinkedIn, agency bio pages, professional associations

**Authoritativeness:**
- Backlinks to agency site (NAIC, DOI, Better Business Bureau)
- Industry citations (Insurance Journal, local media quotes)
- Social proof (Google reviews, client ratings)

**Trustworthiness:**
- Secure HTTPS, clear privacy policy, about us page
- Transparent contact info, physical address
- Clear disclosure of licensing and jurisdictions

### Step 5: Local SEO Assessment

For each Secure Safer market:

**NY-specific:**
- "insurance agent [city] NY" (Albany, Buffalo, NYC, Rochester, Syracuse)
- "NY [insurance type] laws [year]"
- "NY DFS regulation [topic]"

**NJ-specific:**
- "insurance agent [city] NJ" (Newark, Jersey City, Trenton, Paterson)
- "NJ [insurance type] requirements"

**PA-specific:**
- "insurance agent [city] PA" (Philadelphia, Pittsburgh, Harrisburg)
- "PA [insurance type] laws"

**MI-specific:**
- "insurance agent [city] MI" (Detroit, Grand Rapids, Lansing)
- "Michigan auto insurance reform [year]"

### Step 6: Content Gap Analysis

Compare Secure Safer's existing content against competitors:

**Gap Types:**
1. **Complete gap** — topic not covered by anyone (high opportunity)
2. **Quality gap** — covered poorly by competitors (medium opportunity)
3. **Angle gap** — covered but from wrong angle (e.g., national vs local)
4. **Format gap** — covered as blog but not as video/infographic/tool
5. **Freshness gap** — covered but outdated (regulatory changes)

## Output Format

Save to vault: `_research/seo-analysis/[keyword]-YYYY-MM-DD.md`

Use the SEO AEO Analysis template at `_templates/SEO AEO Analysis.md`.

## Scoring Criteria for SEO/AEO Opportunities

| Score | Opportunity Level | Criteria |
|-------|-----------------|----------|
| 1-3 | Low | High competition, low volume, no local angle |
| 4-6 | Medium | Some competition, moderate volume, decent local angle |
| 7-8 | High | Low competition, high volume, strong local angle |
| 9-10 | Priority | Near-zero competition, insurance-specific, Secure Safer has unique authority |

## Pitfalls

- **Don't target national keywords** — Secure Safer serves specific states; rank for "NY workers comp insurance" not "workers comp insurance" (too competitive)
- **Don't ignore PAA** — "People also ask" questions ARE the content roadmap; answer them and you get featured snippets
- **Don't write for SEO alone** — EEAT matters more for YMYL content; thin SEO-optimized content gets de-ranked
- **Local modifiers beat generic** — "homeowners insurance Buffalo NY" converts better than "homeowners insurance" for a local agency
- **Update dates matter** — insurance regulations change yearly; outdated content damages EEAT
