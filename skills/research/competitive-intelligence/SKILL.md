---
name: competitive-intelligence
description: "Crawl competitor websites and produce structured intelligence reports with SWOT analysis, comparison matrices, and actionable insights."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Competitive-Intelligence, Market-Research, Competitor-Analysis, Strategy, SWOT, Pricing, Positioning]
    related_skills: [research-paper-writing, blogwatcher]
---

# Competitive Intelligence

Crawl one or more competitor websites and produce a structured intelligence report covering company profile, products, pricing, positioning, technology, marketing strategy, SWOT analysis, a cross-competitor comparison matrix, strategic opportunities, and risk assessment.

## Quick Reference

| Step | Action | Tool / Script |
|------|--------|---------------|
| 1 | Accept URLs | (input parsing) |
| 2 | Validate URL & robots.txt | `python scripts/ci_recon.py validate <url>` |
| 3 | Crawl public pages | `web_extract(urls=[...])` |
| 4 | Extract & normalize findings | LLM analysis pass |
| 5 | Classify into categories | LLM classification pass |
| 6 | Identify competitive insights | LLM insights pass |
| 7 | Generate SWOT | LLM per-competitor pass |
| 8 | Build comparison matrix | LLM cross-competitor pass |
| 9 | Assign confidence levels | Annotation pass |
| 10 | Cite sources | Inline per finding |
| — | Save artifacts | `python scripts/ci_recon.py save_report <slug>` |

## Guardrails

Before doing anything else, confirm these constraints are met for every URL:

- **Public data only** — never attempt to log in, bypass paywalls, or access protected pages.
- **robots.txt respected** — run `python scripts/ci_recon.py validate <url>` first. If it exits non-zero, document the failure and skip that URL.
- **No authentication bypass** — if a page returns 401/403 or redirects to a login screen, treat it as inaccessible and document it.
- **Facts vs. inferences** — clearly separate direct evidence (High confidence) from indirect signals (Medium) and inferences (Low). Label every claim.
- **Flag missing data** — if a key category (e.g. pricing) cannot be found, explicitly state "Not publicly available."

---

## Step-by-Step Procedure

### Step 1 — Accept URLs

Parse the user's input. Require at least one URL. If none provided, ask:

```
Please provide one or more competitor URLs (space- or newline-separated).
```

Deduplicate and normalise: ensure each URL has a scheme (`https://`).

---

### Step 2 — Validate Each URL

For each URL, run the validation script. Example:

```bash
python scripts/ci_recon.py validate https://competitor.com
```

The script prints JSON:

```json
{"url": "https://competitor.com", "accessible": true, "robots_allows_crawl": true, "status_code": 200, "error": null}
```

**Decision logic:**

| accessible | robots_allows_crawl | Action |
|------------|---------------------|--------|
| true | true | Proceed to Step 3 |
| true | false | Skip; log "robots.txt disallows crawling" |
| false | — | Skip; log error from `"error"` field |

Keep a running `## Access Log` section that documents every outcome.

---

### Step 3 — Crawl Public Pages

For each valid URL, use `web_extract` to fetch these standard pages (skip any that return 404 or redirect to login):

```
/               (homepage)
/pricing
/features
/products
/solutions
/about
/team
/careers
/blog
/resources
/case-studies
/customers
/press
/newsroom
/partners
/integrations
/security
/enterprise
```

**Crawl strategy:**

1. Start with the homepage to discover actual path naming conventions (e.g. the site may use `/plans` instead of `/pricing`).
2. Use `web_search` with query `site:<domain> pricing` or `site:<domain> case studies` to locate pages that don't follow standard paths.
3. Record the exact URL of every page successfully fetched — these become your source citations.
4. **Watch for login redirects** — a 302 response is reported as accessible, but if the redirect destination is a login or paywall page, treat the content as inaccessible. Detect this by checking whether `web_extract` returns a login form or error page instead of the expected content, and mark that page as "Not publicly available."

**Example tool calls:**

```
web_extract(urls=["https://competitor.com", "https://competitor.com/pricing", "https://competitor.com/about"])
web_search(query="site:competitor.com pricing plans features")
```

---

### Step 4 — Extract and Normalize

Feed all collected content to the LLM with this extraction prompt for **each competitor**:

```
You are a competitive intelligence analyst. From the web content below, extract every piece of evidence for these categories. Be exhaustive but cite the source URL for each claim. If a category has no evidence, write "Not found."

Categories to extract:
1. Company overview (founding, size, funding, headquarters, leadership)
2. Products and services (names, descriptions, tiers)
3. Pricing (amounts, tiers, billing models, free trials, enterprise pricing)
4. Target industries (verticals explicitly mentioned or implied)
5. Target customer segments (SMB, mid-market, enterprise, job titles targeted)
6. Value propositions (explicit claims of benefit or differentiation)
7. Messaging themes (repeated language, tone, emotional hooks)
8. Feature sets (capabilities listed on product/features pages)
9. Technology stack indicators (badges, "built with", job postings for specific tech, integrations)
10. Partnerships (named partners, co-marketing, reseller networks)
11. Geographic coverage (regions, languages, data residency claims)
12. Hiring trends (open roles visible on careers page, hiring locations)
13. Recent announcements (press releases, blog posts dated within 90 days)

Web content:
[INSERT RAW CRAWLED CONTENT HERE]
```

Produce a structured JSON object per competitor keyed on the 13 categories.

---

### Step 5 — Classify Findings

Map the extracted data into these 9 report categories:

| Report Category | Populated from extraction categories |
|-----------------|--------------------------------------|
| Company Profile | 1, 11 |
| Products | 2, 8 |
| Pricing | 3 |
| Positioning | 5, 6, 7 |
| Technology | 9, 10 |
| Marketing Strategy | 7, 13 |
| Customer Segments | 4, 5 |
| Partnerships | 10 |
| Growth Signals | 12, 13 |

---

### Step 6 — Identify Competitive Insights

For each competitor, ask the LLM:

```
Based on the findings for [competitor], identify:
1. Unique selling propositions (what they claim no one else does)
2. Differentiators (features or claims that stand out)
3. Market gaps (what they don't cover or mention)
4. Feature advantages (where they appear stronger than typical)
5. Feature weaknesses (gaps in their feature set)
6. Pricing advantages (where their pricing model may be attractive)
7. Emerging strategic initiatives (signals of new direction: new hires, new integrations, recent messaging shifts)

Cite source URLs for each point.
```

---

### Step 7 — SWOT Analysis

Generate a SWOT table per competitor:

```
Using all evidence gathered for [competitor], produce a SWOT analysis.
- Strengths: internal positive attributes directly evidenced
- Weaknesses: internal gaps or limitations evidenced
- Opportunities: external factors they seem positioned to exploit
- Threats: external risks visible in their messaging or market signals

Keep each point to one sentence. Cite source URLs where possible.
```

---

### Step 8 — Cross-Competitor Comparison Matrix

Once all competitors are processed, build comparison tables covering:

**Features matrix** — rows = feature areas, columns = competitors. Values: ✓ (confirmed), ~ (partial/implied), ✗ (no evidence).

**Pricing matrix** — rows = pricing tiers/models, columns = competitors. Include known prices or "Not public."

**Positioning matrix** — rows = target segment, vertical, primary message, geographic focus.

**Technology matrix** — rows = tech signals, deployment model, integrations count, security certifications.

---

### Step 9 — Assign Confidence Levels

Annotate every substantive claim with one of:

| Level | Badge | Definition |
|-------|-------|------------|
| High | `[H]` | Direct evidence found on the crawled page — quote or screenshot possible |
| Medium | `[M]` | Multiple indirect signals across pages that point to the same conclusion |
| Low | `[L]` | Single indirect signal or inference with no corroborating evidence |

---

### Step 10 — Cite Sources

Every claim in the report must end with a parenthetical source URL:

```
The company claims SOC 2 Type II compliance. [H] (https://competitor.com/security)
```

---

## Report Structure

Generate the final report in this exact section order:

```markdown
# Competitive Intelligence Report
**Generated:** <ISO 8601 date>
**URLs analysed:** <list>
**URLs skipped:** <list with reason>

---

## Executive Summary
3–5 bullet points with the most critical findings across all competitors.

---

## Access Log
| URL | Status | Reason |
|-----|--------|--------|

---

## Competitor Overview — [Name]
*(Repeat this block for each competitor)*

### Company Profile
### Products & Services
### Pricing Intelligence
### Positioning Analysis
### Technology Indicators
### Marketing & Messaging Analysis

---

## SWOT Analysis

### [Competitor Name]
| | Strengths | Weaknesses |
|-|-----------|------------|
| **Internal** | ... | ... |

| | Opportunities | Threats |
|-|---------------|---------|
| **External** | ... | ... |

*(Repeat for each competitor)*

---

## Competitive Comparison Matrix

### Feature Comparison
### Pricing Comparison
### Positioning Comparison
### Technology Comparison

---

## Strategic Opportunities
Gaps and white space that appear across the competitor landscape.

---

## Risk Assessment
Where competitors appear strongest relative to your position.

---

## Source References
Numbered list of all cited URLs.
```

---

## Saving Artifacts

After the report is complete, save it:

```bash
python scripts/ci_recon.py save_report <slug> < report.md
```

Where `<slug>` is a short identifier for the competitor or analysis run (e.g. `acme-corp-2026-06`). The script prints the saved path.

Artifacts are stored under `~/.hermes/competitive-intelligence/<slug>/<date>/`.

---

## Export Options

After saving the report, export it in one or more formats:

### Interactive HTML Artifact

```bash
python scripts/ci_recon.py export_html <slug>
```

Generates a self-contained `report.html` alongside `report.md` containing:
- Sidebar table of contents with smooth scroll
- Collapsible H2 sections (click heading to toggle)
- Confidence badges (`[H]` / `[M]` / `[L]`) rendered as colour-coded chips
- Sortable, striped comparison tables
- Print stylesheet for clean hard-copy output

The file is fully self-contained (markdown rendered client-side via `marked.js` CDN). Open it in any browser — no server required.

### PDF

```bash
python scripts/ci_recon.py export_pdf <slug>
```

Tries `wkhtmltopdf`, then `pandoc`, then falls back to generating the HTML and printing browser-print instructions. The PDF path (or HTML fallback path) is printed to stdout.

### Google Sheets / CSV

**Step 1 — extract tables as CSV:**

```bash
python scripts/ci_recon.py export_csv <slug>
```

Writes one `.csv` file per markdown table found in the report (e.g. `feature_comparison.csv`, `pricing_comparison.csv`) into the same artifact directory. Prints a list of paths.

**Step 2 — push to Google Sheets (if Windsor MCP is connected):**

Ask the agent to use the Windsor connector:

```
Use the Windsor Google Sheets connector to create a new spreadsheet named
"CI Report — <slug>" and import each CSV file as a separate sheet tab.
```

**Step 2 (alternative) — manual import:**

1. Open Google Sheets → File → Import → Upload.
2. Select each `.csv` file; choose "Insert new sheet(s)".
3. Repeat for each table CSV.

---

## Example Invocation

```
User: Load the competitive-intelligence skill, then analyze these two competitors:
  https://linear.app
  https://height.app
```

The agent will:
1. Validate both URLs (robots.txt + accessibility).
2. Crawl homepage, pricing, features, about, careers, and blog for each.
3. Extract and classify findings.
4. Generate insights, SWOT, and comparison matrix.
5. Produce a structured report cited to source URLs.
6. Save the report to `~/.hermes/competitive-intelligence/`.

---

## Notes

- If a competitor uses a JavaScript-heavy SPA, `web_extract` via Firecrawl handles JS rendering automatically.
- For pricing pages behind a "request a demo" gate, mark pricing as "Not publicly available [H]" and note the gating mechanism.
- Careers pages often reveal technology choices (job descriptions list required skills) and growth signals (volume of open roles). Always include them.
- `web_search(query="site:competitor.com filetype:pdf")` can surface whitepapers and data sheets not linked from the main nav.
