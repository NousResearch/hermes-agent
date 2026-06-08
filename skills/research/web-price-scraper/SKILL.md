---
name: web-price-scraper
description: "Scrape any URL and extract prices, numbers, and numeric metrics. The user supplies a URL; the agent fetches the page, extracts all price/numeric data, and returns a structured table or JSON report."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Web-Scraping, Price-Tracking, Data-Extraction, E-Commerce, Market-Research, Pricing, Numeric-Data]
    related_skills: [competitive-intelligence, blogwatcher]
---

Extract prices, numbers, and numeric metrics from any webpage. The user provides a URL; the agent fetches it, parses all numeric data, and returns a structured report. Results are optionally saved to disk for tracking over time.

## When to Use

- User says "scrape this URL for prices" or "check prices on [URL]"
- User pastes a product page, pricing page, stats page, or data table URL
- User wants to track how numbers change on a page over time
- User asks "what are the prices on [site]?"

## Quick Reference

| Action | Command |
|---|---|
| Validate URL | `python3 scripts/scrape_helper.py validate <url>` |
| Save results | `echo '<json>' \| python3 scripts/scrape_helper.py save <url>` |
| List saved | `python3 scripts/scrape_helper.py list [domain]` |

## Guardrails

- Always check robots.txt compliance before scraping (step 1 below).
- Do not scrape pages that explicitly forbid automated access in their Terms of Service.
- Limit requests to one page at a time; do not crawl entire sites without explicit user instruction.
- Never store or expose personal data (names, emails, account info) found incidentally on pages.
- If the page requires authentication, inform the user and do not attempt to bypass it.

## Step-by-Step Procedure

### Step 1 — Validate the URL

Run the helper to confirm the URL is reachable and robots.txt allows crawling:

```bash
python3 skills/research/web-price-scraper/scripts/scrape_helper.py validate <URL>
```

Interpret the JSON output:
- `accessible: false` → inform the user the page is unreachable and stop.
- `robots_allows_crawl: false` → inform the user the site disallows automated access and stop (unless the user explicitly accepts this risk and overrides).
- Both true → proceed to step 2.

### Step 2 — Fetch the Page Content

Use `web_extract_tool` to retrieve the page content as markdown:

```python
content = web_extract_tool(["<URL>"], format="markdown")
```

If `web_extract_tool` is unavailable or returns an error, fall back to the browser tool:
```python
browser_result = browser_tool(action="screenshot_and_extract", url="<URL>")
```

### Step 3 — Extract Prices and Numeric Data

Send the fetched content to the LLM with this extraction prompt:

```
You are a data extraction assistant. From the webpage content below, extract ALL prices, 
numeric values, rates, quantities, and statistics. For each item found, return a JSON array 
where each element has these fields:
  - "label": descriptive name or context for the number (e.g. "Monthly plan", "CPU cores", "In stock")
  - "value": the raw numeric value as a number (strip currency symbols, commas, units)
  - "raw": the original string as it appeared on the page (e.g. "$29.99/mo", "1,024 GB")
  - "unit": unit or currency if applicable (e.g. "USD", "GB", "per month", "%")
  - "category": one of: price | quantity | percentage | rating | date | other

Return ONLY valid JSON. If no numeric data is found, return an empty array [].

Webpage content:
<content>
```

Parse the LLM response as JSON. If parsing fails, retry the extraction once with a simpler prompt asking only for a markdown table.

### Step 4 — Present Results

Format the extracted data as a markdown table for the user:

```markdown
## Scraped from: <URL>
**Scraped at:** <timestamp>

| Label | Value | Raw | Unit | Category |
|-------|-------|-----|------|----------|
| ...   | ...   | ... | ...  | ...      |

**Total items found:** N
```

If no numeric data was found, tell the user clearly and suggest trying the browser tool for JavaScript-rendered pages.

### Step 5 — Optionally Save Results

Ask the user: "Would you like me to save these results for future comparison?"

If yes (or if the user said "track" or "monitor" in their original request), pipe the JSON to the save command:

```bash
echo '<json_array>' | python3 skills/research/web-price-scraper/scripts/scrape_helper.py save <URL>
```

The helper prints the saved file path. Report this path to the user.

### Step 6 — Compare with Previous Results (if applicable)

If saved results already exist for this domain, list them:

```bash
python3 skills/research/web-price-scraper/scripts/scrape_helper.py list <domain>
```

Load the most recent previous result and compare values. Report any changes (price increases, decreases, new items, removed items) in a summary section.

## Example Invocations

**User:** "Scrape https://example-store.com/pricing for prices"

Agent flow:
1. Validate URL → accessible, robots allows
2. `web_extract_tool(["https://example-store.com/pricing"], format="markdown")`
3. LLM extraction → JSON array of price items
4. Present markdown table
5. Offer to save

**User:** "What are the prices on https://example.com/products/widget-pro ?"

Agent flow: same as above, no save unless user requests tracking.

**User:** "Check if the price changed on https://example.com/item since last time"

Agent flow:
1. Validate → fetch → extract (as above)
2. `scrape_helper.py list example.com` → load previous result
3. Diff old vs new prices → report changes

## Notes

- JavaScript-heavy pages (SPAs) may return incomplete content via `web_extract_tool`. If prices are missing, use the browser tool with `action="screenshot_and_extract"` or `action="navigate"` to let the JS render.
- Some sites use dynamic pricing (personalized, A/B tested). Results represent the price seen at scrape time.
- Currency symbols are normalized to the `unit` field; always report the `raw` value alongside the parsed number so the user can verify.
- For bulk scraping of many URLs, run this skill once per URL and collate results.
