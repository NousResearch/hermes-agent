---
name: ai-company-research
description: >
  Browse artificialintelligencecompanies.com to find AI vendors and startups
  serving a given niche or problem. Returns company names, URLs, and descriptions
  via the site's public JSON API. No API key required. Agent-friendly site.
category: browser
version: 1.0.0
---

# AI Company Research

## Target
- URL: https://artificialintelligencecompanies.com
- Method: API (web_extract / curl to JSON endpoint)
- Bot detection: None (agent-friendly site, robots.txt allows GPTBot/ClaudeBot)

## Capabilities
- Keyword search across 12 AI vertical categories
- Full category roster via category pages
- Company names, URLs, truncated descriptions
- No side effects: read-only

## Anti-Detection
- Bot detection level: None
- CAPTCHA risk: None
- Rate limit: ~60 requests/hour (generous)

## API Details

### Keyword Search
```
GET https://artificialintelligencecompanies.com/api/search/?q=<query>
```
- `q` required, URL-encode multi-word queries
- Returns: `{"companies": [{name, url, description}], "categories": [{name, url, description}]}`
- Hard-capped at 5 companies + categories per query
- Descriptions truncated to ~100 chars
- Empty arrays on no-match

### Category Roster (Full Vertical Coverage)
```
GET https://artificialintelligencecompanies.com/cat/<slug>/
```
- Slug examples: `healthcare-ai`, `customer-service`, `legal-ai`, `computer-vision`
- Returns server-rendered HTML with JSON-LD embedded
- Extract via web_extract and parse JSON-LD
- Full descriptions, not truncated

## Workflow

### Step 1: Keyword Search
```bash
curl -s "https://artificialintelligencecompanies.com/api/search/?q=customer%20service%20ai"
```

### Step 2: Parse Results
```json
{
  "companies": [
    {
      "name": "Company Name",
      "url": "https://artificialintelligencecompanies.com/company/slug/",
      "description": "AI solution for customer service automation..."
    }
  ],
  "categories": [
    {
      "name": "Customer Service AI",
      "url": "https://artificialintelligencecompanies.com/cat/customer-service/",
      "description": "Companies providing AI customer service solutions"
    }
  ]
}
```

### Step 3 (Optional): Full Category Roster
```bash
curl -s "https://artificialintelligencecompanies.com/cat/<slug>/"
```
Parse JSON-LD embedded in HTML for full descriptions.

### Step 4: Return
```json
{
  "success": true,
  "query": "customer service ai",
  "companies": [
    {
      "name": "Company Name",
      "url": "https://...",
      "description": "AI solution for customer service automation...",
      "category": "Customer Service AI"
    }
  ],
  "categories": [
    {
      "name": "Customer Service AI",
      "url": "https://...",
      "company_count": 15
    }
  ]
}
```

## Use Cases

- **Job hunt (P1)**: Find AI/PropTech/SaaS companies in target niche → research → apply
- **Market research**: Map competitive landscape across AI vertical
- **Content**: Identify trending AI vendors for Twitter/LinkedIn content
- **CoachOS**: Research AI coaching platforms for competitive insight

## Integration Points

- **Job hunt pipeline**: Company data → Glassdoor research → job-prep CV tailoring
- **Content engine**: Identified companies → drafted social posts (CeeCee)
- **Landscape monitoring**: Track new AI companies in relevant verticals

## Pitfalls

- API hard-caps at 5 results per query (use category page for full roster)
- Descriptions truncated (~100 chars) — use category page for full text
- Empty arrays on no-match (HTTP 200, not 404)
- No pagination on search endpoint
- Categories may overlap (same company in multiple categories)
- Site updated regularly — descriptions may change

## Verification

1. Search for a known AI vertical (e.g., "healthcare ai")
2. Verify at least one company returned
3. Test with no-match query (ensure empty arrays handled)
4. Test category page extraction
5. Cross-reference a company URL to confirm it loads
