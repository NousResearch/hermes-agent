# deepkpi-api — REST access to deepKPI

REST API access to Revelata's deepKPI endpoints. Provides company lookup, KPI
discovery, KPI search, similarity search over company summaries, company
narrative summary, and segment breakdown for US public companies via HTTP +
API key. Required for OpenClaw; usable as env-var fallback for Claude when
MCP is unavailable. Other deepKPI skills (kpi, filing, company-summary,
implied-metric, etc.) call through this skill when MCP tools are not present.

Powered by [Revelata](https://www.revelata.com).

## API

Base URL: `https://deepkpi-api.revelata.com`

All requests use POST with JSON body and these headers:
```
Content-Type: application/json
X-API-Key: $DEEPKPI_API_KEY
```

Endpoints:

### POST /v1.0/query_company_id
Look up a company's identifier by name. For US companies, this returns the SEC
CIK value as `company_id`.
```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/query_company_id" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"company_name": "Darden Restaurants", "num_of_res": 3}'
```

### POST /v1.0/list_kpis
List all available KPIs for a company, organized by category. Returns
categories: Company KPIs, Segment Financials, Consolidated Income Statement,
Consolidated Balance Sheet, Consolidated Cash Flow, Capital Structure, Taxes,
Miscellaneous, Off-Balance Sheet Items.
```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/list_kpis" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"company_id": "940944"}'
```

### POST /v1.0/search_kpis
Semantic search for specific KPIs. Each result costs 1 credit (max 15). Default
to 3 results — only increase if you need broader coverage.
```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/search_kpis" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"company_id": "940944", "query": "Olive Garden same-restaurant sales", "num_of_res": 3}'
```

### POST /v1.0/company_summary_search

Find companies whose **10-K-derived narrative summary** best matches a natural-language `query`.
**1 credit per company** returned (via `top_k_companies`, max **15**). Empty results cost nothing.

```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/company_summary_search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"query": "companies that make AIs", "top_k_companies": 10}'
```

### POST /v1.0/get_company_summary

Narrative summary for one company (from latest 10-K). **3 credits** on success.
Same `company_id` as other endpoints (SEC CIK for US companies).

```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/get_company_summary" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"company_id": "940944"}'
```

### POST /v1.0/get_company_segments

Structured segment breakdown for one company (from latest 10-K). **3 credits** on success.

```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/get_company_segments" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"company_id": "940944"}'
```

### POST /v1.0/list_sec_filing_markdowns

List filings available as markdown for a given CIK. **Free**.

```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/list_sec_filing_markdowns" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"cik": 1326801, "form_type": "10-K", "start_date": "2025-01-01", "end_date": "2026-12-31"}'
```

`form_type` is optional. If omitted, all form types under the CIK are included.

### POST /v1.0/get_sec_filing_markdown

Fetch the markdown content for a specific filing. **10 credits** per successful call.

```bash
curl -s -X POST "https://deepkpi-api.revelata.com/v1.0/get_sec_filing_markdown" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEEPKPI_API_KEY" \
  -d '{"cik": 1326801, "acc_no": "0001628280-26-003942", "seq_no": 1}'
```

`seq_no` is optional and defaults to `1`.

**Filing text policy:** For US SEC primary text (quotes, MD&A, “what they said”), call **`list_sec_filing_markdowns` → `get_sec_filing_markdown` first**. Listing is free; markdown content is **10 credits per filing** — use it anyway; do not substitute web/SEC.gov browsing to avoid credits while deepKPI access works. Full workflow and quote rules: see the `filing` skill (if deepKPI is unavailable, follow `connection-failure.md` in this folder before using other sources).

## Workflow

**Always follow this sequence. Do not skip steps.**

### Step 1 — Resolve the company to a company_id

Call `/v1.0/query_company_id` with the company name. This returns matching
companies with their `company_id` values (for US companies, this is the SEC CIK).

- Use the official registrant name (e.g. "Darden Restaurants" not "Olive Garden").
- If the user names a brand or subsidiary, map it to the parent company first.
- The `company_id` from the response is required for all subsequent calls.

### Step 2 — Discover available KPIs (recommended)

Call `/v1.0/list_kpis` with the `company_id` from Step 1. **This call is free**
and is the recommended way to discover what KPIs exist before spending credits on
`search_kpis`. Use the returned KPI names to craft precise, low-cost searches in
Step 3 — this is cheaper and more accurate than requesting many results from
`search_kpis`.

**Use this when:**
- The user asks a broad question ("tell me about their financials")
- You need to disambiguate between similar metrics
- The user asks about segments or brands and you need to discover exact KPI names
- You want to minimize credit usage by identifying the right KPI names first

The `source` filter is optional. Omit it to list KPIs across all filing types.
Set to `10-K` for annual, `10-Q` for quarterly.

The response is a JSON object where keys are categories (e.g. "Company KPIs",
"Segment Financials") and values are arrays of KPI name strings. Use these names
to craft precise `search_kpis` queries.

### Step 3 — Search for specific KPIs

Call `/v1.0/search_kpis` with:
- `company_id` — from Step 1
- `query` — natural language description of the metric
- `num_of_res` — number of results (default 3, max 15). Each result costs 1
  credit — start small and increase only if needed.
- `source` — (optional) `10-K`, `10-Q`, or `8-K`. Omit to search across all
  filing types.

**Query tips:**
- Be specific: "Olive Garden restaurant-level margins" not "margins"
- Include the segment or brand name for business unit questions
- For cost structure: name the cost category ("food and beverage costs as
  percentage of revenue")
- For debt: specify the instrument ("senior unsecured notes interest rate and
  maturity")

**Multiple searches are normal.** A single user question often needs 2–4
searches with different query strings. For example, "How is Darden doing?" needs
separate searches for revenue, same-restaurant sales, margins, and unit growth.

### Optional — company narrative, segments, thematic discovery

For **`get_company_summary`** (what a company does), **`get_company_segments`**
(segment / geo structure), and **`company_summary_search`** (thematic “who does X?”), see the `company-summary` skill.
When the user wants **metrics**, continue with **`list_kpis`** / **`search_kpis`** as
in the steps above.

## Response guidelines

- Always present KPI values with the provenance URL provided. Copy the
  provenance URL exactly.
- Organize segment comparisons in a table.
- If results are weak, rephrase the query or use `/v1.0/list_kpis` to find the
  correct KPI name.
