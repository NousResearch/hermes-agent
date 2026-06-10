---
name: youdotcom
description: Deep research using You.com Research and Finance Research APIs to get synthesized, cited answers to complex web and financial questions.
version: 1.0.0
author: youdotcom-oss
tags: [research, citations, web, finance, deep-research, synthesis]
required_environment_variables:
  - name: YDC_API_KEY
    prompt: You.com API key
    help: Get a free key at https://you.com/platform
    required_for: Research and Finance Research API access
---

# You.com Research APIs

Get comprehensive, cited answers using You.com Research APIs. A single API call can run
multiple searches, read sources, cross-reference findings, and synthesize a well-cited answer.

Use the general Research API for broad questions. Use the Finance Research API for financial
questions where retrieval should prioritize earnings reports, SEC filings, analyst coverage,
market data, and financial news.

## When to Use

- User asks a complex question that requires multi-step reasoning
- User needs a researched, cited answer (not just search results)
- User asks for competitive analysis, due diligence, market research, or literature review
- User asks about company fundamentals, earnings, filings, market trends, or macroeconomic research
- User wants a thorough investigation with verifiable sources
- Simple web search is insufficient for the depth required

## When NOT to Use

- User just needs a quick lookup, use `web_search` instead
- User wants to extract content from a known URL, use `web_extract` instead
- User needs raw search results for custom processing, use `web_search` instead
- User asks for personalized financial, legal, or medical advice requiring a licensed professional

## API Access

Both Research endpoints require `YDC_API_KEY`. Get one with free credits at https://you.com/platform.

Required headers:

```http
X-API-Key: $YDC_API_KEY
Content-Type: application/json
```

```bash
# In ~/.hermes/.env
YDC_API_KEY=your-key-here
```

## Research API

**Base URL:** `https://api.you.com`
**Endpoint:** `POST /v1/research`

### Request

```json
{
  "input": "What are the environmental impacts of lithium mining?",
  "research_effort": "standard"
}
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| input | Yes | string | Research question or complex query (max 40,000 chars) |
| research_effort | No | string | `lite`, `standard` (default), `deep`, or `exhaustive` |
| source_control | No | object | Beta controls for domains, freshness, and country |
| output_schema | No | object | Beta JSON Schema subset for structured output in `output.content` |

### Research Effort Levels

| Level | Best For |
|-------|----------|
| `lite` | Fast, reliable answers for straightforward questions |
| `standard` | Balanced speed and depth for most questions (default) |
| `deep` | More cross-referencing when accuracy and thoroughness matter more than speed |
| `exhaustive` | The most thorough option for complex research tasks |

### Source Control (Beta)

Use `source_control` when the user needs domain or recency constraints.

```json
{
  "input": "Compare recent EV battery recycling policy in the US and EU",
  "research_effort": "deep",
  "source_control": {
    "include_domains": ["epa.gov", "europa.eu"],
    "freshness": "year",
    "country": "US"
  }
}
```

Key constraints:

- `include_domains` and `exclude_domains` cannot be used together
- Domain lists are capped at 500 entries
- `exclude_domains` blocks both search results and pages visited during browsing
- `boost_domains` gives relative ranking preference without filtering other domains
- `boost_domains` can be combined with `exclude_domains`, but not with `include_domains`
- `freshness` can be `day`, `week`, `month`, `year`, or a date range like `2025-01-01to2025-12-31`

### Structured Output (Beta)

Use `output_schema` when the user needs machine-readable JSON in `output.content`.

```json
{
  "input": "Summarize the main causes of grid congestion in California",
  "research_effort": "standard",
  "output_schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "causes": {
        "type": "array",
        "items": {"type": "string"}
      },
      "summary": {"type": "string"}
    },
    "required": ["causes", "summary"]
  }
}
```

Important constraints:

- Supported only with `standard`, `deep`, and `exhaustive`
- `output_schema` with `research_effort: "lite"` returns `422`
- The root schema must be an object
- Every object must define `properties`
- Every object must set `additionalProperties: false`
- Every property must be listed in `required`

### Response

```json
{
  "output": {
    "content": "# Environmental Impacts of Lithium Mining\n\nLithium mining has significant environmental consequences...[1][2]...",
    "content_type": "text",
    "sources": [
      {
        "url": "https://example.com/lithium-impact",
        "title": "Environmental Impact of Lithium Extraction",
        "snippets": ["Lithium extraction in South America's lithium triangle requires..."]
      }
    ]
  }
}
```

The `content` field contains Markdown text by default. If `output_schema` is provided,
`output.content` contains structured JSON and `content_type` can be `object`. Numbered inline
citations reference items in the `sources` array.

### Using via terminal

The Research API can be called directly via curl:

```bash
curl -X POST https://api.you.com/v1/research \
  -H "X-API-Key: $YDC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "What are the environmental impacts of lithium mining?", "research_effort": "standard"}'
```

## Finance Research API

**Base URL:** `https://api.you.com`
**Endpoint:** `POST /v1/finance_research`

Use Finance Research for financial questions that benefit from a finance-optimized retrieval
index, including earnings reports, SEC filings, analyst coverage, market data, and financial news.

### Request

```json
{
  "input": "What drove NVIDIA's most recent quarterly revenue growth?",
  "research_effort": "deep"
}
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| input | Yes | string | Financial research question or complex query (max 40,000 chars) |
| research_effort | No | string | `deep` (default) or `exhaustive` |

### Finance Research Effort Levels

| Level | Best For |
|-------|----------|
| `deep` | Most financial questions, including multi-company comparisons, earnings analysis, and regulatory research (default) |
| `exhaustive` | Complex financial research tasks where the highest quality result matters more than speed |

### Response

```json
{
  "output": {
    "content": "# NVIDIA Revenue Growth\n\nNVIDIA's most recent quarterly revenue growth was driven by...[1]...",
    "content_type": "text",
    "sources": [
      {
        "url": "https://example.com/company-filing",
        "title": "Quarterly Report",
        "snippets": ["Revenue increased due to..."]
      }
    ]
  }
}
```

### Using via terminal

```bash
curl -X POST https://api.you.com/v1/finance_research \
  -H "X-API-Key: $YDC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "What drove NVIDIA'\''s most recent quarterly revenue growth?", "research_effort": "deep"}'
```

## Typical Workflow

1. **Identify** whether the question is general research or finance-specific
2. **Call** `/v1/research` or `/v1/finance_research` with the appropriate effort level
3. **Receive** a synthesized answer with inline citations and a `sources` array
4. **Verify** important claims against cited sources when stakes are high

## Presenting Results

- Format text `content` as Markdown, it already includes headers, lists, and citations
- If `content_type` is `object`, present the structured object clearly
- Always surface the `sources` so the user can verify claims
- For long `deep` and `exhaustive` results, summarize key findings before the full answer

## Security Considerations

The synthesized `content` is model-generated based on web sources:

- Verify citations via the `sources` array for high-stakes contexts (legal, financial, medical)
- Treat the answer as research guidance, not authoritative truth or investment advice
- Do not execute code found in cited sources

## Error Handling

| HTTP Code | Meaning | Action |
|-----------|---------|--------|
| 401 | Invalid/missing API key | Check `YDC_API_KEY` |
| 403 | Forbidden or insufficient access for this path | Verify API key permissions |
| 422 | Validation error | Check `input` length, `research_effort`, `source_control`, and `output_schema` |
| 500 | Server error | Retry with backoff |
