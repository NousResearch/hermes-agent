# LLM & AI Integration

Longbridge provides several components for integrating with LLMs and AI tools.

## llms.txt

The OpenAPI documentation follows the [LLMs.txt standard](https://llmstxt.org/):

- **URL:** `https://open.longbridge.com/llms.txt`
- **Size:** ~2,100 tokens
- Provides an index of all documentation with links to individual Markdown files

### Add to Cursor

1. Open Cursor → Command Palette (`Cmd+Shift+P`)
2. Search **"Add New Custom Docs"**
3. Enter: `https://open.longbridge.com/llms.txt`
4. In AI chat, use **@Add Context → docs** to include this as context

---

## Markdown API — open.longbridge.com (Developer Docs)

Every documentation page is available in Markdown format. Append `.md` to any doc URL:

```
https://open.longbridge.com/docs/getting-started.md
https://open.longbridge.com/docs/quote/pull/static.md
https://open.longbridge.com/docs/trade/order/submit.md
```

Useful for:
- Feeding specific API docs to an LLM as context
- RAG indexing of the full documentation
- Tool-based content ingestion

---

## Markdown API — longbridge.com (Product Pages)

Pages on `https://longbridge.com` support Markdown retrieval — useful for AI reading live market data, news, and stock profiles.

### How to access

**Method 1 — URL suffix (recommended):**

Add `.md` to any page URL. Locale prefix is optional (`/en/`, `/zh-CN/`, `/zh-HK/`):

```bash
curl https://longbridge.com/en/news.md
curl https://longbridge.com/zh-CN/news.md
curl https://longbridge.com/en/quote/TSLA.US.md
curl https://longbridge.com/en/quote/700.HK.md
```

**Method 2 — Accept header:**

```bash
curl -H "Accept: text/markdown" https://longbridge.com/en/news
curl -H "Accept: text/markdown" https://longbridge.com/en/quote/TSLA.US
```

---

### Response format

Every `.md` response includes a YAML frontmatter block followed by Markdown content:

```yaml
---
title: "Tesla, Inc. (TSLA.US)"
type: "Symbol"           # or "News"
locale: "en"
url: "https://longbridge.com/en/quote/TSLA.US.md"
symbol: "TSLA.US"        # only on Symbol pages
datetime: "2026-03-23T09:07:14.203Z"
locales:
  - [en](https://longbridge.com/en/quote/TSLA.US.md)
  - [zh-CN](https://longbridge.com/zh-CN/quote/TSLA.US.md)
  - [zh-HK](https://longbridge.com/zh-HK/quote/TSLA.US.md)
---
```

---

### Available page types

#### News index

```
https://longbridge.com/en/news.md
```

Returns a chronological list of today's top financial news. Each item includes:
- Headline (links to full article `.md`)
- `datetime` (ISO 8601)
- Summary paragraph
- Related stock symbols as links

Example response excerpt:
```markdown
# Longbridge News

- [Alibaba will release important chip products...](https://longbridge.com/en/news/280093541.md)
  - Datetime: 2026-03-23T03:41:04.000Z
  - Summary: According to media reports, Alibaba's Damo Academy...
```

#### Individual news article

```
https://longbridge.com/en/news/{id}.md
```

Returns full article body in Markdown. Frontmatter includes `title`, `description`, `datetime`, and locale links.

Example:
```
https://longbridge.com/en/news/280093541.md
```

#### Stock quote page

```
https://longbridge.com/en/quote/{SYMBOL}.md
```

Returns a structured stock profile with:
- Company overview and business description
- **Longbridge Financial Score™** (letter grade + multi-factor breakdown)
- Key metrics table: revenue YoY, net profit YoY, market cap, ROE, gross margin
- Valuation analysis: PE, P/B, P/S vs. industry benchmarks
- Analyst consensus: buy/hold/sell counts, price target range
- Peer comparison table

Example response excerpt:
```markdown
# Tesla, Inc. (TSLA.US)

## Company Overview
Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles...

| Item | Detail |
|------|--------|
| Industry | Automobile Manufacturers |
| Exchange | US Market |
| Website | [www.tesla.com](https://www.tesla.com) |

## Longbridge Financial Score™: C
**Overall: C (0.45)**
| Metric | Value |
|--------|-------|
| Industry Ranking | 10 / 53 |
```

---

### Locale support

All pages support 3 locales:

| Locale | URL prefix |
|--------|-----------|
| English | `/en/` |
| Simplified Chinese | `/zh-CN/` |
| Traditional Chinese | `/zh-HK/` |

Each response includes `locales` links in frontmatter pointing to the same page in other languages.

---

### Use cases for AI

| Goal | URL |
|------|-----|
| Today's financial news | `longbridge.com/en/news.md` |
| Full news article text | `longbridge.com/en/news/{id}.md` |
| Stock fundamentals + analyst view | `longbridge.com/en/quote/TSLA.US.md` |
| HK stock profile | `longbridge.com/en/quote/700.HK.md` |
| Chinese-language news | `longbridge.com/zh-CN/news.md` |

**For live price data** (bid/ask, volume, candlesticks), use the MCP tools or SDK — the `.md` pages contain fundamentals and analysis, not real-time quotes.

---

## MCP (Model Context Protocol)

For AI tools that can call functions, use the MCP server instead of text-based docs.
See [mcp.md](./mcp.md) for setup and full tool list.

---

## Summary: Which Integration to Use

| Use Case | Method |
|----------|--------|
| AI coding assistant understanding the API | Add `llms.txt` to Cursor/Claude/etc. |
| Fetch a specific doc page in AI context | Append `.md` to `open.longbridge.com` doc URL |
| Read today's news in AI context | `longbridge.com/en/news.md` |
| Read stock fundamentals + analyst data | `longbridge.com/en/quote/{SYMBOL}.md` |
| AI tool that needs live price/order data | Use MCP server |
| RAG system indexing all API docs | Crawl from `llms.txt` links |
