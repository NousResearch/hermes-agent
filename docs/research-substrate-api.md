# Research Substrate API

**Base URL:** `https://research.alyoechosys.dev`
**Version:** 0.7.2
**OpenAPI Spec:** `https://research.alyoechosys.dev/openapi.json`
**Interactive Docs:** `https://research.alyoechosys.dev/docs` (no auth needed)

---

## Authentication

Use an API key when calling private or rate-limited endpoints:

```
X-API-Key: <SUBSTRATE_API_KEY>
```

Live check on May 31, 2026: `/docs`, `/openapi.json`, `/healthz`, `/search`, `/providers`, and `/v1/models` returned `200` without a key. Other endpoints may still require `X-API-Key`; keep examples keyed unless you have verified public access.

**Error responses:**
- `401` — missing or invalid key
- `429` — rate limit exceeded

---

## Search

### `GET /search`

Web search with multi-provider fallback (MiniMax → Serper → SearXNG).

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `q` | string | ✅ | — | Search query (min 1 char) |
| `limit` | integer | — | 5 | Max results (1–20) |
| `provider` | string | — | `auto` | Force: `minimax`, `serper`, `searxng` |

```bash
curl -s "https://research.alyoechosys.dev/search?q=AI+regulation+2026&limit=5" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>"
```

**Response:**
```json
{
  "results": [
    {
      "title": "...",
      "url": "...",
      "description": "...",
      "source": "minimax",
      "position": 1
    }
  ],
  "provider": "minimax",
  "elapsed_s": 2.1
}
```

---

## Read URL

### `POST /read-url`

Extract content from any URL. Returns title + text.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | ✅ | — | URL to read |
| `max_chars` | integer | — | 12000 | Max content length |

```bash
curl -s -X POST "https://research.alyoechosys.dev/read-url" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/2401.00001"}'
```

---

## Research

### `POST /research/ask`

Full research synthesis. Searches web, reads top sources, synthesizes with GLM.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | ✅ | — | Research question |
| `model` | string | — | `glm-5.1` | Synthesis model |
| `max_sources` | integer | — | 3 | Number of sources to consult |
| `max_chars_per_source` | integer | — | 6000 | Max chars read per source |

```bash
curl -s -X POST "https://research.alyoechosys.dev/research/ask" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": "Latest quantum computing breakthroughs in 2026?"}'
```

**Response** (~15–60s):
```json
{
  "output": "## ... (markdown with citations)",
  "sources": [
    {"position": 1, "title": "...", "url": "...", "cited": true}
  ],
  "conversation_id": "rs_...",
  "query": "...",
  "elapsed_s": 13.1,
  "steps": { ... }
}
```

### `POST /research/academic`

Academic paper search (arXiv + Google Scholar via Kimi).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Search query |
| `sources` | string[] | — | `["arxiv","scholar"]` | Sources to search |
| `max_results` | integer | — | 5 | Max papers |
| `date_from` | string | — | `2025-01-01` | Filter start date |
| `synthesize` | boolean | — | `false` | Summarize results |

```bash
curl -s -X POST "https://research.alyoechosys.dev/research/academic" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "max_results": 3}'
```

---

## Yahoo Finance

US stocks, crypto, ETFs, indices. No suffix needed.

### `POST /yahoo/stock-info`

| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

### `POST /yahoo/stock-history`

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `period` | string | — | `1mo` |
| `interval` | string | — | `1d` |

### `POST /yahoo/financials`

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `financial_type` | string | — | `income_stmt` |

### `POST /yahoo/holders`

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `holder_type` | string | — | `institutional_holders` |

### `POST /yahoo/news`

| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

### `POST /yahoo/option-chain`

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `expiration_date` | string | — | `""` (auto-fetches nearest) |
| `option_type` | string | — | `calls` |

`expiration_date` is optional — if omitted, auto-fetches nearest expiry (~60s).

### `POST /yahoo/option-dates`

| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

### `POST /yahoo/recommendations`

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `recommendation_type` | string | — | `all` |
| `months_back` | integer | — | `12` |

### `POST /yahoo/stock-actions`

Dividends, splits.

| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

```bash
# Stock info
curl -s -X POST "https://research.alyoechosys.dev/yahoo/stock-info" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Option chain (auto-expiry)
curl -s -X POST "https://research.alyoechosys.dev/yahoo/option-chain" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "SPY"}'
```

---

## Finance (Kimi)

Requires exchange suffix for non-US markets.

**Ticker format:**
- NASDAQ: `.O` → `AAPL.O`, `NVDA.O`
- NYSE: `.N` → `BRK-B.N`
- Shanghai: `.SH` → `600519.SH`
- Shenzhen: `.SZ` → `000001.SZ`
- HK: `.HK` → `0700.HK`

### `POST /finance/stock-info`
| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

### `POST /finance/stock-price`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `type` | string | — | `realtime_price` |
| `time` | string | — | `""` |

### `POST /finance/stock-forecast`
| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

### `POST /finance/stock-financials`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `statement` | string | — | `all` |
| `period` | string | — | `20241231` |

### `POST /finance/stock-history`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `start_date` | string | ✅ | — |
| `end_date` | string | ✅ | — |
| `interval` | string | — | `D` |
| `adjust` | string | — | `forward` |

### `POST /finance/stock-holders`
| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |

### `POST /finance/stock-index`

Auto-reroutes index tickers (`SPX`, `^GSPC`, `DJI`, `NDX`, `RUT`) to Yahoo.

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `category` | string | — | `profitability` |
| `period` | string | — | `20241231` |

### `POST /finance/stock-screener`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `keyword` | string | ✅ | — |
| `market` | string | — | `stock` |

### `POST /finance/stock-segments`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `ticker` | string | ✅ | — |
| `period` | string | — | `20241231` |

### `POST /finance/stock-announcements`
| Field | Type | Required |
|-------|------|----------|
| `ticker` | string | ✅ |
| `start_date` | string | ✅ |
| `end_date` | string | ✅ |

```bash
# Stock index (auto-reroutes to Yahoo for SPX)
curl -s -X POST "https://research.alyoechosys.dev/finance/stock-index" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "SPX"}'
```

---

## Company (天眼查)

Chinese company data via Tianyancha.

### `POST /company/search`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `keyword` | string | ✅ | — |
| `page_size` | integer | — | 20 |
| `page_num` | integer | — | 1 |

### `POST /company/query`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `api_name` | string | ✅ | — |
| `keyword` | string | ✅ | — |
| `extra_params` | object | — | `{}` |

### `POST /company/search-apis`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `query` | string | ✅ | — |
| `limit` | integer | — | 10 |

```bash
curl -s -X POST "https://research.alyoechosys.dev/company/search" \
  -H "X-API-Key: <SUBSTRATE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"keyword": "华为"}'
```

---

## Macro

### `POST /macro/world-bank-search`
| Field | Type | Required |
|-------|------|----------|
| `query` | string | ✅ |

### `POST /macro/world-bank`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `country` | string | ✅ | — |
| `indicator` | string | ✅ | — |
| `date_range` | string | — | `""` |
| `most_recent` | integer | — | 0 |
| `language` | string | — | `en` |

---

## Generate

### `POST /generate/speech`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `text` | string | ✅ | — |
| `voice` | string | — | `English_expressive_narrator` |
| `speed` | number | — | 1.0 |
| `format` | string | — | `mp3` |
| `model` | string | — | `speech-2.8-hd` |

### `POST /generate/image`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `prompt` | string | ✅ | — |
| `aspect_ratio` | string | — | `1:1` |
| `n` | integer | — | 1 |
| `seed` | integer | — | — |
| `prompt_optimizer` | boolean | — | `false` |

### `POST /generate/music`
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `prompt` | string | ✅ | — |
| `lyrics` | string | — | — |
| `lyrics_optimizer` | boolean | — | `false` |
| `instrumental` | boolean | — | `false` |
| `vocals` | string | — | — |
| `genre` | string | — | — |
| `mood` | string | — | — |
| `instruments` | string | — | — |
| `bpm` | integer | — | — |
| `model` | string | — | `music-2.6` |

---

## Health

### `GET /healthz`

No auth required.

---

## Auto-routes (v0.6.0+)

| Endpoint | Auto-behavior |
|----------|---------------|
| `/yahoo/option-chain` | `expiration_date` omitted → auto-fetches nearest from `/yahoo/option-dates` |
| `/finance/stock-index` | Index tickers (`SPX`, `^GSPC`, `DJI`, `NDX`, `RUT`) → auto-rerouted to `/yahoo/stock-info` |

---

## Rate Limits

| Key type | Limit |
|----------|-------|
| Demo (`sub_pub_*`) | 30 requests/hour |
| Production | Unlimited |

---

## Error Codes

| Code | Meaning |
|------|---------|
| `401` | Invalid or missing API key |
| `405` | Wrong HTTP method |
| `422` | Validation error (missing/invalid fields) |
| `429` | Rate limit exceeded |
| `502` | All upstream providers failed (search only) |
