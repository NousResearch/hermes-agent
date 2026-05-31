# Research Substrate v0.7.2 — Endpoint Reference

**Host:** `https://research.alyoechosys.dev` (Cloudflare Tunnel → `localhost:4020`)
**Auth:** Use `X-API-Key: <SUBSTRATE_API_KEY>` for private/rate-limited calls. Live check on May 31, 2026 found `/docs`, `/openapi.json`, `/healthz`, `/search`, `/providers`, and `/v1/models` public.
**Tested:** May 30, 2026
**Datasources:** Kimi Coding Plan (arxiv, scholar, stock_finance_data, yahoo_finance, world_bank, tianyancha)
**Firewall:** Hetzner cloud `hermes-fw-v3` — ports 22, 80, 443, ICMP. Port 4020 is tunnel-only.
**Process:** systemd `research-substrate.service`

## Endpoint Status Summary

| Category | Total | ✅ Working | ⚠️ Empty data | 💥 Schema/timeout |
|----------|-------|-----------|---------------|-------------------|
| Academic (arxiv + scholar) | 5 | 5 | 0 | 0 |
|| Finance (Kimi) | 10 | 7 | 2 | 1 |
|| Yahoo Finance | 11 | 11 | 0 | 0 |
| Macro (World Bank) | 3 | 3 | 0 | 0 |
| Company (天眼查) | 5 | 4 | 1 | 0 |
| Search | 3 | 3 | 0 | 0 |
| Research/Ask | 2 | 2 | 0 | 0 |
|| **Total** | **39** | **34** | **3** | **1** |

## Academic (arxiv + scholar)

Both sources working. Queries hit both automatically regardless of `source` param.

| Endpoint | Test | Result |
|----------|------|--------|
| `/research/academic` | arxiv: "transformer attention" | ✅ 5 arxiv + 2 scholar results |
| `/research/academic` | arxiv: "quantum error correction" | ✅ 5 arxiv results |
| `/research/academic` | arxiv: "CRISPR gene therapy" | ✅ 2 scholar results |
| `/research/academic` | scholar: "LLM reasoning" | ✅ 4 scholar results |
| `/research/academic` | scholar: "climate change economics" | ✅ 5 arxiv results |

**Note:** `source` param doesn't fully filter — both arxiv and scholar are queried. Response parsing has some field misalignment (authors/abstract/title fields get shuffled) but titles are correct.

## Finance (Kimi `stock_finance_data`)

**Ticker format:** Must include exchange suffix — `.O` (NASDAQ), `.N` (NYSE), `.SZ` (Shenzhen), `.SH` (Shanghai).

| Endpoint | Ticker | Result |
|----------|--------|--------|
| `/finance/stock-info` | AAPL.O | ✅ Full info |
| `/finance/stock-info` | 000001.SZ | ✅ Chinese A-share |
| `/finance/stock-forecast` | TSLA.O | ✅ Predicted values |
| `/finance/stock-forecast` | NVDA.O | ✅ Predicted values |
| `/finance/stock-financials` | MSFT.O | ✅ Balance sheet |
| `/finance/stock-holders` | AMZN.O | ✅ Holder info |
| `/finance/stock-segments` | META.O | ✅ Business segments |
| `/finance/stock-announcements` | NVDA.O | ⚠️ Empty data (no announcements in range) |
| `/finance/stock-screener` | keyword "AI" | ⚠️ Empty data |
|| `/finance/stock-index` | SPX | ✅ Auto-rerouted to Yahoo Finance (index) |

**Working tickers:** `AAPL.O`, `TSLA.O`, `NVDA.O`, `MSFT.O`, `AMZN.O`, `META.O`, `GOOGL.O`, `000001.SZ`

## Yahoo Finance

| Endpoint | Ticker | Result |
|----------|--------|--------|
| `/yahoo/stock-info` | BTC-USD | ✅ Crypto |
| `/yahoo/stock-info` | NVDA | ✅ US stock |
| `/yahoo/stock-info` | ^GSPC | ✅ Index |
| `/yahoo/news` | BTC-USD | ✅ |
|| `/yahoo/news` | NVDA | ✅ (was transient timeout) |
| `/yahoo/stock-history` | META | ✅ |
| `/yahoo/stock-history` | ETH-USD | ✅ Crypto |
| `/yahoo/financials` | AAPL | ✅ |
| `/yahoo/holders` | TSLA | ✅ |
|| `/yahoo/recommendations` | NVDA | ✅ (was transient timeout) |
|| `/yahoo/option-chain` | SPY | ✅ Auto-fetches nearest expiration (~60s) |

**Yahoo tickers** don't need exchange suffixes (unlike Kimi finance).

## Macro (World Bank)

| Endpoint | Test | Result |
|----------|------|--------|
| `/macro/world-bank-search` | "GDP per capita" | ✅ Found indicators |
| `/macro/world-bank-search` | "CO2 emissions" | ✅ Found indicators |
| `/macro/world-bank` | indicator=SP.POP.TOTL, country=SGP | ✅ Singapore population data |

## Company (天眼查 Tianyancha)

Chinese company data. Works best with Chinese company names.

| Endpoint | Keyword | Result |
|----------|---------|--------|
| `/company/search` | 华为 (Huawei) | ✅ |
| `/company/search` | 腾讯 (Tencent) | ✅ |
| `/company/search` | 比亚迪 (BYD) | ✅ |
| `/company/search` | Tesla | ✅ |
| `/company/query` | api_name=搜索-搜索, keyword=Apple | ⚠️ Empty data |

## Search (MiniMax)

| Endpoint | Query | Result |
|----------|-------|--------|
| `/search` (GET) | "OpenAI latest 2026" | ✅ 3 results |
| `/search` (GET) | "Singapore news today" | ✅ 3 results |
| `/search` (GET) | "quantum computing breakthrough" | ✅ 3 results |

## Research/Ask (full synthesis)

Search → read → GLM synthesis with citations. ~47s per query.

| Endpoint | Query | Result | Time |
|----------|-------|--------|------|
| `/research/ask` | "quantum computing breakthroughs 2026" | ✅ 5726 chars | 47.5s |
| `/research/ask` | "AI regulation Europe vs US 2026" | ✅ Full report | ~50s |

**Response shape:** `output` (not `answer`), `sources`, `steps`, `conversation_id`, `elapsed_s`

## Auto-routes (v0.6.0)

- **`/yahoo/option-chain`**: `expiration_date` now optional — auto-fetches nearest available date via option-dates API (~60s)
- **`/finance/stock-index`**: Index tickers (SPX, ^GSPC, DJI, ^DJI, VIX, ^VIX, N225, HSI, etc.) auto-reroute to Yahoo `/yahoo/stock-info`

## Key Schemas (POST body params)

```
AcademicSearchRequest:     query*, source, limit
StockPriceRequest:         ticker*
StockInfoRequest:          ticker*
StockForecastRequest:      ticker*
StockFinancialsRequest:    ticker*, statement
StockHolderRequest:        ticker*
StockAnnouncementRequest:  ticker*, start_date*, end_date*
StockScreenerRequest:      keyword*, market
StockFinancialIndexRequest: ticker*, category, period
YahooStockInfoRequest:     ticker*
YahooNewsRequest:          ticker*
YahooHistoryRequest:       ticker*
YahooFinancialsRequest:    ticker*
YahooHolderRequest:        ticker*
YahooRecommendationsRequest: ticker*
YahooOptionDatesRequest:   ticker*
YahooOptionRequest:        ticker*, expiration_date*
WorldBankSearchRequest:    query*, limit
WorldBankRequest:          indicator*, country*
TianyanchaCompanySearchRequest: keyword*
TianyanchaApiCallRequest:  api_name*, keyword*, extra_params
ResearchAskRequest:        input*, model, max_sources, max_chars_per_source
ReadUrlRequest:            url*, max_chars
```
