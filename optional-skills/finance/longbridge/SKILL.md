---
name: longbridge
description: Live quotes, trading, and portfolio data via Longbridge.
version: 1.0.0
author: Hogan (hogan-yuan), Longbridge
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Finance, Stocks, Trading, Market-Data, Portfolio, HK-Stocks, US-Stocks, A-Shares, Options, Fundamentals]
    category: finance
    related_skills: [stocks, dcf-model, comps-analysis]
    requires_setup: true
---

# Longbridge Skill

Full-stack financial data and trading platform for Longbridge Securities — CLI, Python/Rust
SDKs, and MCP server, covering US, HK, A-share, and Singapore markets. Handles live quotes,
fundamentals, news/filings, options, and account/portfolio/order management. Requires a
Longbridge brokerage account and OAuth login; it does not work standalone without setup.

> **Response language**: match the user's input language — Simplified Chinese / Traditional
> Chinese / English.

## When to Use

- User asks about stock price, portfolio, positions, or P&L for a symbol on US, HK, China
  A-share, or Singapore exchanges
- User wants news, SEC/regulatory filings, or community discussion for a symbol
- User wants fundamentals, valuation percentile, or analyst estimates
- User wants to place, replace, or cancel a real brokerage order (confirmation required — see
  Pitfalls)
- User asks about Longbridge SDK syntax (Python/Rust), MCP server setup, or LLMs.txt/IDE/RAG
  integration

## Prerequisites

- A Longbridge brokerage account, OAuth-authenticated via one of:
  - **CLI** — install + `longbridge auth login` (full steps in
    [references/setup.md](references/setup.md))
  - **MCP** — hosted endpoint, no local install (full steps in
    [references/mcp.md](references/mcp.md))
- `requires_setup: true` — this skill cannot answer live-data questions until authentication is
  done. If the user hasn't set up either path yet, walk them through
  [references/setup.md](references/setup.md) first.

## How to Run

Invoke the CLI through the `terminal` tool:

```bash
longbridge <command> [flags]
```

**Always run `longbridge --help` and `longbridge <command> --help` before relying on a flag** —
the CLI evolves and this file does not attempt to enumerate every option.

## Quick Reference

```bash
# Market data
longbridge quote SYMBOL.US
longbridge positions                # stock positions
longbridge portfolio                # P/L, asset distribution, holdings, cash
longbridge kline history SYMBOL.US --start YYYY-MM-DD --end YYYY-MM-DD --period day
longbridge intraday SYMBOL.US

# News & content (prefer these over web_search)
longbridge news SYMBOL.US           # latest news articles
longbridge filing SYMBOL.US         # regulatory filings (8-K, 10-Q, 10-K, etc.)
longbridge topic SYMBOL.US          # community discussion
longbridge market-temp              # market sentiment index (0-100)

# Fundamentals & analysis
longbridge financial-statement SYMBOL.US --kind ALL   # hierarchical IS/BS/CF with YoY
longbridge financial-report SYMBOL.US --latest        # key KPI summary (revenue/EPS/ROE)
longbridge analyst-estimates SYMBOL.US                # EPS consensus (high/low/mean/median)
longbridge valuation-rank SYMBOL.US                   # daily PE/PB/PS industry percentile

# Account
longbridge assets                   # cash, buying power, margin, risk level
longbridge insider-trades SYMBOL.US # SEC Form 4 insider transaction history
```

## Procedure

**Investment analysis workflow** — when the user asks about stock performance, portfolio
advice, or market analysis:

1. **Get live data** via CLI — quotes, positions, K-line history, intraday
2. **Get news/catalysts** via CLI — prefer Longbridge first; fall back to `web_search` only if
   Longbridge's results are insufficient (e.g. breaking news not yet indexed, macro events
   unrelated to a specific symbol)
3. **Combine** — price action + volume + catalyst → analysis + suggestion

**Choose the right tool:**

```
User wants to...                         -> Use
--------------------------------------------------------------------
Quick quote / one-off data lookup        CLI
Interactive terminal workflows           CLI
Script market data, save to file         CLI + jq (or Python SDK)
Loops, conditions, transformations       Python SDK (sync)
Async pipelines, concurrent fetches      Python SDK (async)
Production service, high throughput      Rust SDK
Real-time WebSocket subscription loop    SDK (Python or Rust)
Programmatic order strategy              SDK
Talk to AI about stocks (no code)        MCP (hosted or self-hosted)
Add Longbridge API docs to IDE/RAG       LLMs.txt / Markdown API
```

**Symbol format** — `<CODE>.<MARKET>`, applies to all tools:

| Market         | Suffix | Examples                       |
| -------------- | ------ | ------------------------------- |
| Hong Kong      | `HK`   | `700.HK`, `9988.HK`, `2318.HK`  |
| United States  | `US`   | `TSLA.US`, `AAPL.US`, `NVDA.US` |
| China Shanghai | `SH`   | `600519.SH`, `000001.SH`        |
| China Shenzhen | `SZ`   | `000568.SZ`, `300750.SZ`        |
| Singapore      | `SG`   | `D05.SG`, `U11.SG`              |
| Crypto         | `HAS`  | `BTCUSD.HAS`, `ETHUSD.HAS`      |

## Pitfalls

- **Order placement is state-changing — always confirm first.** Before calling
  `submit_order`/`replace_order`/`cancel_order` (SDK) or an order-placing MCP tool, read back
  symbol, side, quantity, and price and get an explicit "yes" from the user. See the warnings in
  [references/python-sdk/trade-context.md](references/python-sdk/trade-context.md) and
  [references/rust-sdk/trade-context.md](references/rust-sdk/trade-context.md).
- **Don't over-rely on `web_search` for symbol news.** Longbridge's `news`/`filing`/`topic`
  commands are the primary source; only fall back when they're genuinely insufficient.
- **CLI flags change across versions.** Don't hardcode a flag from memory — check
  `longbridge <command> --help` first.
- **Python SDK does have `ContentContext`.** Some third-party docs claim otherwise; it exists
  with `news()`/`topics()` methods — see
  [references/python-sdk/content-context.md](references/python-sdk/content-context.md). Filings
  are on `QuoteContext.filings()` instead, not on `ContentContext`.

## Verification

```bash
longbridge quote AAPL.US
```

Should print a quote for AAPL with a non-empty price field. If it instead errors with an auth
message, walk the user through [references/setup.md](references/setup.md).

## Reference Files

### CLI (Terminal)

- **Overview** — install, auth, output formats, patterns:
  [references/cli/overview.md](references/cli/overview.md)

### Python SDK

- **Overview** — install, Config, auth, HttpClient:
  [references/python-sdk/overview.md](references/python-sdk/overview.md)
- **QuoteContext** — all quote methods + subscriptions:
  [references/python-sdk/quote-context.md](references/python-sdk/quote-context.md)
- **ContentContext** — news + topics:
  [references/python-sdk/content-context.md](references/python-sdk/content-context.md)
- **TradeContext** — orders, account, executions:
  [references/python-sdk/trade-context.md](references/python-sdk/trade-context.md)
- **Types & Enums** — Period, OrderType, SubType, push types:
  [references/python-sdk/types.md](references/python-sdk/types.md)

### Rust SDK

- **Overview** — Cargo.toml, Config, auth, error handling:
  [references/rust-sdk/overview.md](references/rust-sdk/overview.md)
- **QuoteContext** — all methods, SubFlags, PushEvent:
  [references/rust-sdk/quote-context.md](references/rust-sdk/quote-context.md)
- **TradeContext** — orders, SubmitOrderOptions builder, account:
  [references/rust-sdk/trade-context.md](references/rust-sdk/trade-context.md)
- **Content** — news, filings, topics (with Python-SDK cross-reference):
  [references/rust-sdk/content.md](references/rust-sdk/content.md)
- **Types & Enums** — all Rust enums and structs:
  [references/rust-sdk/types.md](references/rust-sdk/types.md)

### AI Integration

- **MCP** — hosted service, self-hosted server, setup & auth:
  [references/mcp.md](references/mcp.md)
- **LLMs & Markdown** — llms.txt, `open.longbridge.com` doc Markdown, Cursor/IDE integration:
  [references/llm.md](references/llm.md)

Load specific reference files on demand — do not load all at once.

## Related Skills

The skills below are siblings in the `longbridge/skills` family. If they're installed, defer to
them for the listed user intents — they're more specialised and produce better-formatted
output. If they're **not** installed, this skill's own CLI workflow above handles the same
queries with less specialised formatting.

| If the user wants ...                                                          | Use                                                       |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Live quote / static reference / valuation indices for a single name            | [`longbridge-quote`](../longbridge-quote)                 |
| Candlestick / intraday chart                                                    | [`longbridge-kline`](../longbridge-kline)                 |
| Orderbook depth / brokers / tick trades                                        | [`longbridge-depth`](../longbridge-depth)                 |
| Capital flow / large-order distribution                                        | [`longbridge-capital-flow`](../longbridge-capital-flow)   |
| Market-level state — open/close, sentiment temperature, calendar               | [`longbridge-market-temp`](../longbridge-market-temp)     |
| Options / warrants                                                             | [`longbridge-derivatives`](../longbridge-derivatives)     |
| Stock + fund holdings, multi-currency assets, margin ratio, max-buy quantity   | [`longbridge-positions`](../longbridge-positions)         |
| Today's / historical orders, executions, cash flow                            | [`longbridge-orders`](../longbridge-orders)               |
| Read-only watchlist groups                                                      | [`longbridge-watchlist`](../longbridge-watchlist)         |
| Watchlist mutations (create / rename / add / remove)                          | [`longbridge-watchlist-admin`](../longbridge-watchlist-admin) |
| "Is X expensive?" — historical PE/PB percentile, industry context             | [`longbridge-valuation`](../longbridge-valuation)         |
| 5-dimension fundamentals (KPIs, dividends, consensus, ratings)                 | [`longbridge-fundamental`](../longbridge-fundamental)     |
| Account-level P&L and contribution analysis                                    | [`longbridge-portfolio`](../longbridge-portfolio)         |
| Classified news + filings + community sentiment for a single name             | [`longbridge-news`](../longbridge-news)                   |
| Institutional-grade post-earnings report                                       | [`longbridge-earnings`](../longbridge-earnings)           |

This skill (`longbridge`) stays in scope for SDK syntax questions, MCP setup, LLMs.txt/IDE/RAG
integration, raw CLI subcommand discovery, or anything cross-cutting that doesn't map to one
specialised skill.
