---
name: tradingview-mcp
description: AI trading intelligence via the tradingview-mcp-server MCP server — live crypto/stock screening, 30+ technical indicators (RSI, MACD, Bollinger Bands), candlestick patterns, Yahoo Finance data, and 6-strategy backtesting. Multi-exchange (Binance, KuCoin, Bybit). No API keys required. Wired through Hermes' native MCP client.
version: 1.0.0
author: Hermes Agent
license: MIT
tags: [trading, finance, crypto, stocks, technical-analysis, backtesting, mcp]
metadata:
  hermes:
    tags: [MCP, Trading, Market Data]
    related_skills: [native-mcp, polymarket]
    upstream: https://github.com/atilaahmettaner/tradingview-mcp
---

# TradingView MCP — AI Trading Intelligence

Wraps [atilaahmettaner/tradingview-mcp](https://github.com/atilaahmettaner/tradingview-mcp)
(`tradingview-mcp-server` on PyPI) as a Hermes skill. The server is run as an
MCP stdio subprocess through Hermes' [native MCP client](../../mcp/native-mcp/SKILL.md) —
no bridge code in Hermes, just config.

## When to Use

- User asks about stock or crypto prices, screens, or technical indicators
- User wants to backtest trading strategies (RSI, MACD, mean-reversion, etc.)
- User asks about candlestick patterns, Bollinger Bands, Sharpe / Calmar / drawdown metrics
- User wants Yahoo Finance data without setting up an API key
- User asks about multi-exchange crypto data (Binance, KuCoin, Bybit)

For prediction markets (Polymarket) use the `polymarket` skill instead. For generic
MCP plumbing (how Hermes discovers/injects MCP tools), see the `native-mcp` skill.

## Prerequisites

- Python 3.10+
- `uv` / `uvx` installed — recommended runner for the server
- Hermes `mcp` Python extra installed (`pip install mcp`) so the native MCP
  client is enabled

No API keys, no exchange accounts, no OpenAI key — the upstream server talks
directly to public endpoints (Yahoo Finance, Binance, Reddit RSS, etc.).

## Install

Install the server once so `uvx` can launch it quickly:

```bash
uv tool install tradingview-mcp-server
# or, ad hoc, no install step:
#   uvx --from tradingview-mcp-server tradingview-mcp
```

## Configure in Hermes

Add the server to `~/.hermes/config.yaml` under `mcp_servers` so Hermes boots
it as a stdio MCP subprocess and auto-discovers its tools:

```yaml
mcp_servers:
  tradingview:
    command: "uvx"
    args: ["--from", "tradingview-mcp-server", "tradingview-mcp"]
    timeout: 180          # some backtests / screens take a while
    connect_timeout: 60
```

macOS note: GUI-launched Hermes processes may not see `~/.local/bin` on PATH.
If `uvx` can't be found, use the absolute path:

```yaml
mcp_servers:
  tradingview:
    command: "/Users/YOUR_USERNAME/.local/bin/uvx"
    args: ["--from", "tradingview-mcp-server", "tradingview-mcp"]
```

Restart Hermes. On startup the native MCP client will connect, list the
server's tools, and register them under the `mcp_tradingview_*` prefix
(hyphens/dots in names become underscores — see `native-mcp` for the naming
rules).

## What You Get

After discovery, the agent will see tools roughly corresponding to the
upstream server's categories (names are `mcp_tradingview_<tool>`):

- **Screening** — live crypto and equity screens across exchanges
- **Technical analysis** — RSI, MACD, Bollinger Bands (with the project's
  proprietary ±3 rating), ATR, moving averages, candlestick patterns
- **Backtesting** — `backtest_strategy` (6 built-in strategies with Sharpe,
  Calmar, expectancy, max drawdown) and `compare_strategies` to rank all 6
  on a symbol
- **Sentiment / news** — Reddit community sentiment and RSS news feeds
- **Market data** — Yahoo Finance historicals plus Binance / KuCoin / Bybit
  spot data

The upstream README is the source of truth for the exact tool list; it
changes between versions, and Hermes surfaces whatever the server advertises
via `list_tools()` at startup.

## Using the Tools

Once the server is wired up, just ask the agent normally — the native MCP
client injects the tools into every conversation:

- "Run a market snapshot for AAPL, MSFT, and NVDA"
- "Backtest an RSI strategy on BTC-USD over the last year"
- "Compare all 6 strategies on ETH-USD and rank by Sharpe"
- "What do the Bollinger Bands say about TSLA right now?"
- "Give me the Reddit sentiment on $SOL today"

The agent picks the underlying `mcp_tradingview_*` tool automatically. You
don't need to name it.

## Troubleshooting

- **Tools don't appear** — check that `pip install mcp` succeeded and that
  Hermes logs `connected MCP server 'tradingview'` at startup. See the
  `native-mcp` skill's troubleshooting section for the generic checklist.
- **`uvx: command not found`** — use the absolute path in `command:` (see
  macOS note above), or `which uvx` and paste that in.
- **Slow first call** — `uvx --from` may be cold-installing the package on
  first use; `uv tool install tradingview-mcp-server` once up front to warm
  the cache.
- **Rate limits from Yahoo/Binance** — upstream issue; back off and retry.
  The upstream repo tracks these.

## Notes

- This skill is a thin integration over the upstream MCP server — all
  analytics logic lives in `tradingview-mcp-server`. Bugs in indicator math,
  backtest results, or data sourcing should go to
  https://github.com/atilaahmettaner/tradingview-mcp.
- License: upstream is MIT. Nothing in this skill ships upstream code.
- Not investment advice. The backtests and screens are informational.
