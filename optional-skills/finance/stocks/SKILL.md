---
name: stocks
description: Real-time stock quotes, price history, company search, multi-stock compare, and crypto prices via Yahoo Finance. No API key required.
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [stocks, finance, market, trading, crypto, yahoo-finance, investing]
    category: finance
    requires_toolsets: [terminal]
---

# Stocks & Finance Skill

Real-time stock market data via Yahoo Finance.
5 commands: quote, search, history, compare, crypto.

No API key needed. Python stdlib only.

---

## When to Use
- User asks for a stock price (AAPL, TSLA, MSFT...)
- User wants to look up a company by name
- User wants price history or performance over time
- User wants to compare multiple stocks side by side
- User asks for a crypto price (BTC, ETH, SOL...)

---

## Prerequisites
Python 3.8+ stdlib only. No pip installs.
Script path: `~/.hermes/skills/finance/scripts/stocks_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/finance/scripts/stocks_client.py
python3 $SCRIPT quote AAPL
python3 $SCRIPT quote AAPL MSFT GOOGL TSLA
python3 $SCRIPT search "Tesla"
python3 $SCRIPT history NVDA --range 6mo
python3 $SCRIPT compare AAPL MSFT GOOGL
python3 $SCRIPT crypto BTC ETH SOL
```

---

## Commands

### quote SYMBOL [SYMBOL2...]
Current price, change, change%, volume, 52-week high/low.

### search QUERY
Find stocks by company name. Returns top 5: symbol, name, exchange, type.

### history SYMBOL [--range RANGE]
Price history. Ranges: 1mo, 3mo, 6mo, 1y, 5y (default: 1mo).
Returns OHLCV per day + stats (min, max, avg, total_return_pct).

### compare SYMBOL1 SYMBOL2 [...]
Side-by-side: price, change%, 52w performance.

### crypto SYMBOL [SYMBOL2...]
Crypto prices. Pass BTC not BTC-USD (appended automatically).

---

## Pitfalls
- Yahoo Finance API is unofficial and may change without notice.
- market_cap and pe_ratio may return null (require session crumb).
- Rate limits: add delays between bulk requests.

---

## Verification
```bash
python3 ~/.hermes/skills/finance/scripts/stocks_client.py quote AAPL
```
