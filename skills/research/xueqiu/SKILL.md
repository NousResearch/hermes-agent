---
name: xueqiu
description: "Use when the user wants Chinese stock market data — real-time quotes, stock search, hot stock rankings, or trending community posts from Xueqiu (雪球). Requires login cookie (xq_a_token) extracted from Chrome after logging into xueqiu.com."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [xueqiu, snowball, stocks, finance, china, market-data, quotes]
    related_skills: [polymarket]
prerequisites:
  commands: [curl]
---

# Xueqiu (雪球) Skill

Xueqiu (Snowball Finance) is a major Chinese investment community and market
data platform. Its API provides real-time quotes, stock search, hot stock
rankings, and community posts — covering A-shares (沪/深), HK stocks, and US
stocks.

**Login is mandatory for all API endpoints.** The xq_a_token cookie from
logging into xueqiu.com is required.

## When to Use

- User asks for a Chinese stock quote (茅台, 腾讯, AAPL, 00700, etc.)
- User wants to search for Chinese stocks by name or ticker
- User wants hot stock rankings or market sentiment
- User asks about Chinese investment community discussions

Don't use for: US market data (use other financial APIs), or any write
operations (posting, commenting).

## One-Time Setup

All API endpoints require the `xq_a_token` login cookie. Extract it from Chrome:

```bash
pip install browser-cookie3
python3 -c "
import browser_cookie3
for c in browser_cookie3.chrome(domain_name='.xueqiu.com'):
    if c.name == 'xq_a_token':
        print(c.value)
"
```

If browser-cookie3 can't access Chrome (keychain locked, etc.), use
Cookie-Editor (Chrome extension) to manually export the `xq_a_token` value.

Set up the session:

```bash
export XQ_TOKEN="your_token_here"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
curl -s -c /tmp/xq_cookies.txt -o /dev/null -A "$UA" "https://xueqiu.com/"
```

## Quick Reference

All commands need both session cookie and auth token:

```bash
# Stock quote
curl -s -b /tmp/xq_cookies.txt -A "$UA" -e "https://xueqiu.com/" \
  -H "Cookie: xq_a_token=$XQ_TOKEN" \
  "https://stock.xueqiu.com/v5/stock/batch/quote.json?symbol=SH600519"

# Multi-symbol
curl -s -b /tmp/xq_cookies.txt -A "$UA" -e "https://xueqiu.com/" \
  -H "Cookie: xq_a_token=$XQ_TOKEN" \
  "https://stock.xueqiu.com/v5/stock/batch/quote.json?symbol=SH600519,SZ000858,AAPL"

# Stock search (URL-encode Chinese names)
curl -s -b /tmp/xq_cookies.txt -A "$UA" -e "https://xueqiu.com/" \
  -H "Cookie: xq_a_token=$XQ_TOKEN" \
  "https://xueqiu.com/stock/search.json?code=URL_ENCODED_QUERY&size=10"

# Hot stocks (type 10=popular, 12=attention)
curl -s -b /tmp/xq_cookies.txt -A "$UA" -e "https://xueqiu.com/" \
  -H "Cookie: xq_a_token=$XQ_TOKEN" \
  "https://stock.xueqiu.com/v5/stock/hot_stock/list.json?size=10&type=10"

# Hot posts
curl -s -b /tmp/xq_cookies.txt -A "$UA" -e "https://xueqiu.com/" \
  -H "Cookie: xq_a_token=$XQ_TOKEN" \
  "https://xueqiu.com/v4/statuses/public_timeline_by_category.json?since_id=-1&max_id=-1&count=15&category=-1"
```

## Stock Symbol Format

| Market | Prefix | Example | Stock |
|--------|--------|---------|-------|
| Shanghai | SH | SH600519 | 贵州茅台 |
| Shenzhen | SZ | SZ000858 | 五粮液 |
| US stocks | (none) | AAPL | Apple |
| HK stocks | (none) | 00700 | 腾讯 |

## Response Fields

**Stock Quote** (`data.items[].quote`): symbol, name, current, percent, chg,
high, low, open, last_close, volume, amount, market_capital, turnover_rate,
pe_ttm, timestamp.

**Stock Search** (`stocks[]`): code, name, exchange.

**Hot Stocks** (`data.items[]`): code, name, current, percent.

**Hot Posts** (`list[]`): each item's `data` field is a JSON string — parse
twice. Post fields: id, title, text (HTML), user.screen_name, like_count.

## Procedure

1. If XQ_TOKEN not set, guide user through cookie extraction (browser-cookie3
   or Cookie-Editor).
2. Get session cookie: visit xueqiu.com homepage with `-c /tmp/xq_cookies.txt`.
3. For quotes: determine symbol format and call the quote endpoint.
4. For search: URL-encode Chinese queries with `urllib.parse.quote()`.
5. For hot posts: parse the double-JSON (item.data is JSON string within JSON).
6. Strip HTML from post text fields before presenting.

## Common Pitfalls

1. **Login is mandatory.** All API endpoints return error 400016 without
   `xq_a_token`. The homepage session cookie alone is not enough.
2. **Cookie extraction on macOS.** `browser-cookie3` may fail if Chrome's
   cookies are locked by Keychain. Fall back to Cookie-Editor manual export.
3. **URL-encode Chinese.** Search queries with Chinese characters must be
   percent-encoded.
4. **Double JSON in posts.** Hot post responses embed a JSON string in the
   `data` field — parse it with `fromjson` (jq) or `json.loads()`.
5. **U.S./HK stocks.** Use the ticker directly without SH/SZ prefix.

## Verification Checklist

- [ ] `xq_a_token` cookie is extracted and `XQ_TOKEN` is set
- [ ] Quote endpoint returns valid data for a known symbol (e.g. SH600519)
- [ ] Search endpoint returns results for a Chinese name query
