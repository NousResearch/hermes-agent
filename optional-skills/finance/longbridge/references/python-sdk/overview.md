# Python SDK Overview

**Docs:** https://longbridge.github.io/openapi/python/index.html

## Install

```bash
pip install longbridge
```

> **Note:** The package was previously named `longport`. If upgrading, run `pip uninstall longport` first.

## Import

```python
from longbridge.openapi import (
    Config, OAuthBuilder,
    QuoteContext, AsyncQuoteContext,
    TradeContext, AsyncTradeContext,
    ContentContext, AsyncContentContext,
    HttpClient,
)
```

## Authentication

### OAuth 2.0 (Recommended)

Token cached at `~/.longbridge/openapi/tokens/<client_id>`. Re-runs browser auth only when token is expired.

**Register once:**

```bash
curl -X POST https://openapi.longbridge.com/oauth2/register \
  -H "Content-Type: application/json" \
  -d '{"client_name":"My App","redirect_uris":["http://localhost:60355/callback"],"grant_types":["authorization_code","refresh_token"],"response_types":["code"]}'
# Response: {"client_id": "your-client-id", ...}
```

**Sync:**

```python
from longbridge.openapi import OAuthBuilder, Config

oauth = OAuthBuilder("your-client-id").build(
    lambda url: print(f"Open this URL to authorize: {url}")
)
config = Config.from_oauth(oauth)
```

**Async:**

```python
import asyncio
from longbridge.openapi import OAuthBuilder, Config

async def main():
    oauth = await OAuthBuilder("your-client-id").build_async(
        lambda url: print(f"Open this URL to authorize: {url}")
    )
    config = Config.from_oauth(oauth)

asyncio.run(main())
```

## Config Options

```python
Config.from_oauth(
    oauth,
    http_url=None,           # override LONGBRIDGE_HTTP_URL
    quote_ws_url=None,       # override LONGBRIDGE_QUOTE_WS_URL
    trade_ws_url=None,       # override LONGBRIDGE_TRADE_WS_URL
    language=None,           # Language.ZH_CN / Language.ZH_HK / Language.EN
    enable_overnight=False,  # enable overnight quote
    push_candlestick_mode=PushCandlestickMode.Realtime,
    enable_print_quote_packages=True,
    log_path=None,           # path to log directory
)
```

## Environment Variables

| Variable                           | Description               | Default                                 |
| ---------------------------------- | ------------------------- | --------------------------------------- |
| `LONGBRIDGE_LANGUAGE`              | `zh-CN`, `zh-HK`, `en`    | `en`                                    |
| `LONGBRIDGE_HTTP_URL`              | HTTP endpoint             | `https://openapi.longbridge.com`        |
| `LONGBRIDGE_QUOTE_WS_URL`          | Quote WebSocket           | `wss://openapi-quote.longbridge.com/v2` |
| `LONGBRIDGE_TRADE_WS_URL`          | Trade WebSocket           | `wss://openapi-trade.longbridge.com/v2` |
| `LONGBRIDGE_ENABLE_OVERNIGHT`      | Enable overnight quotes   | `false`                                 |
| `LONGBRIDGE_PUSH_CANDLESTICK_MODE` | `realtime` or `confirmed` | `realtime`                              |
| `LONGBRIDGE_PRINT_QUOTE_PACKAGES`  | Print packages on connect | `true`                                  |
| `LONGBRIDGE_LOG_PATH`              | Log file directory        | (no logs)                               |

**China Mainland:** SDK auto-selects `.cn` endpoints. To force: `LONGBRIDGE_REGION=cn` or `LONGBRIDGE_REGION=hk`.

## HttpClient (Raw HTTP)

For endpoints not wrapped by the typed SDK contexts:

```python
from longbridge.openapi import HttpClient, OAuthBuilder

oauth = OAuthBuilder("your-client-id").build(lambda url: print(url))
http_cli = HttpClient.from_oauth(oauth)

# Sync GET
resp = http_cli.request("get", "/v1/trade/execution/today")
print(resp)  # parsed JSON dict/list

# Sync POST
resp = http_cli.request("post", "/v1/trade/order",
    body={"symbol": "700.HK", "order_type": "LO", "side": "Buy",
          "submitted_quantity": "100", "time_in_force": "Day",
          "submitted_price": "50.00"})

# Async GET
resp = await http_cli.request_async("get", "/v1/trade/execution/today")

# With custom headers
resp = http_cli.request("get", "/v1/some/endpoint",
    headers={"X-Custom-Header": "value"})
```


## Sync vs Async

| Class                 | Nature          | When to use                                |
| --------------------- | --------------- | ------------------------------------------ |
| `QuoteContext`        | Sync            | Scripts, data pipelines, simple tools      |
| `AsyncQuoteContext`   | Async (asyncio) | Concurrent fetches, FastAPI, Jupyter async |
| `TradeContext`        | Sync            | Scripts, command-line tools                |
| `AsyncTradeContext`   | Async (asyncio) | Async servers, concurrent trade ops        |
| `ContentContext`      | Sync            | Fetch news and discussion topics           |
| `AsyncContentContext` | Async (asyncio) | Async news/topics fetching                 |

Both variants share identical method signatures — async versions return awaitables instead of values.

## Error Handling

```python
from longbridge.openapi import OpenApiException

try:
    quotes = ctx.quote(["INVALID.XX"])
except OpenApiException as e:
    print(f"kind={e.kind}, code={e.code}, trace_id={e.trace_id}")
    print(f"message={e.message}")
```

`e.kind`: `ErrorKind.Http`, `ErrorKind.OpenApi`, or `ErrorKind.Other`
