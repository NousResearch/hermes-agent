# Ozon MCP connector — server

Local MCP server wrapping the Ozon **Seller API** (`https://api-seller.ozon.ru`)
and **Performance API** (`https://api-performance.ozon.ru`) as MCP tools.
Catalog metadata is in `../manifest.yaml`.

Business-logic guidance for which Ozon API/report covers what stays in the
existing `ozon-*` Hermes skills (category `ozon`) — **unchanged by this
connector**. This server is pure transport: auth + whitelisted HTTP calls.

## Design: category tools, not one tool per endpoint

The Seller + Performance APIs span **222 documented endpoints across 18
categories** (products, prices/stocks, analytics, finance, reviews, rating,
seller info, FBS postings, FBO postings, FBP, carriage, returns, orders,
delivery, documents, polygon, digital postings, ads). Hand-writing one
Python function per endpoint would be unmaintainable, so each category gets
one generic tool instead:

- `ozon_list_endpoints(category?)` — discover exact paths + descriptions
  before calling (never guess a path).
- `ozon_seller_call(category, endpoint, body?, params?)` — call any
  Seller API endpoint; `endpoint` is checked against a whitelist built from
  the real documentation, an unknown path is rejected rather than attempted.
- `ozon_performance_call(endpoint, body?, params?)` — same, for the
  Performance (ads) API.

The endpoint registry (paths, methods, descriptions, "write" flags) was
built from:
- `ozon-api-overview/references/api.md` (190 misc Seller API methods:
  FBS/FBO postings, returns, carriage, FBP, documents, polygon, orders)
- `ozon-seller-skill/references/{products,finance,analytics,reviews,
  sellerrating,sellerinfo,practice}.md` (the already-proven, body-shape-
  confirmed subset used by that skill's `scripts/*.py`)
- `ozon-ads-performance/references/` and its working scripts (Performance
  API campaigns/budget/stats)

Every path in the registry is real and documented — nothing invented.
Extend `REGISTRY` in `ozon_mcp_server.py` only against a confirmed doc or
script, never a guess.

## Setup

```bash
python3 -m venv ~/.hermes/mcp-servers/ozon/venv
PYTHONPATH="" ~/.hermes/mcp-servers/ozon/venv/bin/pip install -r requirements.txt
```

(`PYTHONPATH=""` matters on this install — it otherwise points at Hermes's
own venv and breaks a plain `pip install mcp`.)

The catalog manifest (`../manifest.yaml`) points `transport.command` at that
venv's `python3` and `transport.args` at `ozon_mcp_server.py` in this
directory. If you move the venv, update the manifest's `command` path.

## Auth

- **Seller API**: `Client-Id` + `Api-Key` headers, from
  `OZON_API_KEY` / `OZON_CLIENT_ID` (Seller personal account → Settings →
  API keys). No login tool needed — every `ozon_seller_call` just sends the
  headers.
- **Performance API**: OAuth2 `client_credentials`, from
  `OZON_PERFORMANCE_CLIENT_ID` / `OZON_PERFORMANCE_CLIENT_SECRET` (Реклама →
  API). Token is fetched and cached in-process (~30 min TTL) automatically
  on first `ozon_performance_call` — no separate login tool.

The catalog install flow prompts for all four values and saves them to
`~/.hermes/.env`.

## Write endpoints

A handful of endpoints change live data (`/v3/product/import`,
`/v2/review/change-status`, `PUT /api/client/campaign/{id}`) and are marked
`"note": "write"` in the registry. Per the shop's standing rule (see
`ozon-api-overview`): show the current/proposed value and wait for explicit
confirmation before calling one — this connector does not enforce that
itself, it's the calling agent's responsibility.
