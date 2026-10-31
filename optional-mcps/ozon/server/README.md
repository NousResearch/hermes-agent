# Ozon MCP connector — server

Local MCP server wrapping the Ozon **Seller API** (`https://api-seller.ozon.ru`)
and **Performance API** (`https://api-performance.ozon.ru`) as MCP tools.
Catalog metadata is in `../manifest.yaml`.

Business-logic guidance for which Ozon API/report covers what stays in the
existing `ozon-*` Hermes skills (category `ozon`) — **unchanged by this
connector**. This server is pure transport: auth + HTTP calls.

## Design: one tool per documented endpoint (222 tools)

Matches the industry pattern for large API surfaces instead of a few wide
generic "call" tools — e.g. Cloudflare's official MCP catalog entry ships
~3,300 one-tool-per-OpenAPI-operation tools. Named, individually typed tools
beat a generic `call(endpoint, body)` wrapper because the model's tool-call
validation happens per-function (real parameter names/types in the schema)
instead of on a free-form dict it has to assemble correctly by hand.

Of the 222 endpoints:

- **31 are REAL implementations** with typed, named parameters, built from a
  confirmed request schema: `ozon-seller-skill/references/{products,finance,
  analytics,reviews,sellerrating,sellerinfo,practice}.md` + its proven
  `scripts/*.py`, and `ozon-ads-performance`'s working scripts.
- **191 are explicit `not_implemented` stubs.** Their path/method/
  description are real (from `ozon-api-overview/references/api.md`'s 190
  misc Seller API methods — FBS/FBO postings, returns, carriage, FBP,
  documents, polygon, orders), but no confirmed request body/params schema
  exists yet. Calling one returns `{"status": "not_implemented", ...}`
  **immediately, with no HTTP request sent** — the agent sees the tool
  exists in `tools/list`, learns from the result that it can't be used yet,
  and stops retrying with guessed fields instead of silently failing
  against the real API (or worse, firing a malformed write at a destructive
  endpoint like `carriage/cancel` or `return/giveout/barcode-reset`).

## Upgrading a stub to real

Confirm the exact request schema against Ozon's official docs
(`docs.ozon.ru/api/seller`) or a newly-tested script, then replace that one
function body in `ozon_mcp_server.py` — never guess a field name into an
existing stub. The function name already matches the endpoint
(`method`+`path` → `ozon_<version>_<path_segments>`, with a `_get`/`_put`
suffix only where two methods share one path, e.g.
`ozon_api_client_campaign_id_get` / `_put`).

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
  API keys). No login tool needed — every Seller tool just sends the headers.
- **Performance API**: OAuth2 `client_credentials`, from
  `OZON_PERFORMANCE_CLIENT_ID` / `OZON_PERFORMANCE_CLIENT_SECRET` (Реклама →
  API). Token is fetched and cached in-process (~30 min TTL) automatically
  on first ads-tool call — no separate login tool.

The catalog install flow prompts for all four values and saves them to
`~/.hermes/.env`.

## Write tools

A handful of real (non-stub) tools change live data — `ozon_v3_product_import`,
`ozon_v2_review_change_status`, `ozon_api_client_campaign_id_put` — and are
marked **WRITE** in their docstring. Per the shop's standing rule (see
`ozon-api-overview`): show the current/proposed value and wait for explicit
confirmation before calling one — this connector does not enforce that
itself, it's the calling agent's responsibility. A number of the 191 stubs
are also destructive writes (cancel, delete, create) once implemented — the
same rule will apply to them.
