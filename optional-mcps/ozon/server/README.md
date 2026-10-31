# Ozon MCP connector — server

Local MCP server wrapping the Ozon **Seller API** (`https://api-seller.ozon.ru`)
and **Performance API** (`https://api-performance.ozon.ru`) as MCP tools.
Catalog metadata is in `../manifest.yaml`.

Business-logic guidance for which Ozon API/report covers what stays in the
existing `ozon-*` Hermes skills (category `ozon`) — **unchanged by this
connector**. This server is pure transport: auth + HTTP calls.

## Design: 10 grouped tools, action parameter selects the endpoint (222 actions total)

Instead of one `@server.tool()` per documented endpoint, endpoints are
grouped by API area into exactly 10 tools: `ozon_fbs`, `ozon_fbo`,
`ozon_fbp`, `ozon_carriage`, `ozon_returns`, `ozon_delivery`,
`ozon_products`, `ozon_finance`, `ozon_reviews`, `ozon_ads`. Each takes
`action: str` (+ optional `body: dict`, `params: dict`) and dispatches to
the matching endpoint internally. This keeps the admin-visible tool count
at 10 (down from 222) while every endpoint stays reachable and
individually documented — each tool's docstring enumerates every `action`
it accepts, one line per action, replacing what a JSON-schema enum
description would otherwise do. No HTTP method/path/business-logic
changed in this regrouping — only how the 222 functions are packaged into
callable tools.

Of the 222 actions:

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
action's handler inside the relevant group tool in `ozon_mcp_server.py` --
never guess a field name into an existing stub. Each group tool has a
`handlers = {...}` dict mapping `action` string -> a lambda/helper function;
find the action by its name (it matches the endpoint's old function name
with the `ozon_<version>_` prefix stripped, e.g. the old
`ozon_v4_posting_fbs_unfulfilled_list` is now action
`'unfulfilled_list_v4'` inside the `ozon_fbs` tool) and replace just that
entry.

## Setup

Venv lives **inside this directory** (`./venv`, gitignored), so the whole
connector — code and venv together — travels with the fork: any
profile/client using this same hermes-agent checkout gets it, instead of
depending on some external directory that only exists on whichever
machine happened to run the install first.

```bash
cd optional-mcps/ozon/server
python3 -m venv venv
PYTHONPATH="" venv/bin/pip install -r requirements.txt
```

(`PYTHONPATH=""` matters on this install — it otherwise points at Hermes's
own venv and breaks a plain `pip install mcp`.)

The catalog manifest (`../manifest.yaml`) points `transport.command` at
`optional-mcps/ozon/server/venv/bin/python3` and `transport.args` at
`ozon_mcp_server.py` in this directory, both as absolute paths under this
checkout's own location. If this fork is ever cloned somewhere other than
`/home/max/.hermes/hermes-agent`, update both paths in the manifest.

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

## Write actions

A handful of real (non-stub) actions change live data — `product_import`
and `product_import_info` lookup in `ozon_products`, `review_change_status`
in `ozon_reviews`, `api_client_campaign_id_put` in `ozon_ads` — and are
marked **[WRITE]** in their tool's docstring action list. Per the shop's
standing rule (see `ozon-api-overview`): show the current/proposed value
and wait for explicit confirmation before calling one — this connector
does not enforce that itself, it's the calling agent's responsibility. A
number of the 191 stub actions are also destructive writes (cancel,
delete, create) once implemented — the same rule will apply to them.
