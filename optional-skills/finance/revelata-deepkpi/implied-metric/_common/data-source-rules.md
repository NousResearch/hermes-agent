# Data source rules — deepKPI runtime routing

**deepKPI** is Revelata's data service for US public companies, sourcing data from SEC filings (10-K / 10-Q / 8-K).
The operations available are listed below. Every numeric figure or filing excerpt produced
by skills in this bundle must trace back to a deepKPI call.

deepKPI is reachable two ways depending on the runtime's capabilities. Use the
**MCP tools** when available; fall back to the **REST API** only when the
runtime cannot reach an MCP server.

| Runtime capability | How to call deepKPI |
|---|---|
| **MCP-capable** (Hermes, Claude, ChatGPT, any host with an MCP client) | Use the native MCP tools directly — no API key needed. On Hermes: `hermes mcp install revelata` (or add `https://deepkpi-mcp.revelata.com/mcp` with OAuth) |
| **Non-MCP** (generic shell) | Read `deepkpi-api.md` (sibling file in this folder) and call the REST endpoints using `$DEEPKPI_API_KEY` |

The runtime distinction that matters at call time is **whether MCP is
available**, not which product is hosting the agent. Treat any MCP-capable host
as using native tools for the purposes of these rules.

## Canonical inventory of deepKPI operations

These are the operations exposed both as MCP tools and (for non-MCP runtimes) as
REST endpoints. Names match across both surfaces:

- `query_company_id` — resolve official registrant name → SEC CIK
- `list_kpis` — enumerate every KPI deepKPI has for a company
- `search_kpis` — retrieve specific KPI time series
- `company_summary_search` — thematic / activity-based discovery across summaries
- `get_company_summary` — narrative business description from latest 10-K
- `get_company_segments` — segment / geography structure from latest 10-K
- `list_sec_filing_markdowns` — list available filings for a company
- `get_sec_filing_markdown` — retrieve full filing markdown

## Caveat — env-var fallback

The `DEEPKPI_API_KEY` env var and the `deepkpi-api.md` reference doc are only
needed in non-MCP runtimes (or as an env-var fallback when MCP is unavailable in
an otherwise MCP-capable host).
