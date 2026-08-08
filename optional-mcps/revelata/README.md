# Revelata deepKPI MCP integration

[Revelata](https://www.revelata.com) provides SEC-sourced financial data for US public companies through its **deepKPI** MCP server. Hermes connects to deepKPI as a remote HTTP MCP server with native OAuth 2.1 + Dynamic Client Registration (DCR) — the same pattern used by the Linear and Comfy-Cloud catalog entries. No local installation is required.

## Prerequisites

- Hermes Agent installed with MCP support (included in the standard install)
- A Revelata account (free tier is fine — 100 credits/month)
- A browser for the one-time OAuth login (or SSH tunnel if running headless)

## Quick start

```bash
hermes mcp install revelata
hermes mcp login revelata
```

The `install` command writes the `mcp_servers.revelata` block to `~/.hermes/config.yaml`. The `login` command opens your browser for the OAuth flow and caches the token at `~/.hermes/mcp-tokens/revelata.json`.

After login, restart your Hermes session so the deepKPI tools are loaded:

```bash
hermes chat
```

Verify the tools are registered by asking Hermes:

```text
Which Revelata deepKPI tools are available right now?
```

You should see 8 tools prefixed with `mcp_revelata_`:

```
mcp_revelata_query_company_id
mcp_revelata_list_kpis
mcp_revelata_search_kpis
mcp_revelata_company_summary_search
mcp_revelata_get_company_summary
mcp_revelata_get_company_segments
mcp_revelata_list_sec_filing_markdowns
mcp_revelata_get_sec_filing_markdown
```

### Try it

```text
Find Apple's CIK using Revelata, then pull its latest 10-K company summary.
```

Hermes will call `mcp_revelata_query_company_id` to resolve "Apple" to an SEC CIK, then call `mcp_revelata_get_company_summary` to retrieve the narrative summary from the latest 10-K.

## Configuration

The `hermes mcp install revelata` command writes the following block to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  revelata:
    url: https://deepkpi-mcp.revelata.com/mcp
    auth: oauth
    enabled: true
```

You can also add it manually:

```yaml
mcp_servers:
  revelata:
    url: https://deepkpi-mcp.revelata.com/mcp
    auth: oauth
```

No API keys or environment variables are needed — the OAuth 2.1 + DCR flow handles authentication entirely. The MCP client discovers the authorization server, registers a client dynamically (RFC 7591), runs PKCE, and caches the resulting token.

## Token management

Tokens are cached at `~/.hermes/mcp-tokens/revelata.json` with `0600` permissions. Subsequent sessions reuse the cached token silently until refresh fails.

To re-authenticate (expired token, changed account, etc.):

```bash
hermes mcp login revelata
```

## Headless / remote hosts

When Hermes runs on a server without a browser, the OAuth loopback callback can't reach your laptop. Two options:

**Paste-back (no setup):** On an interactive terminal, Hermes prints an authorize URL and a paste prompt. Open the URL in your browser, approve, copy the full URL the browser ends up on (the redirect will show a connection error — that's expected), and paste it at the Hermes prompt. Bare `?code=...&state=...` query strings also work.

**SSH port forward:** From a separate terminal:

```bash
ssh -N -L <port>:127.0.0.1:<port> user@host
```

Then let the redirect flow normally.

:::tip
Use `hermes mcp login revelata` from a fresh terminal — it waits up to 5 minutes for you to complete the OAuth flow. Editing `~/.hermes/config.yaml` from inside a running Hermes session triggers auto-reload with a 30-second timeout, which is too short for interactive OAuth.
:::

## Tools and credit costs

deepKPI exposes 8 read-only tools. Credit costs vary per call:

| Tool | Registered as | Cost | Description |
|------|---------------|------|-------------|
| `query_company_id` | `mcp_revelata_query_company_id` | Free | Free-text company name to SEC CIK lookup |
| `list_kpis` | `mcp_revelata_list_kpis` | Free | KPI catalog for a company |
| `search_kpis` | `mcp_revelata_search_kpis` | 1 credit/result | Semantic KPI search across companies |
| `company_summary_search` | `mcp_revelata_company_summary_search` | 1 credit/company | Thematic discovery across companies |
| `get_company_summary` | `mcp_revelata_get_company_summary` | 3 credits | Narrative summary from latest 10-K |
| `get_company_segments` | `mcp_revelata_get_company_segments` | 3 credits | Segment breakdown from latest 10-K |
| `list_sec_filing_markdowns` | `mcp_revelata_list_sec_filing_markdowns` | Free | Available filings for a CIK |
| `get_sec_filing_markdown` | `mcp_revelata_get_sec_filing_markdown` | 10 credits | Markdown for one SEC filing |

Revelata provides 100 free credits per month per user. Lookups (`query_company_id`, `list_kpis`) are always free. Purchase additional credits at [https://www.revelata.com/ai-credits](https://www.revelata.com/ai-credits).

## Tool selection

By default, all 8 tools are pre-checked at install time. To prune tools you don't want, use `hermes mcp configure revelata` to reopen the checklist:

```bash
hermes mcp configure revelata
```

You can also filter tools in `config.yaml`:

```yaml
mcp_servers:
  revelata:
    url: https://deepkpi-mcp.revelata.com/mcp
    auth: oauth
    tools:
      include:
        - query_company_id
        - get_company_summary
        - list_sec_filing_markdowns
        - get_sec_filing_markdown
```

## Coverage

deepKPI covers US public companies in the S&P 500, 400, and 600. International tickers (Japan, China, Hong Kong, Singapore) are enterprise-only.

## Companion skills

Revelata publishes companion skills for financial workflows — KPI pulls, filings, pressure-tests, peer benchmarks, idea generation:

```bash
hermes skills install revelata/deepkpi-agents/kpi
```

Once merged into the official optional catalog:

```bash
hermes skills install official/finance/kpi
```

## Troubleshooting

### `hermes mcp install revelata` fails with "not found in catalog"

The Revelata manifest may not be in your Hermes version yet (PR #80453 not merged upstream). Check for the manifest at `optional-mcps/revelata/manifest.yaml` in the hermes-agent repo. As a workaround, add the entry manually:

```yaml
mcp_servers:
  revelata:
    url: https://deepkpi-mcp.revelata.com/mcp
    auth: oauth
```

Then run `hermes mcp login revelata` from a fresh terminal.

### OAuth flow hangs or times out

Run `hermes mcp login revelata` from a **fresh terminal** — not from inside a running Hermes session. The session's auto-reload timeout (30s) is too short for interactive OAuth. The `login` subcommand waits up to 5 minutes.

### Tools list OK, but tool calls time out

This can happen if the OAuth probe completed without actually acquiring a token (the `tools/list` endpoint may respond without auth, making it look like login succeeded). Check that the token file exists:

```bash
ls -la ~/.hermes/mcp-tokens/revelata.json
```

If it's missing, re-run `hermes mcp login revelata`.

### "credits exhausted" or "402 Payment Required"

You have used your 100 free monthly credits. Purchase additional credits at [https://www.revelata.com/ai-credits](https://www.revelata.com/ai-credits). Free lookups (`query_company_id`, `list_kpis`, `list_sec_filing_markdowns`) still work without credits.

### Probe fails at install time (server unreachable, OAuth not complete)

The install still succeeds — no tool filter is written. Once the server is reachable and you've completed OAuth, re-run:

```bash
hermes mcp configure revelata
```

This probes the server again and lets you select tools.

### International ticker returns "not covered"

deepKPI covers US public companies (S&P 500/400/600) only. International tickers (Japan, China, Hong Kong, Singapore) require an enterprise plan. See [Revelata](https://www.revelata.com) for enterprise options.
