# Hermes M3 and Hermes Spark Shared Tool Broker

Snapshot: 2026-04-22

## Host roles

- **openclaw-gcp** is the OneCLI authority and secret boundary. Provider secrets and OneCLI injection rules live there.
- **rj-spark** runs Hermes Spark, Ollama/Qwen3-Next, the shared tool broker, and the `hermes-spark-proxy` OneCLI client identity.
- **rj-m3max** runs Hermes M3 as a lightweight client. It uses HTTPS MCP and model endpoints on `rj-spark`; it does not run provider daemons or store provider secrets.

`deme-proxy` remains separate from `hermes-spark-proxy`. Keep `deme-proxy` for Deme if Deme makes direct provider/API calls. Hermes M3 does not need its own OneCLI provider proxy for the shared broker tools.

## Architecture

Hermes M3 and Hermes Spark share one broker on `rj-spark`:

```text
Hermes M3
  -> https://rj-spark.tailc13f7e.ts.net/mcp/*
  -> rj-spark shared tool broker
  -> OneCLI proxy identity: hermes-spark-proxy
  -> openclaw-gcp OneCLI authority/injection proxy
  -> provider APIs/MCP services
```

Hermes Spark uses the same broker locally:

```text
Hermes Spark
  -> http://127.0.0.1:8767/mcp/*
  -> shared tool broker
```

## MCP endpoints

Hermes Spark local endpoints:

- `http://127.0.0.1:8767/mcp/grain`
- `http://127.0.0.1:8767/mcp/granola`
- `http://127.0.0.1:8767/mcp/notion-api`
- `http://127.0.0.1:8767/mcp/gws-api`
- `http://127.0.0.1:8767/mcp/zoom-api`
- `http://127.0.0.1:8767/mcp/affinity-api`

Hermes M3 target endpoints:

- `https://rj-spark.tailc13f7e.ts.net/mcp/grain`
- `https://rj-spark.tailc13f7e.ts.net/mcp/granola`
- `https://rj-spark.tailc13f7e.ts.net/mcp/notion-api`
- `https://rj-spark.tailc13f7e.ts.net/mcp/gws-api`
- `https://rj-spark.tailc13f7e.ts.net/mcp/zoom-api`
- `https://rj-spark.tailc13f7e.ts.net/mcp/affinity-api`

Use the MagicDNS HTTPS hostname, not `https://100.65.197.14`. Tailscale Serve is bound to hostname/SNI and the raw IP fails TLS.

## Tailscale Serve

Current route layout:

```text
https://rj-spark.tailc13f7e.ts.net/
|-- /                 proxy http://100.65.197.14:11434
|-- /mcp/grain        proxy http://127.0.0.1:8767/mcp/grain
|-- /mcp/granola      proxy http://127.0.0.1:8767/mcp/granola
|-- /mcp/gws-api      proxy http://127.0.0.1:8767/mcp/gws-api
|-- /mcp/zoom-api     proxy http://127.0.0.1:8767/mcp/zoom-api
|-- /mcp/notion-api   proxy http://127.0.0.1:8767/mcp/notion-api
|-- /mcp/affinity-api proxy http://127.0.0.1:8767/mcp/affinity-api
```

Apply or repair the route layout with:

```bash
cd ~/.hermes/hermes-agent
./scripts/configure_tailscale_mcp_routes.sh
```

If Tailscale denies route updates, run this once:

```bash
sudo tailscale set --operator=$USER
```

## Secret model

- OneCLI on `openclaw-gcp` is the preferred secret boundary.
- Hermes M3 stores only endpoint declarations.
- Hermes Spark stores only broker-local metadata when unavoidable.
- `~/.config/onecli/hermes-spark-proxy.env` contains proxy/token/trust material for the `hermes-spark-proxy` client identity.
- `~/.hermes/tool-broker.env` may contain non-secret or fallback-only broker metadata.
- Google Workspace currently uses Spark-local OAuth files: `~/.hermes/google_client_secret.json` and `~/.hermes/google_token.json`.
- Zoom uses Spark-local `ZOOM_ACCOUNT_ID` as non-secret token-mint metadata. OneCLI secret `oc_Zoom_S2S_OAuth` on `openclaw-gcp` injects `Authorization: Basic <base64(client_id:client_secret)>` for `zoom.us` `/oauth/token`.

## Provider implementation

MCP-backed:

- Grain: `npx -y mcp-remote https://api.grain.com/_/mcp`
- Granola: `npx -y mcp-remote https://mcp.granola.ai/mcp`

API-backed:

- Notion REST API via OneCLI HTTPS auth injection.
- Google Workspace via Spark-local OAuth token files and deterministic wrapper tools.
- Zoom REST API via broker-minted server-to-server OAuth bearer, with Basic auth injected by OneCLI on the token request.
- Affinity REST API via OneCLI HTTPS auth injection.

Grain OAuth refresh is handled by the upstream OneCLI/openclaw-gcp path. The broker does not run a Spark-side Grain refresh timer.

## Tool namespaces

Grain:

- `grain.meetings.search`
- `grain.meeting.get`
- `grain.transcript.get`
- `grain.highlights.list`
- `grain.notes.get`

Granola:

- `granola.meetings.search`
- `granola.meeting.get`
- `granola.transcript.get`
- `granola.notes.get`
- `granola.folders.list`

Notion:

- `notion.search`
- `notion.page.get`
- `notion.page.create`
- `notion.page.update_properties`
- `notion.blocks.list`
- `notion.blocks.append`
- `notion.blocks.replace_range`
- `notion.blocks.patch_text`
- `notion.database.query`
- `notion.database.upsert_page`

Google Workspace:

- `gws.gmail.search`
- `gws.gmail.get`
- `gws.gmail.send`
- `gws.calendar.events.search`
- `gws.calendar.freebusy`
- `gws.calendar.event.create`
- `gws.drive.search`
- `gws.drive.file.get`
- `gws.docs.get`
- `gws.docs.patch`
- `gws.sheets.read`
- `gws.sheets.update`

Zoom:

- `zoom.meetings.list`
- `zoom.meeting.get`
- `zoom.meeting.create`
- `zoom.recordings.list`
- `zoom.recording.get`
- `zoom.recording.transcript_get`
- `zoom.users.get`

Affinity:

- `affinity.person.search`
- `affinity.person.get`
- `affinity.person.upsert`
- `affinity.organization.search`
- `affinity.organization.get`
- `affinity.organization.upsert`
- `affinity.opportunity.search`
- `affinity.opportunity.get`
- `affinity.opportunity.update_stage`
- `affinity.note.create`

## Validation

Connection tests on Hermes Spark:

```bash
hermes mcp test grain
hermes mcp test granola
hermes mcp test notion_api
hermes mcp test gws_api
hermes mcp test zoom_api
hermes mcp test affinity_api
```

All six connection tests pass.

Functional smoke tests validated:

- Notion `notion.search` returned real Notion page results through OneCLI injection.
- Google Workspace `gws.gmail.search` returned a recent Gmail result.
- Zoom `zoom.meetings.list` minted a bearer through OneCLI injection and returned scheduled meetings.
- Affinity `affinity.person.search` returned real CRM contact results through OneCLI injection.
- Granola OAuth completed. `granola.folders.list` reaches the provider and can return the provider limitation `Meeting folders are only available to paid Granola tiers`.
- Remote Tailscale HTTPS MCP discovery works for `https://rj-spark.tailc13f7e.ts.net/mcp/gws-api`.

Run broker tests:

```bash
cd ~/.hermes/hermes-agent
venv/bin/python -m pytest tests/shared_tool_broker
```

## Rollback

- Keep existing Hermes MCP entries until replacement endpoints are validated.
- Disable one broken MCP server with `enabled: false`; do not revert the whole broker.
- Stop the broker with `systemctl --user disable --now hermes-shared-tool-broker.service`.
- Remove only the new broker-backed `mcp_servers` entries if rolling back Hermes config.
