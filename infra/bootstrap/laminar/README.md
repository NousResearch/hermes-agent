# Local Laminar For Hermes Back Gate

V1 assumes Laminar runs on the gateway Mac and is used only as a trace store plus SQL substrate. Hermes remains canonical for reports, proposals, review, and closure.

## Footprint

The local self-host footprint is heavier than a single app process:

- ClickHouse for trace/span data.
- Postgres for Laminar app metadata.
- Laminar app server.
- Laminar frontend.

Keep retention and sampling bounded for a single-operator gateway host. Start with `HERMES_LAMINAR_RETENTION_DAYS=14` and `HERMES_LAMINAR_SAMPLE_RATE=1.0`, then lower sampling if local storage growth is noticeable.

## Required V1 Checks

1. Start Laminar from the pinned self-host checkout:

   ```bash
   docker compose up -d
   ```

2. Confirm the dashboard responds at `http://127.0.0.1:5667`.

3. Confirm OTLP endpoints are reachable:

   - HTTP/proto or JSON endpoint: `http://127.0.0.1:8000/v1/traces`
   - gRPC endpoint: `127.0.0.1:8001`

4. Export one Hermes trace with `HERMES_LAMINAR_EXPORT_ENABLED=true`.

5. Confirm the SQL API is wired to the read-only ClickHouse client. This catches the common setup failure where the API returns `ClickHouse read-only client is not configured`.

   ```bash
   curl -sS \
     -H "Authorization: Bearer $HERMES_LAMINAR_API_KEY" \
     -H "Content-Type: application/json" \
     "$HERMES_LAMINAR_BASE_URL/v1/sql/query" \
     -d '{"query":"SELECT count() AS count FROM spans WHERE start_time >= {start_time:DateTime64(3)}","parameters":{"start_time":0}}'
   ```

   The milestone is not complete until this returns rows for real ingested trace data.

## V1 Non-Goals

- Do not set `GOOGLE_GENERATIVE_AI_API_KEY`; Laminar Signals stay disabled.
- Do not make Oryn Workspace read Laminar directly.
- Do not use Laminar as canonical state for proposals or outcomes.

## Weekly Digest Hook

Use `scripts/run_dev_signal_digest.py` as the launchd target for the weekly advisory digest. It uses a sentinel lock at `/tmp/hermes_dev_signal_digest.lock` by default, runs read -> cluster -> propose, and prints the persisted report JSON.

Example command:

```bash
cd /path/to/hermes-agent
.venv/bin/python scripts/run_dev_signal_digest.py --source laminar --window-days 7
```
