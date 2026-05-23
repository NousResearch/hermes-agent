# jcode Bridge v1 Contract

These schemas are the portable boundary between Hermes and jcode. They are
kept outside `plugins/jcode_bridge/` so a future mother repo can copy or
submodule this directory without importing Hermes Python internals.

Contract version: `jcode-bridge.v1`

Files:

- `run_json.schema.json`: one-shot `jcode run --json` final payload.
- `run_ndjson_event.schema.json`: one event from `jcode run --ndjson`.
- `run_ndjson_stream.schema.json`: the whole NDJSON stream after parsing lines
  into JSON objects.
- `debug_command.schema.json`: newline-JSON request sent to a jcode debug
  socket.
- `debug_response.schema.json`: newline-JSON response from a jcode debug
  socket.
- `upstream_sync_report.schema.json`: JSON shape emitted by
  `scripts/jcode_bridge_upstream_report.py --format json`.

The Hermes bridge also has lightweight Python validators in
`plugins/jcode_bridge/contracts.py`. Those validators are intentionally kept
stdlib-only; the schemas are for cross-repo tooling, CI, and non-Python bridge
clients.
