# ttm-control-plane

PR-F-H1 of the [Hermes alignment plan](../../../developer-handbook/docs/control-plane). Receives runtime dispatches from TTM's `HermesAdapter`, validates the principal-scoped payload, binds the run to a Hermes session, and reports `run.dispatched` back to TTM ingress.

## Wire contract

Mounted at `/api/plugins/ttm-control-plane/` on the Hermes dashboard (`127.0.0.1:9119` by default).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Plugin metadata + binding count (unauthenticated). |
| POST | `/runs/dispatch` | Initial run-spawn dispatch from TTM. Returns 202 + `runtime_run_ref`. |
| GET | `/runs/{ref}/status` | Last-known status for a previously-dispatched run. |
| POST | `/runs/{ref}/stop` | Tear down the binding so a follow-on dispatch can rebind. |

The dispatch body mirrors TTM's `RuntimeDispatchPayload` plus `runtime_id`, `ingress_base_url`, and `principal_token`. See [`RUNTIME-ADAPTER-CONTRACT.md`](https://github.com/you-kol/developer-handbook/blob/main/docs/control-plane/RUNTIME-ADAPTER-CONTRACT.md) and [`RUNTIME-PRINCIPAL-CONTRACT.md`](https://github.com/you-kol/developer-handbook/blob/main/docs/control-plane/RUNTIME-PRINCIPAL-CONTRACT.md).

## Auth

Shared-secret header `X-TTM-Control-Plane-Secret` whose value matches the `TTM_CONTROL_PLANE_SECRET` env var. The dashboard's general auth middleware deliberately bypasses `/api/plugins/*`; this plugin owns its own check.

If `TTM_CONTROL_PLANE_SECRET` is unset, the plugin runs unauthenticated (dev/CI fallback). Production deployment must set the env var on both sides:

```bash
# Hermes side — load on dashboard service start
echo 'TTM_CONTROL_PLANE_SECRET="<long-random-value>"' >> ~/.hermes/.env

# TTM side — Doppler
doppler secrets set TTM_CONTROL_PLANE_SECRET=<long-random-value> --project ttm
doppler secrets set HERMES_GATEWAY_URL=http://127.0.0.1:9119/api/plugins/ttm-control-plane --project ttm
```

## Service install

The plugin lives inside the dashboard FastAPI app, so the dashboard must run continuously. The provided launchd plist runs it as a per-user agent:

```bash
cp launchd/ai.hermes.dashboard.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.hermes.dashboard.plist
launchctl list | grep ai.hermes.dashboard
curl -s http://127.0.0.1:9119/api/plugins/ttm-control-plane/health | jq .
```

The plist runs `hermes_cli.main dashboard --no-browser --port 9119` with `KeepAlive` on non-success and writes to `~/.hermes/logs/dashboard.{log,error.log}`.

## Status of original H1 deferrals

- **Headless agent spawn**: landed. `_spawn_headless_session` spawns `hermes chat -q ... -Q --max-turns 200` with `TTM_RUN_ID`, `TTM_PRINCIPAL_TOKEN`, `TTM_INGRESS_BASE_URL`, and `TTM_RUNTIME_ID` injected as env vars. Failure to resolve the binary or missing token logs and skips — never crashes the dispatch route. Set `TTM_CONTROL_PLANE_DISABLE_SPAWN=1` to suppress the spawn (tests/dev).
- **Persistence**: landed. The binding registry is SQLite-backed at `~/.hermes/state.db` (override via `TTM_CONTROL_PLANE_DB_PATH`). Binding metadata survives dashboard restarts. The principal token is **never** persisted — after a restart the operator must trigger a TTM rebind to issue a fresh token, which arrives via `/runs/{run_id}/rebind-token`.
- **TTM rebind alignment**: landed. TTM's `POST /control-plane/{run_id}/rebind` returns `token=None` in the response (operator never sees plaintext); the new token flows directly to the plugin via `notify_rebind` → `/runs/{run_id}/rebind-token`.

## Still deferred (H4 and beyond)

- **Pause / resume / retry-slice routes**: the plugin does not expose `/runs/{ref}/pause`, `/resume`, or `/slices/{slice_id}/retry`. TTM's `HermesAdapter` returns `unsupported` for these (PR #658) until H4 lands the matching surface. Stop is supported — `POST /runs/{ref}/stop` tears down the binding for re-dispatch.

## Tests

```bash
cd ~/.hermes/hermes-agent
venv/bin/python -m pytest tests/plugins/test_ttm_control_plane_plugin.py -v
```
