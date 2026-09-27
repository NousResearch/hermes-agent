# litco-agent: the matter host

`litco-agent` is a fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) that runs as the agent for one litigation matter on its own machine. A firm's LitKit instance drives it over HTTP through the **turn server** described here. Everything LitCo adds lives in the top-level `litco/` package and the `plugins/platforms/litco_turn/` platform plugin, so the upstream tree stays syncable (`git fetch upstream && git merge upstream/main`).

## How it runs

The turn server is a Hermes gateway platform named `litco_turn`. It is a bundled platform plugin, so the gateway discovers it with no change to core code, and it runs in the same process as cron, memory, skills, and the other channels (Slack, Telegram). The gateway enables it whenever `LITCO_HOST_SECRET` is set, or when `gateway.platforms.litco_turn.enabled` is true in `config.yaml`.

Each turn runs as a Hermes `AIAgent` built the same way the API-server adapter builds one: the provider and model come from the profile's config, the toolsets come from `platform_toolsets.litco_turn`, and the transcript is the profile's SessionDB. The agent's streaming and tool callbacks are translated into LitKit's agent2 events (see `litco/hermes_runner.py`).

The server can also run on its own, as a sidecar started by the same systemd unit: `python -m litco.turn_server --host 127.0.0.1 --port 8765`. The gateway platform is the intended mode.

## Environment

| Variable | Required | Meaning |
|---|---|---|
| `LITCO_HOST_SECRET` | yes | Shared secret LitKit sends in `X-Host-Secret`. It is also the HMAC key for user assertions. With no secret, the server refuses every request. |
| `LITCO_MATTER_ID` | yes | The one LitKit matter this host serves. A turn for any other matter gets 403. |
| `LITCO_MATTER_HOME` | no | Root of the per-thread working directories. Default `~/matter`. |
| `LITCO_AGENT_TOKEN` | no | The matter-pinned LitKit bearer token (`lkm_…`). The server sends it when fetching attachments. |
| `LITCO_INSTANCE_URL` | no | Base URL of the firm's LitKit instance, for LitKit tools. |
| `LITCO_TURN_HOST` | no | Bind address. Default `127.0.0.1`. |
| `LITCO_TURN_PORT` | no | Port. Default `8765`. |

## Authentication

Every endpoint except `GET /health` requires `X-Host-Secret`, compared in constant time with `LITCO_HOST_SECRET`.

A request may also name the human it acts for. It then sends both `X-LitKit-Acting-User: <userId>` and `X-LitKit-User-Assertion: <assertion>`. The assertion uses the wire format LitKit already mints in `src/lib/agent-daemon/user-assertion.ts`:

```
v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>.<base64url HMAC-SHA256>
```

The MAC covers `v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>` and is keyed on the host secret. Both times are milliseconds, as in the TypeScript code. The server rejects the request with 401 when the MAC fails, the assertion has expired (60 seconds of clock skew allowed), it names another matter, the header user differs from the asserted user, or only one of the two headers is present. When an assertion verifies, the body's `userId` must equal the asserted user (403 otherwise). A request with neither header is a service request, such as cron work.

## Endpoints

### `POST /turn`

Body:

```json
{
  "matterId": "…", "userId": "…", "sessionId": "…", "text": "…",
  "attachments": [{"fileId": "…", "mime": "application/pdf", "filename": "complaint.pdf", "url": "https://…"}],
  "channel": "slack" | "web" | "telegram",
  "kind": "channel" | "dm",
  "budgetMs": 600000
}
```

`matterId`, `sessionId`, and `text` are required. `channel` defaults to `web` and `kind` to `channel`. A `dm` turn requires `userId`. `budgetMs` is optional; without it, a turn has no time or token ceiling.

The response is `text/event-stream`. The header `X-Turn-Id` carries the turn id. Each frame is `event: <type>` followed by one `data:` line of JSON, and every payload carries `type`, `turnId`, `stepId` (1, 2, 3, … within the turn), and `ts` (milliseconds since the epoch). A `: keepalive` comment is sent every 15 seconds of silence. The frames, in the order they can occur:

| Event | Fields | When |
|---|---|---|
| `goal_accepted` | `sessionId` | Always first. |
| `assistant_delta` | `delta` | Streamed answer text. |
| `assistant_reset` | `reason` | The text streamed so far was interim commentary (for example, before a tool call). Discard it; later deltas start fresh. |
| `tool_started` | `call{toolCallId,name}`, `args` | A tool call begins. |
| `tool_progress` | `call`, `message` | Progress from a long tool, such as a delegated subagent. |
| `tool_complete` | `call`, `result{status,summary,durationMs}` | A tool call ends. `status` is `ok` or `error`. The summary never carries the tool's output, only a tool-supplied summary, an error message, or the size of the result. |
| `error_classified` | `category`, `message`, `recovery` | The turn failed. |
| `loop_halted` | `reason`, `explanation` | The turn stopped early: `interrupted`, `budget_exhausted`, or `shutdown`. |
| `final` | `text`, `citations`, `usage{inputTokens,outputTokens,cacheReadTokens?,cacheWriteTokens?}`, `durationMs`, `modelUsed?`, `deliverables?` | Always last. |

Hermes has no plan events, so `plan_drafted` and `plan_step_*` are never sent.

`final.deliverables` lists every file the turn created or changed under the thread's `deliverables/` folder, as `{fileId, filename, mime, path}`. `path` is relative to `LITCO_MATTER_HOME`. `fileId` encodes that path and can be fetched from `GET /deliverables/{fileId}`.

If the client disconnects, the turn keeps running and its transcript lands in the session. `POST /interrupt/{turnId}` still stops it.

### `POST /interrupt/{turnId}`

Stops the turn. Returns 202 while the turn winds down, 200 if it had already finished, and 404 for an unknown id. The turn's stream ends with `loop_halted{reason:"interrupted"}` and `final`.

### `GET /health`

Returns `{ok, version, hermesVersion, uptimeSeconds, activeTurns, matterId}`. No secret is required.

### `GET /deliverables/{fileId}`

Returns the bytes of a file listed in `final.deliverables`. Ids that resolve outside a `deliverables/` folder get 404.

## Sessions and concurrency

One LitKit thread is one `sessionId`, and one `sessionId` is one Hermes session. A second turn on the same `sessionId` continues the conversation. Hermes may rotate its internal session id when it compacts a long conversation; the mapping from `sessionId` to the current Hermes id is kept in `$HERMES_HOME/litco_sessions.json`, so the thread follows the rotation.

Turns on the same `sessionId` run one at a time, in arrival order. Turns on different `sessionId`s run concurrently. There is no global queue.

## Working directories

The server creates these under `LITCO_MATTER_HOME` on first use:

```
shared/                   channel threads (the whole case team)
  deliverables/
users/<userId>/           dm threads (one lawyer's private work)
  deliverables/
inbox/<turnId>/           attachments fetched for one turn
```

A turn's working directory is `shared/` for a `channel` thread and `users/<userId>/` for a `dm` thread. The terminal and file tools start there, so a relative path such as `deliverables/memo.docx` lands in the thread's own folder.

Attachments that carry a `url` are downloaded into `inbox/<turnId>/` with `Authorization: Bearer $LITCO_AGENT_TOKEN`, and the agent is told where each one landed. An attachment without a `url`, or one that fails to download, is named in the prompt with its LitKit `fileId`.

## Code map

| Path | Role |
|---|---|
| `litco/turn_server.py` | aiohttp app: auth, parsing, SSE framing, per-session locks, budgets, interrupts, deliverables. |
| `litco/hermes_runner.py` | Builds the `AIAgent` for a turn and maps Hermes callbacks to events. |
| `litco/assertion.py` | Host-secret comparison and the user-assertion MAC. |
| `litco/homes.py` | Working-directory layout and deliverable ids. |
| `plugins/platforms/litco_turn/` | Registers the `litco_turn` gateway platform. |
| `tests/litco/` | Contract tests with a fake runner, and runner tests with a fake agent. |

## Testing

```
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -e ".[messaging]" --group dev
.venv/bin/python -m pytest tests/litco -q
```
