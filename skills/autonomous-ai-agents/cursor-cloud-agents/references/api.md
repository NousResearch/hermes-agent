# Cursor Cloud Agents API — surface used by this skill

Authoritative sources, both maintained by Cursor:

- Endpoint docs: https://cursor.com/docs/cloud-agent/api/endpoints
- OpenAPI spec: https://cursor.com/docs-static/cloud-agents-openapi.yaml

Base URL: `https://api.cursor.com` (no host override in the helper).

## Authentication

```http
Authorization: Basic <base64(api-key + ":")>
```

The API key is the Basic username with an empty password. `Authorization: Bearer <key>` is accepted as an equivalent. Keys come from https://cursor.com/dashboard → Settings → API Keys, and are read here from the `CURSOR_API_KEY` environment variable.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/agents` | Create an agent and enqueue its first run |
| GET | `/v1/agents` | List agents (newest first; `limit`, `cursor`, `prUrl`, `includeArchived`) |
| GET | `/v1/agents/{id}` | Durable agent metadata (execution status lives on runs) |
| DELETE | `/v1/agents/{id}` | Permanent delete; prefer archive |
| POST | `/v1/agents/{id}/runs` | Follow-up run on an existing agent |
| GET | `/v1/agents/{id}/runs` | List runs (newest first) |
| GET | `/v1/agents/{id}/runs/{runId}` | One run's status and timestamps |
| GET | `/v1/agents/{id}/runs/{runId}/stream` | Server-sent events for one run |
| POST | `/v1/agents/{id}/runs/{runId}/cancel` | Cancel the active run (terminal) |
| GET | `/v1/agents/{id}/usage` | Per-run token usage (early access; `403 feature_unavailable`) |
| POST | `/v1/agents/{id}/archive` · `/unarchive` | Reversible removal |
| GET | `/v1/agents/{id}/artifacts` · `/artifacts/download` | Run artifacts |
| GET | `/v1/me` | Identity of the API key |
| GET | `/v1/models` | Recommended models; use `id` (and optional `params`) |
| GET | `/v1/repositories` | GitHub repos reachable through Cursor's GitHub App |
| POST | `/v1/sub-tokens` | One-hour user-scoped worker token (My Machines) |

## Create agent request

Only `prompt` is required. The fields the helper sets:

```json
{
  "prompt": { "text": "Add a README with setup instructions" },
  "repos": [{ "url": "https://github.com/acme/widgets", "startingRef": "main" }],
  "model": { "id": "composer-2" },
  "name": "Add README",
  "mode": "agent",
  "workOnCurrentBranch": false,
  "autoCreatePR": false
}
```

- `repos[].url` is always required on a repo entry; `startingRef` may be a branch or a commit SHA.
- `repos[].prUrl` makes the agent work on that PR's branches; `startingRef` is then ignored.
- `workOnCurrentBranch: false` (default) pushes to a new `cursor/...` branch. `true` pushes directly to the starting ref.
- `autoCreatePR` opens a pull request on completion; `skipReviewerRequest` only applies alongside it.
- `repos` is mutually exclusive with a named cloud environment (`env: {"type": "cloud"|"pool"|"machine", "name": "..."}`).
- Omit `repos` (or pass `[]`) for a repo-less agent.
- `envVars` are session-scoped, encrypted at rest, cannot be combined with a client-supplied `agentId`, and names cannot start with `CURSOR_`. The field is beta and may be silently ignored.

Response: `{"agent": {...}, "run": {...}}`. Keep the agent id — follow-ups and cancellation are agent-scoped.

## Run object

| Field | Meaning |
|-------|---------|
| `id` | Run id (`run-...`) |
| `agentId` | Owning agent |
| `status` | `CREATING`, `RUNNING`, `FINISHED`, `ERROR`, `CANCELLED`, `EXPIRED` |
| `createdAt` / `updatedAt` / `durationMs` | Timestamps; duration populated once terminal |
| `result` | Final assistant reply, populated once terminal |
| `git.branches` / `git.prs` | Pushed branches and PR URLs; per-agent state, not per-run |

Terminal statuses: `FINISHED`, `ERROR`, `CANCELLED`, `EXPIRED`. Everything else is in flight.

## Follow-up run request

```json
{ "prompt": { "text": "Also add troubleshooting steps" }, "mode": "plan" }
```

`mode` is optional on follow-ups: omitting it keeps the conversation's current mode. Only one run may be active per agent — a second create returns `409`.

## Cancellation

`POST /v1/agents/{id}/runs/{runId}/cancel` transitions the run to `CANCELLED`. Cancelling a run that is already terminal or was never active returns `409 run_not_cancellable`; continue the conversation by creating a new run instead.

## Streaming (not used by the helper)

`GET /v1/agents/{id}/runs/{runId}/stream` emits SSE events `status`, `assistant`, `thinking`, `tool_call`, `interaction_update`, `heartbeat`, `result`, `error`, `done`. Reconnect with the `Last-Event-ID` header; an event id from a different run returns `400 invalid_last_event_id`. `X-Cursor-Stream-Retention-Seconds` gives the retention window, after which the endpoint may return `410 stream_expired`. Polling `GET .../runs/{runId}` sidesteps all of this, which is why `watch` polls.

## Errors

| Status | Typical cause |
|--------|---------------|
| `400` | Malformed body, unknown pool name, `startingRef` without a repo |
| `401` | Missing/invalid API key |
| `403` | Key lacks access, or `feature_unavailable` (e.g. usage endpoint) |
| `404` | Unknown agent/run id |
| `409` | Active run already exists, `run_not_cancellable`, `agent_id_conflict` |
| `410` | `stream_expired` |
| `429` | Rate limited. `/v1/repositories` is 1 request/user/minute and 30/hour — cache it |

The helper surfaces these as `error: <METHOD> <path> failed: HTTP <code> <body>` on stderr and exits `2`.
