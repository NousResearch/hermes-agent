# /api/jobs HTTP Surface

The `/api/jobs` endpoints expose full CRUD management for cron jobs (scheduled routines) over
HTTP. They are served by the `APIServerAdapter` in `gateway/platforms/api_server.py` and mirror
the functionality available through the `routines-ipc.ts` IPC channel used by the desktop client.

## Authentication

When `API_SERVER_KEY` is configured, every request must include a Bearer token:

```
Authorization: Bearer <your-api-key>
```

Requests without a valid token receive `401 Unauthorized`. If no key is configured, all requests
are accepted without authentication.

## Availability

All endpoints return `501 Not Implemented` with `{"error": "Cron module not available"}` when the
underlying cron subsystem (`_CRON_AVAILABLE`) is not loaded.

## Endpoints

| Method   | Path                        | Description                                      |
|----------|-----------------------------|--------------------------------------------------|
| `GET`    | `/api/jobs`                 | List all enabled jobs (or all with `?include_disabled=true`) |
| `POST`   | `/api/jobs`                 | Create a new cron job                            |
| `GET`    | `/api/jobs/{id}`            | Fetch a single job by ID                         |
| `PATCH`  | `/api/jobs/{id}`            | Update allowed fields on an existing job         |
| `DELETE` | `/api/jobs/{id}`            | Delete a job permanently                         |
| `POST`   | `/api/jobs/{id}/pause`      | Pause a job (disables scheduling)                |
| `POST`   | `/api/jobs/{id}/resume`     | Resume a paused job                              |
| `POST`   | `/api/jobs/{id}/run`        | Trigger immediate execution of a job             |

### Job ID format

All `{id}` path parameters must be lowercase hexadecimal strings. Non-hex values return
`400 Bad Request` with `{"error": "Invalid job_id"}` and log the request metadata for
security investigation.

## Request / Response Fields

### Create (`POST /api/jobs`) — request body

| Field      | Type    | Required | Notes                                          |
|------------|---------|----------|------------------------------------------------|
| `name`     | string  | yes      | Max 200 characters                             |
| `schedule` | string  | yes      | Cron expression, e.g. `"*/5 * * * *"`         |
| `prompt`   | string  | no       | Max 5 000 characters                           |
| `deliver`  | string  | no       | Delivery target, default `"local"`             |
| `skills`   | array   | no       | List of skill identifiers to enable            |
| `repeat`   | integer | no       | Positive integer; how many times to repeat     |

### Update (`PATCH /api/jobs/{id}`) — allowed fields

`name`, `schedule`, `prompt`, `deliver`, `skills`, `skill`, `repeat`, `enabled`

These match the `_UPDATE_ALLOWED_FIELDS` whitelist enforced by `gateway/platforms/api_server.py`.
Any other keys (for example `model`, `provider`, `base_url`, `script`, `no_agent`,
`context_from`, `enabled_toolsets`, `workdir`) are silently dropped from the sanitized
payload. A body containing only disallowed/unknown fields returns `400 Bad Request`
with `{"error": "No valid fields to update"}`.

### Job object (response)

| Field                 | Type    | Notes                                         |
|-----------------------|---------|-----------------------------------------------|
| `id`                  | string  | Hex job identifier                            |
| `name`                | string  | Human-readable label                          |
| `prompt`              | string  | Prompt sent to the agent on execution         |
| `schedule`            | string  | Cron expression                               |
| `schedule_display`    | string  | Human-readable schedule description           |
| `enabled`             | boolean | Whether the job is actively scheduled         |
| `paused_at`           | string  | ISO-8601 timestamp set when paused            |
| `paused_reason`       | string  | Optional reason string                        |
| `created_at`          | string  | ISO-8601 creation timestamp                   |
| `next_run_at`         | string  | ISO-8601 next scheduled execution             |
| `last_run_at`         | string  | ISO-8601 last execution timestamp             |
| `last_status`         | string  | Status of last run (`ok`, `error`, ...)       |
| `last_error`          | string  | Error message from last run, if any           |
| `last_delivery_error` | string  | Delivery-layer error from last run, if any    |
| `deliver`             | string  | Delivery target                               |
| `origin`              | object  | Platform/chat context that created the job    |
| `skills`              | array   | Enabled skill identifiers                     |
| `model`               | string  | Override model for this job                   |
| `provider`            | string  | Override provider for this job                |
| `base_url`            | string  | Override base URL for this job                |
| `script`              | string  | Script path (alternative to prompt)           |
| `no_agent`            | boolean | Run without agent wrapper when true           |
| `context_from`        | string  | Context source identifier                     |
| `repeat`              | integer | Repeat count                                  |
| `enabled_toolsets`    | array   | Toolsets available during execution           |
| `workdir`             | string  | Working directory for execution               |

### Action responses

- **pause / resume / run**: `{"job": <job-object>}`
- **delete**: `{"ok": true}`
- **list**: `{"jobs": [<job-object>, ...]}`
- **create / get / update**: `{"job": <job-object>}`

## Usage Example

```bash
# Create a job
curl -X POST http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"nightly-report","schedule":"0 3 * * *","prompt":"Generate the daily summary"}'

# List jobs
curl http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $API_KEY"

# Pause a job
curl -X POST http://localhost:8080/api/jobs/aabbccddeeff/pause \
  -H "Authorization: Bearer $API_KEY"

# Run immediately
curl -X POST http://localhost:8080/api/jobs/aabbccddeeff/run \
  -H "Authorization: Bearer $API_KEY"

# Delete
curl -X DELETE http://localhost:8080/api/jobs/aabbccddeeff \
  -H "Authorization: Bearer $API_KEY"
```

## Compatibility with routines-ipc.ts

The `routines-ipc.ts` multi-key normalization layer accepts both `skill` (singular) and `skills`
(plural) field names, as well as camelCase variants such as `scheduleDisplay`, `nextRunAt`,
`lastRunAt`, `lastStatus`, `lastError`, `enabledToolsets`. The HTTP surface stores and returns the
snake_case canonical names; the IPC layer handles translation for desktop clients.
