# Hermes GitHub Integration (P1)

P1 adds a secure GitHub metadata + webhook + ChatOps bridge for Hermes Code
Mode. This port is backend/API/CLI/docs/tests only.

`hermesWeb/` is not present in this repository, and deprecated `web/` was not
used for new UI work.

## Capabilities

- Detect GitHub App configuration and expose safe status.
- Optional local/dev PAT fallback gated by:
  - `HERMES_GITHUB_DEV_PAT`
  - `HERMES_GITHUB_ALLOW_DEV_PAT=1`
- Sync repository, branch, issue, and pull request metadata.
- Validate GitHub webhook signatures before processing.
- Persist webhook deliveries and deduplicate by delivery ID.
- Parse ChatOps commands from comments:
  - `@hermes plan`
  - `@hermes review`
  - `@hermes fix`
  - `@hermes explain`
  - `@hermes status`
- Create P0 `ArtifactLedger` artifacts and `AgentOrchestrator` runs from
  ChatOps requests.
- Prepare PR metadata (no auto-push, no auto-merge).
- Support GitHub write safety via persistent Code Mode approval requests.

## GitHub App Setup

Recommended minimum repository permissions:

- Metadata: read
- Contents: read
- Issues: read/write
- Pull requests: read/write
- Checks: read/write
- Commit statuses: read/write

Supported webhook events:

- `installation`
- `installation_repositories`
- `issues`
- `issue_comment`
- `pull_request`
- `pull_request_review`
- `pull_request_review_comment`
- `check_suite`
- `check_run`
- `push`

Example env variable names:

```bash
HERMES_GITHUB_APP_ID=
HERMES_GITHUB_APP_PRIVATE_KEY_PATH=
HERMES_GITHUB_WEBHOOK_SECRET=
HERMES_GITHUB_DEV_PAT=
HERMES_GITHUB_ALLOW_DEV_PAT=
```

Do not commit `.env` values.

## Security Model

- Private key contents are loaded from file path and are not persisted in DB.
- Installation tokens are cached in memory with expiry metadata.
- Webhook processing requires valid `X-Hub-Signature-256`.
- Sensitive values are redacted from errors/log-safe strings.
- GitHub write actions require approved persistent approval requests.
- P1 does not auto-merge, force-push, delete branches, or modify repository
  settings/permissions.

## API Endpoints

- `GET /api/code/github/status`
- `GET /api/code/github/installations`
- `GET /api/code/github/repositories`
- `POST /api/code/github/repositories/sync`
- `GET /api/code/github/repositories/{owner}/{repo}`
- `GET /api/code/github/repositories/{owner}/{repo}/issues`
- `GET /api/code/github/repositories/{owner}/{repo}/pulls`
- `POST /api/code/github/webhooks`
- `POST /api/code/github/chatops/{command_id}/run`
- `POST /api/code/github/comments`
- `POST /api/code/github/pull-requests/prepare`

GitHub write endpoints now follow a two-step flow:

1. Call without `approval_id` to receive `requires_approval: true` and a new
   persistent `approval_id`.
2. Approve the request via `/api/code/approvals/{approval_id}/approve`, then
   call the same write endpoint with `approval_id` to execute once.

Replay protection verifies action kind + resource + payload binding; executed
approvals cannot be reused.

P3 adds realtime fanout for GitHub-related Code Mode events through
`WS /api/code/events/ws` and REST event filters in `/api/code/events*`.

`/api/code/github/webhooks` is public only for GitHub delivery. It is protected
by webhook HMAC validation, not by dashboard session token.

## CLI

Minimal CLI support:

- `/github status`
- `/github repos`
- `/github sync`

## Schema

P3 keeps schema at `v17` and reuses `code_events` for realtime replay/fanout.

P2/P3 related tables include:

- `github_app_installations`
- `github_repositories`
- `github_branches`
- `github_issues`
- `github_pull_requests`
- `github_webhook_deliveries`
- `github_chatops_commands`
- `github_status_reports`
- `code_approval_requests`

## Local Webhook Testing

Expose local backend with a tunnel (for example ngrok/cloudflared) and set
GitHub App webhook URL:

`https://<tunnel-host>/api/code/github/webhooks`

Use the same secret in GitHub App settings and `HERMES_GITHUB_WEBHOOK_SECRET`.

## Non-goals in P1

- HermesWeb GitHub panel
- Autonomous coding loop
- SSH/VPS flows
- Desktop app integration
- Auto PR merge/push/delete operations
