# Company Second Brain - Admin and Handoff Spec

## Current Public URLs

- Gateway: __PUBLIC_BASE_URL__
- User install page: __PUBLIC_BASE_URL__/install
- Admin guide: __PUBLIC_BASE_URL__/admin
- API docs: __PUBLIC_BASE_URL__/docs
- Health check: __PUBLIC_BASE_URL__/health

## VPS Layout

- Admin SSH target: __ADMIN_SSH_TARGET__
- Runtime directory: __DEPLOY_ROOT__
- Compose file: __DEPLOY_ROOT__/docker-compose.yml
- Environment file: __DEPLOY_ROOT__/.env
- Static install assets: __DEPLOY_ROOT__/services/company-ai-gateway/static

Do not expose `.env` or admin keys on public pages.

## Running Services

- company-ai-gateway: public FastAPI gateway behind Traefik
- knowledge-api: internal workspace policy, query fan-out, and ingestion API
- knowledge-worker: background Redis queue consumer that indexes documents into LightRAG
- lightrag-company-public
- lightrag-c-level
- ollama-embed: local embedding model
- postgres: metadata and workspace registry
- redis
- minio

Old upgraded servers may still have data directories for department workspaces.
They are dormant unless Compose services and API policy are re-added.

## Workspace Policy

Current query workspaces:

- `company_public`: all employee tokens can query this.
- `department_c_level`: only admins can query this.

Normal users always query only `company_public`. Existing or accidental old
groups like `department_marketing` do not grant extra query access.

Admins query both `company_public` and `department_c_level`.

## Auth Model

Admin has two auth options:

- `X-API-Key: GATEWAY_API_KEY`
- bearer token with `role=admin` / `role_admin`

Normal users use:

- `Authorization: Bearer USER_TOKEN`

User-supplied groups in `/api/query` are ignored by the gateway. The gateway
derives query groups from authenticated role on every request.

Valid groups for newly generated tokens:

- `company_all`
- `role_admin`

## Agent/User Runbook

Send a normal user or agent only:

- __PUBLIC_BASE_URL__/install
- their bearer token

Install/connect:

```bash
curl -fsSL __PUBLIC_BASE_URL__/install.sh -o install-company-second-brain.sh
bash install-company-second-brain.sh
```

The installer asks for the bearer token, installs the skill/CLI, verifies
identity, and stores config at `~/.second-brain/config.json`.

Verify access:

```bash
second-brain me
second-brain workspaces
second-brain query "toi co the xem tai lieu nao?"
```

Expected normal user workspace:

```text
company_public
```

Generic agent/IDE settings:

```text
Base URL: __PUBLIC_BASE_URL__
Auth header: Authorization: Bearer USER_TOKEN
Query endpoint: POST /api/query
Workspaces endpoint: GET /api/workspaces
```

Normal users cannot upload documents, create tokens, configure sources, or query
`department_c_level`.

## Admin Setup

Install the skill/CLI:

```bash
curl -fsSL __PUBLIC_BASE_URL__/install.sh -o install-company-second-brain.sh
bash install-company-second-brain.sh
```

Get the admin key from the VPS:

```bash
ssh __ADMIN_SSH_TARGET__ 'cd __DEPLOY_ROOT__ && grep "^GATEWAY_API_KEY=" .env'
```

Configure local admin CLI:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain config \
  --base-url __PUBLIC_BASE_URL__ \
  --admin-key "PASTE_GATEWAY_API_KEY"
```

## Create Tokens

Employee token:

```bash
second-brain admin-token-create \
  --email user@company.com \
  --name "Company User" \
  --group company_all \
  --expires-days 90
```

Admin bearer token:

```bash
second-brain admin-token-create \
  --email admin@company.com \
  --name "Admin" \
  --group role_admin \
  --admin \
  --expires-days 365
```

Send normal users only:

- __PUBLIC_BASE_URL__/install
- their bearer token

Do not send the admin key.

## Admin Operations Checklist

Daily/weekly checks:

```bash
second-brain queue-status
second-brain sources-list
docker compose ps
docker compose logs --tail=100 knowledge-worker
```

Common admin actions:

```bash
# Upload one text document
second-brain ingest-text --file ./company-handbook.md --title "Company Handbook" --target public

# Queue source scan immediately
second-brain source-scan SOURCE_ID

# Change scan interval and restart schedule from now
second-brain source-update SOURCE_ID --interval-minutes 1440 --reset-schedule

# Pause/resume source
second-brain source-update SOURCE_ID --disabled
second-brain source-update SOURCE_ID --enabled --reset-schedule
```

## Upload Documents

Public document:

```bash
second-brain ingest-text \
  --file ./company-handbook.md \
  --title "Company Handbook" \
  --target public
```

C-Level document:

```bash
second-brain ingest-text \
  --file ./board-plan.md \
  --title "Board Plan" \
  --target c_level \
  --classification restricted
```

Multipart API upload:

```bash
curl -X POST __PUBLIC_BASE_URL__/api/documents/file \
  -H "X-API-Key: PASTE_GATEWAY_API_KEY" \
  -F "file=@./company-handbook.md" \
  -F "title=Company Handbook" \
  -F "target=public"
```

MVP upload supports UTF-8 text files. Extract PDF/DOCX to text before upload, or add parser support in the gateway.

Uploads are asynchronous. `ingest-text` and `/api/documents/file` return `status: queued` and a `document_id`; `knowledge-worker` then indexes the document in the background.

Check one document:

```bash
second-brain document-status DOCUMENT_ID
```

Check queue depth and document status counts:

```bash
second-brain queue-status
```

Document statuses:

- `queued`: accepted and waiting for worker
- `indexing`: worker is sending the document to LightRAG
- `indexed`: LightRAG accepted the document
- `failed`: max retries reached; inspect `ingest_error`

Treat a document as large when extracted text is over 1MB, source file is over
10MB, PDF/DOCX is over 50 pages, or the document likely creates more than 200 chunks.
For large documents, extract text first, clean boilerplate, split into stable
sections, then upload sections.

## Scheduled Sources: Notion and Drive Public

Source scans are stored in Postgres and executed by `knowledge-worker`.

Tables:

- `document_sources`: source config, target workspace, interval, last/next scan
- `source_scan_runs`: queued/running/complete/failed scan runs
- `source_items`: dedupe map from external document id to checksum/document id

Redis queue:

```text
second_brain:source_scan_jobs
```

Create a Notion source:

```bash
second-brain source-create \
  --type notion \
  --name "Company Notion" \
  --notion-api-key "PASTE_NOTION_API_KEY" \
  --notion-page-url "https://www.notion.so/..." \
  --target public \
  --interval-minutes 360
```

Create a public Drive source:

```bash
second-brain source-create \
  --type drive_public \
  --name "Public Drive Doc" \
  --drive-url "https://docs.google.com/document/d/.../edit" \
  --target public \
  --interval-minutes 720
```

Manual scan and schedule operations:

```bash
second-brain sources-list
second-brain source-scan SOURCE_ID
second-brain source-runs SOURCE_ID
second-brain source-update SOURCE_ID --interval-minutes 1440 --reset-schedule
second-brain source-update SOURCE_ID --disabled
second-brain source-update SOURCE_ID --enabled --reset-schedule
```

Notion supports page URL/id, data source id, or search across pages shared with
the integration. Drive public MVP supports direct public Docs/Sheets/Slides or
file links; public folder listing requires Drive API/OAuth.

## Query

```bash
second-brain query "tom tat tai lieu cong ty"
```

Normal user response should show:

```text
Allowed workspaces: company_public
```

Admin response should show:

```text
Allowed workspaces: company_public, department_c_level
```

Raw API:

```bash
curl -X POST __PUBLIC_BASE_URL__/api/query \
  -H "Authorization: Bearer USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"tom tat tai lieu cong ty","mode":"mix","include_references":true}'
```

## Performance Notes

The old multi-department design queried two or more workspaces for many users.
The current MVP queries one workspace for normal users and two for admins, so
latency is lower and fewer LightRAG containers compete for RAM/CPU.

`QUERY_WORKSPACE_CONCURRENCY=4` is enough for the current two-workspace setup.

## Upgrade Flow

Local source of truth:

- `deploy/second-brain`
- `skills/productivity/company-second-brain`
- `docs/design/company-second-brain-lightrag-honcho.md`

Build the skill bundle:

```bash
deploy/second-brain/scripts/build-install-assets.sh
```

On small VPS hosts, configure 4-8GB host swap once:

```bash
sudo deploy/second-brain/scripts/configure-host-swap.sh 6G
```

Deploy:

```bash
tar --no-xattrs --exclude='.env' --exclude='__pycache__' \
  -czf /tmp/second-brain-deploy.tgz -C deploy second-brain
scp /tmp/second-brain-deploy.tgz __ADMIN_SSH_TARGET__:/tmp/second-brain-deploy.tgz
ssh __ADMIN_SSH_TARGET__ '
  set -euo pipefail
  tar -xzf /tmp/second-brain-deploy.tgz -C /opt
  find __DEPLOY_ROOT__ -name "._*" -delete
  chmod +x __DEPLOY_ROOT__/install-company-second-brain-skill.sh
  chmod +x __DEPLOY_ROOT__/scripts/build-install-assets.sh
  chmod +x __DEPLOY_ROOT__/scripts/configure-host-swap.sh
  chmod +x __DEPLOY_ROOT__/services/company-ai-gateway/static/install-company-second-brain-skill.sh
  cd __DEPLOY_ROOT__
  docker compose up -d --build --remove-orphans company-ai-gateway knowledge-api knowledge-worker lightrag-company-public lightrag-c-level
'
```

## Verification

```bash
curl -fsSL __PUBLIC_BASE_URL__/health
curl -fsSL __PUBLIC_BASE_URL__/install | head
curl -fsSL __PUBLIC_BASE_URL__/admin | head
curl -fsSIL __PUBLIC_BASE_URL__/download/company-second-brain-skill.tar.gz
curl -fsSL __PUBLIC_BASE_URL__/api/queue/status -H "X-API-Key: PASTE_GATEWAY_API_KEY"
```

Use an employee token:

```bash
second-brain me
second-brain workspaces
second-brain query "toi co the xem tai lieu nao?"
```

Check runtime services:

```bash
cd __DEPLOY_ROOT__
docker compose ps
docker ps --format '{{.Names}}' | grep second-brain-lightrag
docker compose logs -f knowledge-worker
swapon --show
```

Expected LightRAG containers:

```text
second-brain-lightrag-company-public
second-brain-lightrag-c-level
```

## Extending Beyond MVP

Add a new dedicated department workspace only when there is a real isolation or
performance reason. To do that, add a LightRAG service in Compose, add the URL
to `knowledge-api`, add `rag_workspaces` metadata, update gateway policy, update
CLI/docs, then deploy with `--remove-orphans`.

## Production Gaps

- Add proper admin web login instead of CLI-only admin setup.
- Add token revoke/rotation list with token IDs stored in Postgres.
- Add PDF/DOCX parser pipeline for upload.
- Add dynamic workspace provisioning if departments need dedicated containers again.
- Add audit log for token creation, document upload, and queries.
- Add backup/restore jobs for Postgres and LightRAG volumes.
- Rotate root password and move VPS access to SSH keys.
