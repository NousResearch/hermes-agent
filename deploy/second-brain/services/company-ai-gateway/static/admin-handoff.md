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
- knowledge-api: internal RAG routing and ingestion API
- knowledge-worker: background Redis queue consumer that indexes documents into LightRAG
- lightrag-company-public
- lightrag-company-internal
- lightrag-marketing
- lightrag-financial
- lightrag-hr
- lightrag-engineering
- lightrag-c-level
- ollama-embed: local embedding model
- postgres: metadata and workspace registry
- redis
- minio

## Auth Model

Admin has two auth options:

- `X-API-Key: GATEWAY_API_KEY`
- Bearer token with `role=admin`

Normal users use:

- `Authorization: Bearer USER_TOKEN`

User access is decided by token groups. User-supplied groups in `/api/query` are ignored by the gateway for bearer-token requests.

## Groups and Workspaces

Current groups:

- `department_marketing`
- `department_financial`
- `department_hr`
- `department_engineering`
- `department_c_level`
- `role_admin`

Current query workspaces:

- `company_public`
- `company_internal`
- `department_marketing`
- `department_financial`
- `department_hr`
- `department_engineering`
- `department_c_level`

Every normal user can query `company_public` and `company_internal`, plus their department workspaces.

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

## Create User Tokens

Marketing:

```bash
second-brain admin-token-create \
  --email user@company.com \
  --name "Marketing User" \
  --group department_marketing \
  --expires-days 90
```

Admin bearer token:

```bash
second-brain admin-token-create \
  --email admin@company.com \
  --name "Admin" \
  --group role_admin \
  --group department_marketing \
  --group department_financial \
  --group department_hr \
  --group department_engineering \
  --group department_c_level \
  --admin \
  --expires-days 365
```

Send normal users only:

- __PUBLIC_BASE_URL__/install
- their bearer token

Do not send the admin key.

## Upload Documents

Department-scoped document:

```bash
second-brain ingest-text \
  --file ./marketing-plan.md \
  --title "Marketing Plan Q3" \
  --department marketing \
  --visibility department \
  --classification internal
```

Company-wide public document:

```bash
second-brain ingest-text \
  --file ./company-faq.md \
  --title "Company FAQ" \
  --department marketing \
  --visibility company \
  --classification public
```

Multipart API upload:

```bash
curl -X POST __PUBLIC_BASE_URL__/api/documents/file \
  -H "X-API-Key: PASTE_GATEWAY_API_KEY" \
  -F "file=@./marketing-plan.md" \
  -F "title=Marketing Plan Q3" \
  -F "department=marketing" \
  -F "visibility=department" \
  -F "classification=internal"
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

Treat a document as large when extracted text is over 1MB, source file is over 10MB, PDF/DOCX is over 50 pages, or the document likely creates more than 200 chunks. For large documents, extract text first, clean boilerplate, split into stable sections, then upload sections.

## Query

```bash
second-brain query "tom tat tai lieu marketing"
```

Gateway query fan-out runs in parallel across allowed workspaces. Tune concurrency with:

```text
QUERY_WORKSPACE_CONCURRENCY=4
```

Parallel fan-out helps when LightRAG/LLM capacity is available. If the shared
embedding service or remote LLM provider becomes the bottleneck, lower this
value for more stable latency.

Raw API:

```bash
curl -X POST __PUBLIC_BASE_URL__/api/query \
  -H "Authorization: Bearer USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"tom tat tai lieu marketing","mode":"mix","include_references":true}'
```

## Add a Department

MVP departments are static. To add a department, update these places:

1. Gateway:
   - `services/company-ai-gateway/app.py`
   - Add `department_new_name` to `ALLOWED_GROUPS`.
2. Knowledge API:
   - `services/knowledge-api/app.py`
   - Add the new workspace to `DEPARTMENT_WORKSPACES`.
   - Add `LIGHTRAG_DEPARTMENT_NEW_NAME_URL` to `LIGHTRAG_URLS`.
3. Docker Compose:
   - `docker-compose.yml`
   - Add a `lightrag-new-name` service using the existing LightRAG service pattern.
   - Add the env var under `knowledge-api`.
   - Add dependency from `knowledge-api` to the new LightRAG service.
4. Postgres seed:
   - `postgres/init/001_schema.sql`
   - Add row to `rag_workspaces`.
   - For an existing database, also run an `INSERT ... ON CONFLICT DO NOTHING`.
5. CLI:
   - `tools/second-brain`
   - `skills/productivity/company-second-brain/scripts/second-brain`
   - Add the short department name to `ingest-text --department choices`.
6. Rebuild:

```bash
cd __DEPLOY_ROOT__
docker compose up -d --build
```

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
  docker compose up -d --build company-ai-gateway knowledge-api knowledge-worker
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

Use a department user token:

```bash
second-brain me
second-brain workspaces
second-brain query "toi co the xem tai lieu nao?"
```

Check runtime services:

```bash
cd __DEPLOY_ROOT__
docker compose ps
docker compose logs -f knowledge-worker
swapon --show
```

## Production Gaps

- Add proper admin web login instead of CLI-only admin setup.
- Add token revoke/rotation list with token IDs stored in Postgres.
- Add PDF/DOCX parser pipeline for upload.
- Add dynamic department management instead of static Compose services.
- Add audit log for token creation, document upload, and queries.
- Add backup/restore jobs for Postgres and LightRAG volumes.
- Rotate root password and move VPS access to SSH keys.
