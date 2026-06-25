# Company Second Brain Deployment

This package deploys a two-workspace company second brain:

- `company_public`: all employees and agents can query this workspace.
- `department_c_level`: only admins can query this workspace.

The simplified layout reduces normal query fan-out to one LightRAG container
and admin fan-out to two containers.

## Services

- Company AI Gateway: public FastAPI entrypoint for auth, install pages, token creation, query, and upload.
- Knowledge API: internal router for workspace policy and queued ingestion.
- Knowledge Worker: Redis queue consumer that indexes documents into LightRAG.
- LightRAG: `lightrag-company-public` and `lightrag-c-level`.
- Postgres: metadata, workspace registry, and ingest payloads.
- Redis: ingest queue with append-only persistence.
- Ollama embedding service.
- Optional Honcho memory integration hook via `HONCHO_API_KEY`.

## Quick Start for a New Organization

1. Copy the example environment:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env`:

   ```text
   SECOND_BRAIN_DOMAIN=second-brain.your-company.com
   PUBLIC_BASE_URL=https://second-brain.your-company.com
   ADMIN_EMAIL=admin@your-company.com
   ADMIN_SSH_TARGET=root@YOUR_SERVER_IP
   POSTGRES_PASSWORD=<strong password>
   MINIO_ROOT_PASSWORD=<strong password>
   OLLAMA_CLOUD_API_KEY=<provider key>
   GATEWAY_API_KEY=<strong random key>
   LIGHTRAG_API_KEY=<strong random key>
   GATEWAY_TOKEN_SECRET=<strong random key>
   ```

3. Build install assets:

   ```bash
   ./scripts/build-install-assets.sh
   ```

4. Configure host swap on small VPS hosts:

   ```bash
   sudo ./scripts/configure-host-swap.sh 6G
   ```

5. Start the stack:

   ```bash
   docker compose up -d --build
   ```

6. Open:

   ```text
   https://second-brain.your-company.com/install
   https://second-brain.your-company.com/admin
   ```

## Auth Model

Normal users use bearer tokens and are always routed to `company_public`.
Department groups are not used for query routing in this MVP.

Admins can use either:

- `X-API-Key: GATEWAY_API_KEY`
- bearer token with `role_admin`

Admin queries route to both `company_public` and `department_c_level`.

## Admin Flow

Configure the admin CLI:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain config \
  --base-url https://second-brain.your-company.com \
  --admin-key "GATEWAY_API_KEY"
```

Create a normal employee token:

```bash
second-brain admin-token-create \
  --email user@your-company.com \
  --name "Company User" \
  --group company_all \
  --expires-days 90
```

Create an admin bearer token:

```bash
second-brain admin-token-create \
  --email admin@your-company.com \
  --name "Admin" \
  --group role_admin \
  --admin \
  --expires-days 365
```

Send normal users only:

- the `/install` URL
- their bearer token

Do not send `GATEWAY_API_KEY` to normal users.

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

Multipart upload through the gateway:

```bash
curl -X POST https://second-brain.your-company.com/api/documents/file \
  -H "X-API-Key: GATEWAY_API_KEY" \
  -F "file=@./company-handbook.md" \
  -F "title=Company Handbook" \
  -F "target=public"
```

Uploads are queued. The API returns `status: queued` plus a `document_id`; the
`knowledge-worker` container indexes the document into the routed LightRAG
workspace in the background.

Check a single document:

```bash
second-brain document-status DOCUMENT_ID
```

Check queue depth and status counts:

```bash
second-brain queue-status
```

MVP file upload supports UTF-8 text files. Add PDF/DOCX extraction before production document ingestion.

## Query Performance

Normal users query one workspace: `company_public`.
Admins query two workspaces: `company_public` and `department_c_level`.

`QUERY_WORKSPACE_CONCURRENCY` bounds parallel admin fan-out. The default is
`4`; with two workspaces this is already enough. Lower it only if the LLM or
embedding provider becomes the bottleneck.

Old department workspace rows or volume directories can remain on upgraded
servers. The current API filters them out and Compose removes the old containers
when deployed with `docker compose up -d --build --remove-orphans`.

## Large Document Thresholds

Treat a document as "large" when any of these are true:

- extracted text is over 1MB
- source file is over 10MB
- PDF/DOCX is over 50 pages
- document likely creates more than 200 chunks
- document includes many tables, OCR text, or repeated boilerplate

For large documents, extract and clean text first, split into sections, and
upload sections with stable titles. Do not push large PDF/DOCX binaries directly
until a parser/chunker pipeline is added.

## Agent Handoff

After deployment, the gateway serves an agent-ready handoff document at:

```text
/download/admin-handoff.md
```

Give that URL to another agent with the repo branch and target organization settings.

## Production Gaps

- Add token revoke/rotation storage.
- Add audit log writes for query/upload/token creation.
- Add PDF/DOCX parser pipeline.
- Add dynamic workspace provisioning if the organization later needs separate department containers.
- Add scheduled backups for Postgres and LightRAG volumes.
- Use SSH keys and rotate bootstrap passwords after initial setup.
