# Company Second Brain Deployment

This package deploys a multi-department second brain using:

- Company AI Gateway: public FastAPI entrypoint for auth, install pages, token creation, and upload.
- Knowledge API: internal router for LightRAG workspaces.
- Knowledge Worker: background Redis queue consumer for document ingestion.
- LightRAG: one workspace per company/department boundary.
- Postgres: metadata, workspace registry, and ingest job payloads.
- Redis: durable-ish ingest queue with append-only persistence.
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

4. Configure host swap on small VPS hosts. A 4-8GB swap file is recommended for
   4 vCPU / 16GB RAM deployments running multiple LightRAG containers.

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

## Admin Flow

Configure the admin CLI:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain config \
  --base-url https://second-brain.your-company.com \
  --admin-key "GATEWAY_API_KEY"
```

Create a department user token:

```bash
second-brain admin-token-create \
  --email user@your-company.com \
  --name "Marketing User" \
  --group department_marketing \
  --expires-days 90
```

Send the user:

- the `/install` URL
- their bearer token

Do not send `GATEWAY_API_KEY` to normal users.

## Upload Documents

Plain text upload through the CLI:

```bash
second-brain ingest-text \
  --file ./document.md \
  --title "Document Title" \
  --department marketing \
  --visibility department \
  --classification internal
```

Multipart upload through the gateway:

```bash
curl -X POST https://second-brain.your-company.com/api/documents/file \
  -H "X-API-Key: GATEWAY_API_KEY" \
  -F "file=@./document.md" \
  -F "title=Document Title" \
  -F "department=marketing" \
  -F "visibility=department" \
  -F "classification=internal"
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

Gateway query fan-out is parallelized across allowed LightRAG workspaces and
bounded by `QUERY_WORKSPACE_CONCURRENCY`. The default is `4`. Tune this value:
parallel fan-out helps when LightRAG/LLM capacity is available, but a single
shared embedding service or remote LLM rate limit can make lower concurrency
more stable.

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

Give that URL to another agent with the repo branch and target organization settings. The handoff document includes service layout, auth model, add-department steps, upgrade flow, and verification commands.

## Production Gaps

- Add token revoke/rotation storage.
- Add audit log writes for query/upload/token creation.
- Add PDF/DOCX parser pipeline.
- Convert static department workspaces to dynamic workspace provisioning.
- Add scheduled backups for Postgres and LightRAG volumes.
- Use SSH keys and rotate bootstrap passwords after initial setup.
