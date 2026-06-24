# Company Second Brain Deployment

This package deploys a multi-department second brain using:

- Company AI Gateway: public FastAPI entrypoint for auth, install pages, token creation, and upload.
- Knowledge API: internal router for LightRAG workspaces.
- LightRAG: one workspace per company/department boundary.
- Postgres: metadata and workspace registry.
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

4. Start the stack:

   ```bash
   docker compose up -d --build
   ```

5. Open:

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

MVP file upload supports UTF-8 text files. Add PDF/DOCX extraction before production document ingestion.

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
