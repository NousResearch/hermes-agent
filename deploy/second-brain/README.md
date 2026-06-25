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

## Agent/User Flow

Send a normal employee or agent only:

- the `/install` URL
- their bearer token

Copy-paste this prompt to any agent or IDE with a macOS/Linux/WSL terminal:

```text
Bạn là agent được cấp quyền truy cập Company Second Brain.

Gateway: https://second-brain.your-company.com
Token: USER_TOKEN

Hãy cài skill/CLI và kết nối:
curl -fsSL https://second-brain.your-company.com/install.sh -o install-company-second-brain.sh
bash install-company-second-brain.sh

Khi installer hỏi token, dán token ở trên.
Token có thể ở dạng raw JWT, "Bearer ...", hoặc "Authorization: Bearer ...".

Sau đó chạy:
second-brain me
second-brain workspaces
second-brain query "toi co the xem tai lieu nao?"

Khi cần hỏi tài liệu công ty, dùng:
second-brain query "CAU_HOI"

Quyền user thường chỉ là workspace company_public. Không hỏi admin key.
```

User setup:

```bash
curl -fsSL https://second-brain.your-company.com/install.sh \
  -o install-company-second-brain.sh
bash install-company-second-brain.sh
```

The installer asks for the bearer token, installs the CLI plus two skills, verifies
`/api/me`, and writes local config to `~/.second-brain/config.json`.

Installed skills:

- `company-second-brain-start`: short session bootstrap for fresh agents
- `company-second-brain`: full operations guide for query, upload, sources, admin, analytics

User verification:

```bash
second-brain me
second-brain workspaces
second-brain query "What documents can I access?"
```

Expected normal user workspace:

```text
company_public
```

Generic agent/IDE API settings:

```text
Base URL: https://second-brain.your-company.com
Auth header: Authorization: Bearer USER_TOKEN
Query endpoint: POST /api/query
Workspaces endpoint: GET /api/workspaces
```

Do not send `GATEWAY_API_KEY` to normal users.

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

Copy-paste this prompt to an admin agent:

```text
Bạn là admin-agent được cấp quyền vận hành Company Second Brain.

Gateway: https://second-brain.your-company.com
Token: ADMIN_TOKEN

Hãy cài skill/CLI và kết nối:
curl -fsSL https://second-brain.your-company.com/install.sh -o install-company-second-brain.sh
bash install-company-second-brain.sh

Khi installer hỏi token, dán admin token ở trên.
Token có thể ở dạng raw JWT, "Bearer ...", hoặc "Authorization: Bearer ...".

Sau đó chạy:
second-brain me
second-brain workspaces
second-brain query "kiem tra quyen truy cap admin"

Admin token query được company_public và department_c_level.
Chỉ dùng GATEWAY_API_KEY trên máy admin tin cậy khi cần tạo token mới hoặc cấu hình hệ thống.
```

Send normal users only:

- the `/install` URL
- their bearer token

Do not send `GATEWAY_API_KEY` to normal users.

Admin verification:

```bash
second-brain queue-status
second-brain sources-list
second-brain query "admin workspace smoke test"
```

Expected admin workspaces:

```text
company_public
department_c_level
```

## Analytics

Every query is logged with actor email, role, groups, query text, allowed
workspaces, workspace latency/error, and LightRAG references returned in the
answer. Admins can inspect usage through the CLI or API:

```bash
second-brain analytics --days 30 --limit 20

curl "https://second-brain.your-company.com/api/analytics?days=30&limit=20" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

The analytics response includes:

- `summary`: total queries, unique users, success rate, average latency, document hits
- `top_documents`: documents/references surfaced most often
- `top_users`: users with the most queries
- `recent_queries`: who asked what, status, latency, referenced document count
- `workspace_usage`: query/error/latency counts per LightRAG workspace
- `top_questions`: repeated question text

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

Duplicate protection is per LightRAG workspace. The Knowledge API stores a
SHA-256 checksum for normalized document text in `document_dedupe_keys`
(`workspace_slug`, `checksum`). If the same text is uploaded again to the same
workspace, the API returns `status: duplicate` with the existing `document_id`
and does not push a Redis job, so LightRAG is not indexed twice. Source scans
also map the new source item to the existing document instead of queueing a new
document.
On service startup, older indexed documents are backfilled from remaining ingest
payloads or LightRAG full-doc storage when the workspace title mapping is
unambiguous.

Check a single document:

```bash
second-brain document-status DOCUMENT_ID
```

Check queue depth and status counts:

```bash
second-brain queue-status
```

MVP file upload supports UTF-8 text files. Add PDF/DOCX extraction before production document ingestion.

## Automatic Sources: Notion and Public Drive

Admins can register document sources and let `knowledge-worker` scan them on a
schedule. Admins can also trigger a manual scan at any time.

Notion page/data source/search:

```bash
second-brain source-create \
  --type notion \
  --name "Company Notion" \
  --notion-api-key "PASTE_NOTION_API_KEY" \
  --notion-page-url "https://www.notion.so/..." \
  --target public \
  --interval-minutes 360
```

Public Google Docs/Sheets/Slides or public file link:

```bash
second-brain source-create \
  --type drive_public \
  --name "Public Drive Doc" \
  --drive-url "https://docs.google.com/document/d/.../edit" \
  --target public \
  --interval-minutes 720
```

Operate sources:

```bash
second-brain sources-list
second-brain source-scan SOURCE_ID
second-brain source-runs SOURCE_ID
second-brain source-update SOURCE_ID --interval-minutes 1440 --reset-schedule
```

Drive public MVP supports direct public Docs/Sheets/Slides or file links. Public
folder listing needs Drive API/OAuth and is intentionally not scraped from HTML.

Source dedupe has two layers:

- unchanged source item: same `source_id`, `external_id`, and checksum only
  refreshes `last_seen_at`
- duplicate content from another source: same workspace checksum is linked to
  the existing document and skipped

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
- Add audit log writes for upload/token creation and token revocation.
- Add PDF/DOCX parser pipeline.
- Add dynamic workspace provisioning if the organization later needs separate department containers.
- Add scheduled backups for Postgres and LightRAG volumes.
- Use SSH keys and rotate bootstrap passwords after initial setup.
