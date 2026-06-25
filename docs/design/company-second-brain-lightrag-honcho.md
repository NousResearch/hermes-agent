# Company Second Brain: LightRAG + Honcho

Current status: MVP simplified to two LightRAG workspaces.

## Objective

Deploy a company second brain that any employee can query from an agent or IDE
with only a token, while keeping C-Level documents restricted to admins.

The system supports:

- one-click skill/CLI installation from the gateway landing page
- bearer-token access for employees
- admin token generation and upload workflows
- queued document ingestion into LightRAG
- optional Honcho memory configuration for agents
- handoff documentation for future agents/operators

## Current Workspace Model

LightRAG does not provide document-level ACLs. The system enforces isolation by
running separate LightRAG workspaces and controlling which workspace each query
can reach.

Current workspaces:

```text
company_public
department_c_level
```

Policy:

- Normal employees query only `company_public`.
- Admins query `company_public` and `department_c_level`.
- Old department groups such as `department_marketing`, `department_hr`, or
  `department_engineering` are ignored by the gateway for query routing.
- Upload routing uses `target=public` or `target=c_level`.

This keeps normal query fan-out at one LightRAG container and admin query
fan-out at two containers.

## Runtime Services

```text
company-ai-gateway
knowledge-api
knowledge-worker
lightrag-company-public
lightrag-c-level
ollama-embed
postgres
redis
minio
```

`lightrag-company-public` uses:

```text
WORKSPACE=company_public
INPUT_DIR=/data/workspaces/company_public/inputs
WORKING_DIR=/data/workspaces/company_public/rag_storage
```

`lightrag-c-level` uses:

```text
WORKSPACE=department_c_level
INPUT_DIR=/data/workspaces/department_c_level/inputs
WORKING_DIR=/data/workspaces/department_c_level/rag_storage
```

The old multi-department services were removed from Compose. When upgrading an
existing server, deploy with:

```bash
docker compose up -d --build --remove-orphans company-ai-gateway knowledge-api knowledge-worker lightrag-company-public lightrag-c-level
```

This stops old LightRAG containers without deleting their volume data.

## Request Flow

Employee query:

```text
Agent/IDE -> Gateway bearer auth -> groups=["company_all"]
          -> Knowledge API -> company_public -> response
```

Admin query:

```text
Agent/IDE/Admin CLI -> Gateway admin auth -> groups=["role_admin"]
                    -> Knowledge API -> company_public + department_c_level
                    -> merged response payload
```

Document upload:

```text
Admin -> Gateway upload endpoint -> Knowledge API route_workspace()
      -> Postgres document row + Redis job
      -> knowledge-worker -> target LightRAG /documents/text
```

## Auth

Admin:

```text
X-API-Key: GATEWAY_API_KEY
```

or bearer token generated with:

```text
group=role_admin
role=admin
```

Employee:

```text
Authorization: Bearer USER_TOKEN
```

Employee tokens should normally include:

```text
group=company_all
role=member
```

The gateway ignores user-supplied `groups` in `/api/query` and derives query
groups from the authenticated role.

## APIs

Query:

```http
POST /api/query
Authorization: Bearer USER_TOKEN
Content-Type: application/json

{
  "query": "What documents can I access?",
  "mode": "mix",
  "include_references": true
}
```

Create token:

```http
POST /api/admin/tokens
X-API-Key: GATEWAY_API_KEY
Content-Type: application/json

{
  "email": "user@company.com",
  "name": "Company User",
  "groups": ["company_all"],
  "expires_days": 90
}
```

Upload text:

```http
POST /api/documents/text
X-API-Key: GATEWAY_API_KEY
Content-Type: application/json

{
  "title": "Company Handbook",
  "target": "public",
  "classification": "internal",
  "text": "..."
}
```

C-Level upload:

```json
{
  "title": "Board Plan",
  "target": "c_level",
  "classification": "restricted",
  "text": "..."
}
```

## CLI Examples

Employee setup:

```bash
second-brain connect --base-url https://second-brain.example.com --token "USER_TOKEN"
second-brain me
second-brain workspaces
second-brain query "tom tat tai lieu cong ty"
```

Admin setup:

```bash
second-brain config \
  --base-url https://second-brain.example.com \
  --admin-key "GATEWAY_API_KEY"
```

Create employee token:

```bash
second-brain admin-token-create \
  --email user@company.com \
  --name "Company User" \
  --group company_all \
  --expires-days 90
```

Create admin token:

```bash
second-brain admin-token-create \
  --email admin@company.com \
  --name "Admin" \
  --group role_admin \
  --admin \
  --expires-days 365
```

Upload public document:

```bash
second-brain ingest-text \
  --file ./company-handbook.md \
  --title "Company Handbook" \
  --target public
```

Upload C-Level document:

```bash
second-brain ingest-text \
  --file ./board-plan.md \
  --title "Board Plan" \
  --target c_level \
  --classification restricted
```

## Performance

The simplified model improves speed by reducing workspace fan-out:

- employee query: 1 LightRAG request
- admin query: 2 parallel LightRAG requests

The previous multi-department model could require common + department fan-out for
normal users and many more workspaces for admins. Removing unused containers also
frees RAM/CPU on a single VPS.

`QUERY_WORKSPACE_CONCURRENCY=4` is sufficient for the current two-workspace
architecture. If LLM or embedding rate limits become the bottleneck, lower it.

## Document Size Guidance

Treat a document as large if any of these are true:

- extracted text exceeds 1MB
- source file exceeds 10MB
- PDF/DOCX exceeds 50 pages
- expected chunk count exceeds 200
- content contains many tables, OCR artifacts, or repeated boilerplate

For large documents, extract and clean text first, split into stable sections,
then upload sections separately.

## Honcho Memory

Honcho is configured for agent memory, not document ACL enforcement. Keep
organization or user-specific memory IDs outside public docs and avoid using
Honcho as the source of truth for LightRAG authorization.

Recommended agent handoff fields:

```text
SECOND_BRAIN_BASE_URL
SECOND_BRAIN_TOKEN
HONCHO_API_KEY
HONCHO_APP_ID
HONCHO_USER_ID
```

## Extension Path

Add a dedicated department LightRAG only when there is a real isolation or
performance need. Required changes:

1. Add a new LightRAG service in `deploy/second-brain/docker-compose.yml`.
2. Add the workspace URL in `knowledge-api`.
3. Add the workspace row in Postgres.
4. Add gateway policy deciding which role/group can query it.
5. Add upload routing and CLI/docs examples.
6. Deploy with `--remove-orphans` and verify container list.

Do not add a department workspace just to label documents. Use metadata and
titles for labels; use separate containers only for isolation or scale.

## Verification Checklist

- `pytest tests/second_brain/test_workspace_policy.py` passes.
- `python -m py_compile` passes for gateway, knowledge API, worker, and CLI scripts.
- `docker compose config` succeeds on the VPS.
- `docker ps` shows only `second-brain-lightrag-company-public` and `second-brain-lightrag-c-level`.
- Employee token `/api/workspaces` returns only `company_public`.
- Employee query returns `allowed_workspaces=["company_public"]`.
- Admin query returns `allowed_workspaces=["company_public","department_c_level"]`.
- Public upload can be queried by employee token.
- C-Level upload cannot be queried by employee token but can be queried by admin.
