# Company Second Brain With LightRAG And Honcho

## Goal

Build an internal company second brain where employees and agents can search,
summarize, and act on company knowledge without leaking documents across
departments.

The system uses:

- Hermes Agent as the agent gateway and IDE/MCP-facing agent runtime.
- LightRAG as the document and knowledge-graph retrieval engine.
- Honcho as long-term memory for users, agents, groups, projects, and sessions.
- A policy gateway as the only authorization enforcement point for retrieval,
  memory scope, and tool execution.

## Non-Negotiable Security Rule

Authorization must be enforced before context reaches the LLM.

Prompts, model instructions, UI filters, and agent self-restraint are not access
control. Every document chunk, LightRAG result, Honcho memory read, and tool call
must pass server-side policy checks.

## Target Architecture

```text
Employee / IDE / Chat / Agent
          |
          v
SSO / API Key / Service Token
          |
          v
Company AI Gateway
  - identity resolution
  - RBAC / ABAC / ACL policy checks
  - query routing
  - audit logging
  - prompt/context assembly
          |
          +-------------------------+
          |                         |
          v                         v
Hermes Agent Gateway          Company Knowledge API
  - agent sessions              - ingestion API
  - MCP tools                   - retrieval API
  - Honcho integration          - document ACL API
  - IDE integration             - citation API
          |                         |
          v                         v
Honcho Memory              LightRAG Workspaces
  - user memory              - public company graph
  - agent memory             - department graphs
  - team/project memory      - project graphs
          |                         |
          +-----------+-------------+
                      v
        Postgres / pgvector / Object Storage / Logs
```

## VPS Deployment Shape

For the first production pilot, run a single VPS with Docker Compose and strict
network boundaries.

Recommended minimum:

- 4 vCPU, 16 GB RAM, 160 GB SSD for fewer than 50 users and mostly text docs.
- 8 vCPU, 32 GB RAM, 300+ GB NVMe for heavier PDFs, OCR, and more departments.
- Ubuntu 24.04 LTS.
- Docker Engine and Docker Compose plugin.
- UFW firewall with only 22, 80, and 443 public.
- Daily encrypted backups.

Services:

```text
nginx / caddy
  public: 443
  routes:
    /app        -> web UI
    /api        -> company-ai-gateway
    /mcp        -> mcp gateway
    /rag        -> knowledge-api, internal auth required

company-ai-gateway
  private: 8000
  owns authz, policy, context assembly, audit logs

hermes-gateway
  private: localhost / docker network
  exposes agent runtime and messaging gateway

knowledge-api
  private: 8010
  wraps LightRAG and document ACL checks

lightrag
  private only
  separate workspace/index per visibility boundary

honcho
  option A: managed Honcho Cloud
  option B: self-hosted Honcho API

postgres
  private only
  stores users, groups, ACLs, audit, metadata, optional vectors

redis
  private only
  queues ingestion, OCR, background jobs, rate limits

object storage
  local MinIO or S3-compatible bucket
  stores original documents and parsed artifacts
```

## Workspace And Index Strategy

Use separated retrieval surfaces by default.

```text
company_public
department_hr
department_finance
department_sales
department_engineering
project_<project_id>
executive_private
```

Each document can be inserted into one or more authorized workspaces, but the
gateway must decide which workspaces are queryable for each request.

For highly sensitive departments such as HR and Finance, prefer physically
separate LightRAG storage namespaces instead of relying only on metadata filters.

### LightRAG Structure

Use one logical LightRAG namespace per security boundary. A user query can fan
out to multiple namespaces, but only after the policy gateway calculates which
namespaces the user may access.

```text
lightrag/
  workspaces/
    company_public/
      docs/
      kv_store/
      vector_store/
      graph_store/
    department_hr/
      docs/
      kv_store/
      vector_store/
      graph_store/
    department_finance/
      docs/
      kv_store/
      vector_store/
      graph_store/
    department_engineering/
      docs/
      kv_store/
      vector_store/
      graph_store/
    project_payroll_migration/
      docs/
      kv_store/
      vector_store/
      graph_store/
```

Recommended workspace naming:

```text
company_public
company_internal
department_<department_slug>
project_<project_slug>
restricted_<case_slug>
```

Every indexed chunk must carry metadata even when stored inside an isolated
workspace:

```json
{
  "org_id": "acme",
  "document_id": "doc_01J...",
  "document_version": 3,
  "source_uri": "s3://second-brain/raw/doc_01J...",
  "title": "Q3 Hiring Plan",
  "department_id": "hr",
  "workspace_ids": ["department_hr"],
  "classification": "confidential",
  "allowed_group_ids": ["group_hr"],
  "owner_user_id": "user_123",
  "created_at": "2026-06-24T00:00:00Z"
}
```

The metadata is not the only security layer. It is used for auditing,
post-retrieval filtering, citations, and re-indexing.

### Upload And Department Routing

Documents are not inserted directly into LightRAG. They go through an upload
pipeline with policy checks and optional approval.

```text
1. User uploads a document.
2. Gateway authenticates the uploader.
3. Upload form requires department, visibility, and optional project.
4. Backend stores the original file in object storage.
5. Backend creates a document row with status = pending_classification.
6. Classifier proposes department/classification from content and source path.
7. Policy checks whether uploader can publish to that department/project.
8. If sensitive or ambiguous, department owner approval is required.
9. Ingestion worker parses and chunks the document.
10. Worker indexes chunks into the selected LightRAG workspace(s).
11. Document status becomes indexed.
12. Audit log records uploader, approver, workspaces, and ACL decision.
```

Upload fields:

```text
required:
  title
  file
  department
  visibility: company | department | project | restricted
  classification: public | internal | confidential | restricted

optional:
  project
  allowed_users
  allowed_groups
  expires_at
  source_url
  tags
```

Routing rules:

```text
visibility = company
  -> company_public or company_internal
  allowed if uploader has company_publish permission

visibility = department
  -> department_<department_slug>
  allowed if uploader is in department or has department_admin role

visibility = project
  -> project_<project_slug>
  allowed if uploader is project member or project_admin

visibility = restricted
  -> restricted_<case_slug>
  requires explicit ACL and approver
```

Multi-department documents should not be copied everywhere by default. Prefer a
project workspace or explicit document ACL. Copy to multiple department
workspaces only when the document is intentionally shared and approved by the
owner.

Example routing:

```text
Employee handbook
  department: hr
  visibility: company
  classification: internal
  workspace: company_internal

Payroll policy
  department: hr
  visibility: department
  classification: confidential
  workspace: department_hr

Budget forecast
  department: finance
  visibility: department
  classification: confidential
  workspace: department_finance

Payroll migration technical spec
  department: engineering
  project: payroll_migration
  visibility: project
  classification: confidential
  workspace: project_payroll_migration
```

### Query Routing

For each query, the gateway calculates the allowed workspace set:

```text
active employee:
  company_public
  company_internal

HR member:
  company_public
  company_internal
  department_hr

Engineering member on payroll migration:
  company_public
  company_internal
  department_engineering
  project_payroll_migration
```

The knowledge API sends the query only to these workspaces. Results are merged,
reranked, then checked against `document_acl` again before being sent to the LLM.

## Authorization Model

Use a combination of RBAC and ABAC.

Core tables:

```text
users
  id
  email
  display_name
  status

groups
  id
  name
  type: company | department | project | role

user_groups
  user_id
  group_id

documents
  id
  title
  source_uri
  owner_user_id
  department_id
  classification: public | internal | confidential | restricted
  checksum
  version
  created_at

document_acl
  document_id
  principal_type: user | group | role
  principal_id
  permission: read | write | admin

rag_workspaces
  id
  name
  visibility_boundary

document_workspace_membership
  document_id
  workspace_id

audit_events
  id
  actor_user_id
  actor_agent_id
  action
  resource_type
  resource_id
  decision: allow | deny
  reason
  request_id
  created_at
```

Policy examples:

```text
Company public:
  allow read if user.status = active

Department private:
  allow read if user belongs to matching department group

Project private:
  allow read if user belongs to project group

Restricted:
  allow read only through explicit document_acl

Agent service account:
  allow only delegated access from an authenticated user session
```

## Retrieval Flow

```text
1. User asks a question.
2. Gateway resolves identity from SSO, API key, or IDE token.
3. Gateway calculates allowed workspace IDs and document ACL filters.
4. Gateway sends query only to allowed LightRAG workspaces.
5. Knowledge API receives LightRAG candidates.
6. Knowledge API re-checks document ACLs on every candidate.
7. Gateway assembles LLM context with only allowed snippets.
8. LLM answers with citations.
9. Audit log stores query metadata, retrieved document IDs, and policy decisions.
```

The LLM must never receive unauthorized chunks and then be asked to ignore them.

## Honcho Memory Scoping

Honcho memory should be scoped by organization, peer, department, project, and
session.

Recommended peer naming:

```text
user:<org_id>:<user_id>
agent:<org_id>:<agent_id>
group:<org_id>:department:<department_id>
project:<org_id>:<project_id>
```

Memory visibility:

```text
private_user
private_agent
project_shared
department_shared
company_shared
```

Rules:

- A user can always read their own private memory.
- An agent can read user memory only while acting for that user.
- Department memory is readable only by department members.
- Project memory is readable only by project members.
- Company shared memory must never include confidential document excerpts.
- Honcho conclusions derived from restricted documents inherit the same
  restriction label.

For the first deployment, use Honcho managed service if data policy allows it.
Use self-hosted Honcho if company policy requires all memory to stay on the VPS.

## Agent And IDE Access

Expose company knowledge through MCP tools instead of direct database access.

MCP tools:

```text
company_search(query, scope?, max_results?)
company_read_document(document_id)
company_summarize_document(document_id)
company_write_memory(scope, content, source?)
company_recall_memory(scope, query)
company_create_task(project_id, title, body)
```

Every tool call receives:

```text
actor_user_id
actor_agent_id
session_id
allowed_groups
request_id
```

Cursor, Codex, Claude Code, OpenCode, and Hermes can all use this MCP gateway.
They must not receive raw Postgres, LightRAG, object storage, or Honcho admin
credentials.

## Ingestion Pipeline

```text
1. Upload or sync source document.
2. Extract metadata and classify sensitivity.
3. Parse document into text, tables, and images where supported.
4. Create document row and ACL records.
5. Chunk and index into the selected LightRAG workspace.
6. Store original file and parsed artifacts.
7. Emit audit event.
```

Initial source connectors:

- Manual upload.
- Git repository documentation.
- Google Drive or shared folders, if company SSO is available.
- Notion/Confluence later if needed.

## Deployment Phases

### Phase 1: Secure MVP

- One VPS.
- Hermes gateway running in Docker.
- Company AI Gateway and Knowledge API.
- LightRAG with separate workspaces for public, one department, and one project.
- Honcho configured for user and agent memory.
- Local admin bootstrap users.
- Upload and query documents through API.
- Audit logs in Postgres.

### Phase 2: Department Rollout

- Add SSO.
- Add department group sync.
- Add more LightRAG workspaces.
- Add MCP endpoint for IDE access.
- Add ingestion jobs and scheduled re-indexing.
- Add admin UI for ACL and document status.

### Phase 3: Production Hardening

- Move object storage to S3-compatible bucket or hardened MinIO.
- Add encrypted offsite backups.
- Add monitoring and alerting.
- Add per-department retention policy.
- Add red-team tests for cross-department leakage.
- Split databases or hosts if HR/Finance isolation requirements demand it.

## Compose Blueprint

This is the intended service layout, not a final production file:

```yaml
services:
  reverse-proxy:
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - company-ai-gateway

  company-ai-gateway:
    build: ./services/company-ai-gateway
    env_file: .env
    depends_on:
      - postgres
      - redis
      - knowledge-api

  knowledge-api:
    build: ./services/knowledge-api
    env_file: .env
    depends_on:
      - postgres
      - redis
      - lightrag
      - minio

  hermes-gateway:
    build: .
    command: ["gateway", "run"]
    env_file: .env
    volumes:
      - hermes-data:/opt/data

  lightrag:
    image: company/lightrag:latest
    env_file: .env
    volumes:
      - lightrag-data:/data

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: second_brain
      POSTGRES_USER: second_brain
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis-data:/data

  minio:
    image: minio/minio
    command: ["server", "/data", "--console-address", ":9001"]
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio-data:/data

volumes:
  hermes-data:
  lightrag-data:
  postgres-data:
  redis-data:
  minio-data:
```

## Operational Requirements

Backups:

- Postgres: daily `pg_dump` plus WAL or volume snapshot.
- Object storage: daily encrypted sync.
- LightRAG data: backup with document metadata version.
- Hermes/Honcho config: encrypted backup, secrets excluded from logs.

Observability:

- Request IDs across gateway, knowledge API, LightRAG, and LLM calls.
- Audit every allow/deny decision.
- Track retrieval count, denied candidates, answer citations, and token cost.
- Alert on cross-department denied retrieval attempts.

Secrets:

- No secrets in git.
- Use `.env` on VPS with file permissions `600`.
- Prefer API keys scoped per service.
- Rotate LLM, Honcho, SSO, and storage credentials quarterly.

## Deployment Information Needed

To deploy on the VPS, collect:

```text
VPS:
  host/ip
  ssh port
  sudo username
  ssh public-key access
  OS version
  CPU/RAM/disk

Domain:
  domain or subdomain
  DNS provider
  whether Cloudflare is used

Auth:
  SSO provider, if any
  initial admin emails
  department/group list

AI:
  LLM provider and API key
  embedding provider/model
  whether models must be self-hosted

Memory:
  Honcho Cloud or self-hosted Honcho
  retention requirements

Documents:
  first departments
  initial source folders/repos
  classification levels
  who can upload and approve documents
```

Do not paste private keys or long-lived root passwords into chat. Use an SSH key
for a sudo-capable deploy user, then store runtime secrets in the VPS `.env`.

## First Implementation Milestone

Build the following before indexing sensitive company documents:

1. `company-ai-gateway` with authenticated `/query`.
2. `knowledge-api` with `/ingest`, `/search`, and `/documents/{id}`.
3. Postgres schema for users, groups, documents, ACLs, workspaces, and audits.
4. LightRAG workspace routing.
5. Honcho memory scope wrapper.
6. MCP server exposing only policy-checked tools.
7. Leakage tests proving a Finance user cannot retrieve HR snippets, and the
   reverse.

## Current VPS Installation Spec

Production pilot host:

```text
Domain: __PUBLIC_BASE_URL__
VPS: YOUR_SERVER_IP
OS: Ubuntu 24.04 LTS
Runtime: Docker Compose behind existing Coolify Traefik
Deploy path: /opt/second-brain
Public service: company-ai-gateway only
Internal services: knowledge-api, LightRAG, Ollama embedding, Postgres, Redis, MinIO
```

Running containers:

```text
second-brain-gateway
second-brain-knowledge-api
second-brain-lightrag-company-public
second-brain-lightrag-company-internal
second-brain-lightrag-marketing
second-brain-lightrag-financial
second-brain-lightrag-hr
second-brain-lightrag-engineering
second-brain-lightrag-c-level
second-brain-ollama-embed
second-brain-postgres
second-brain-redis
second-brain-minio
```

Public endpoints:

```text
GET  /health
GET  /docs
GET  /api/me
GET  /api/workspaces
POST /api/admin/tokens
POST /api/documents/text
POST /api/query
```

Admin-only routes accept:

```text
Header: X-API-Key: <GATEWAY_API_KEY from /opt/second-brain/.env>
```

Member/agent/IDE routes accept:

```text
Header: Authorization: Bearer <user token generated by admin>
```

Do not expose LightRAG containers directly to the internet. Team members and
agents must call the gateway only.

## Admin-Generated Tokens For Members, Agents, And IDEs

The admin should not share `GATEWAY_API_KEY` with normal users. Instead, admin
generates a signed bearer token for each user, agent, or IDE profile.

### Generate A User Token

```bash
curl -X POST __PUBLIC_BASE_URL__/api/admin/tokens \
  -H "X-API-Key: <GATEWAY_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@company.com",
    "name": "Alice",
    "groups": ["department_marketing"],
    "expires_days": 30
  }'
```

Response:

```json
{
  "token_type": "Bearer",
  "token": "...",
  "expires_at": 1790000000,
  "user": {
    "email": "alice@company.com",
    "name": "Alice",
    "groups": ["department_marketing"],
    "role": "member"
  },
  "setup": {
    "base_url": "__PUBLIC_BASE_URL__",
    "headers": {
      "Authorization": "Bearer ..."
    },
    "query_endpoint": "__PUBLIC_BASE_URL__/api/query",
    "workspaces_endpoint": "__PUBLIC_BASE_URL__/api/workspaces"
  }
}
```

Admin can generate admin-capable tokens only for trusted operators:

```json
{
  "email": "ops@company.com",
  "name": "Ops Admin",
  "groups": ["role_admin", "department_engineering"],
  "expires_days": 7
}
```

### User IDE/Agent Configuration

Minimum config every IDE or agent needs:

```text
Base URL: __PUBLIC_BASE_URL__
Auth header: Authorization: Bearer <token>
Query endpoint: POST /api/query
Workspace endpoint: GET /api/workspaces
```

Generic JSON config:

```json
{
  "companySecondBrain": {
    "baseUrl": "__PUBLIC_BASE_URL__",
    "headers": {
      "Authorization": "Bearer USER_TOKEN"
    },
    "queryEndpoint": "/api/query",
    "workspacesEndpoint": "/api/workspaces"
  }
}
```

Test the token:

```bash
curl __PUBLIC_BASE_URL__/api/me \
  -H "Authorization: Bearer USER_TOKEN"
```

Query from an IDE/agent:

```bash
curl -X POST __PUBLIC_BASE_URL__/api/query \
  -H "Authorization: Bearer USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the documents I can access.",
    "mode": "mix",
    "include_references": true
  }'
```

When a bearer token is used, the gateway ignores user-supplied `groups` and uses
the groups embedded in the signed token. This prevents a user from adding another
department manually in JSON.

### CLI For Codex, IDEs, And Admins

A small dependency-free CLI is available:

```text
Local repo:
  deploy/second-brain/tools/second-brain

VPS:
  /opt/second-brain/tools/second-brain
```

Admin local setup:

```bash
deploy/second-brain/tools/second-brain config \
  --base-url __PUBLIC_BASE_URL__ \
  --admin-key "<GATEWAY_API_KEY>"
```

Admin generate a Codex/IDE token:

```bash
deploy/second-brain/tools/second-brain admin-token-create \
  --email codex-engineering@company.com \
  --name "Codex Engineering" \
  --group department_engineering \
  --expires-days 30
```

Generate and save the token on the current machine:

```bash
deploy/second-brain/tools/second-brain admin-token-create \
  --email codex-engineering@company.com \
  --name "Codex Engineering" \
  --group department_engineering \
  --expires-days 30 \
  --save-user-token
```

User setup with a token generated by admin:

```bash
deploy/second-brain/tools/second-brain config \
  --base-url __PUBLIC_BASE_URL__ \
  --token "USER_TOKEN"
```

Verify identity:

```bash
deploy/second-brain/tools/second-brain me
```

List accessible workspaces:

```bash
deploy/second-brain/tools/second-brain workspaces
```

Query from Codex/terminal:

```bash
deploy/second-brain/tools/second-brain query \
  "What documents can I access?"
```

Admin ingest text:

```bash
deploy/second-brain/tools/second-brain ingest-text \
  --title "Engineering Runbook" \
  --department engineering \
  --visibility department \
  --classification confidential \
  --file ./runbook.md
```

Generate an `AGENTS.md` snippet for Codex:

```bash
deploy/second-brain/tools/second-brain codex-snippet
```

The CLI stores local config at:

```text
~/.second-brain/config.json
```

### Token Scope Design

```text
Marketing token
  groups: ["department_marketing"]

Financial token
  groups: ["department_financial"]

HR token
  groups: ["department_hr"]

Engineering token
  groups: ["department_engineering"]

C Level token
  groups: ["department_c_level"]

Company-only token
  groups: []

Admin token
  groups: ["role_admin"]
```

Token rotation:

```text
Short-lived contractor token: 1-7 days
Normal employee token: 30-90 days
IDE token on personal laptop: 30 days
Service/agent token: 7-30 days
Admin token: 1-7 days
```

## LightRAG Team Configuration

### Workspace Map

```text
company_public
  Intended for public or externally shareable company knowledge.
  Queryable by all active users.

company_internal
  Intended for internal company-wide docs.
  Queryable by all active users.

department_marketing
  Marketing-only documents, campaigns, brand, content, market research.

department_financial
  Finance-only documents, budgets, invoices, forecasts, revenue plans.

department_hr
  HR-only documents, policies, payroll, hiring, people operations.

department_engineering
  Engineering-only documents, architecture, runbooks, incidents, specs.

department_c_level
  Executive-only documents, strategy, board notes, leadership decisions.
```

### LightRAG Runtime Parameters

Each LightRAG workspace currently uses the same runtime settings:

```text
Image: ghcr.io/hkuds/lightrag:latest
Port: 9621 internal only
KV storage: JsonKVStorage
Vector storage: NanoVectorDBStorage
Graph storage: NetworkXStorage
Document status storage: JsonDocStatusStorage
Embedding binding: ollama
Embedding host: http://ollama-embed:11434
Embedding model: bge-m3:latest
Embedding dimension: 1024
Embedding token limit: 8192
LLM binding: openai-compatible
LLM host: https://ollama.com/v1
LLM model: deepseek-v4-flash
Keyword/query/extraction model: deepseek-v4-flash
Max async LLM: 2
Chunk size: 1200 tokens
Chunk overlap: 100 tokens
Default query mode: mix
API auth: X-API-Key via LIGHTRAG_API_KEY
```

### LightRAG Storage Paths

Inside the shared Docker volume:

```text
/data/workspaces/company_public/
/data/workspaces/company_internal/
/data/workspaces/department_marketing/
/data/workspaces/department_financial/
/data/workspaces/department_hr/
/data/workspaces/department_engineering/
/data/workspaces/department_c_level/
```

Each workspace contains:

```text
inputs/
rag_storage/
```

The gateway also creates a logical folder structure:

```text
/data/lightrag/workspaces/<workspace>/
  docs/
  kv_store/
  vector_store/
  graph_store/
```

The active LightRAG API storage is the `/data/workspaces/<workspace>/rag_storage`
tree.

### Ingest Text API

Use this endpoint for plain text ingestion:

```http
POST __PUBLIC_BASE_URL__/api/documents/text
X-API-Key: <gateway key>
Content-Type: application/json
```

Body:

```json
{
  "title": "HR Vacation Policy",
  "department": "hr",
  "visibility": "department",
  "classification": "confidential",
  "text": "Document body..."
}
```

Allowed departments:

```text
marketing
financial
hr
engineering
c_level
```

Routing:

```text
visibility=company + classification=public
  -> company_public

visibility=company + classification=internal/confidential/restricted
  -> company_internal

visibility=department + department=hr
  -> department_hr

visibility=department + department=financial
  -> department_financial
```

Current MVP supports `company` and `department` visibility. Project and
restricted-case workspaces are in the extension plan.

### Query API

Use this endpoint for RAG queries:

```http
POST __PUBLIC_BASE_URL__/api/query
X-API-Key: <gateway key>
Content-Type: application/json
```

Body:

```json
{
  "query": "What is the HR vacation policy?",
  "groups": ["department_hr"],
  "mode": "mix",
  "include_references": true
}
```

Group-to-workspace access:

```text
groups=[]
  -> company_public
  -> company_internal

groups=["department_hr"]
  -> company_public
  -> company_internal
  -> department_hr

groups=["department_financial"]
  -> company_public
  -> company_internal
  -> department_financial
```

Current MVP accepts `groups` in the request body for testing. Production should
derive groups from SSO/JWT, not from user-supplied JSON.

## Honcho Team Configuration

Honcho is configured as the long-term memory layer, not as the document
authorization system.

Current configuration values:

```text
HONCHO_API_KEY: stored in /opt/second-brain/.env
Recommended workspace: second-brain-company
Recommended host/app identity: company-second-brain
Recommended AI peer: agent:<org_id>:second-brain
```

Recommended peer naming:

```text
user:<org_id>:<user_id>
agent:<org_id>:<agent_id>
group:<org_id>:department:<department_slug>
project:<org_id>:<project_slug>
```

Recommended memory visibility scopes:

```text
private_user
private_agent
department_shared
project_shared
company_shared
```

Rules:

```text
private_user
  Readable only by the user and agents acting for that user.

private_agent
  Readable only by the specific agent/service account.

department_shared
  Readable only by members of that department.

project_shared
  Readable only by members of that project.

company_shared
  Readable by all active employees.
```

Honcho should store summaries, preferences, decisions, and working context. It
should not store raw confidential document chunks unless the memory entry
inherits the same department/project ACL.

Recommended Honcho write metadata:

```json
{
  "org_id": "company",
  "actor_user_id": "user_123",
  "actor_agent_id": "agent_second_brain",
  "scope": "department_shared",
  "department": "engineering",
  "project": null,
  "source_document_id": "doc_uuid",
  "classification": "confidential"
}
```

## Team Onboarding

### Admin Setup

Initial admin:

```text
admin@example.com
```

Admin responsibilities:

```text
1. Create or sync users.
2. Assign users to department groups.
3. Approve sensitive uploads.
4. Review audit logs.
5. Rotate API keys and SSH access.
```

### Department Group Names

Use these exact group slugs:

```text
department_marketing
department_financial
department_hr
department_engineering
department_c_level
role_admin
company_all
```

### Member Access Matrix

```text
All active members
  company_public
  company_internal

Marketing member
  company_public
  company_internal
  department_marketing

Financial member
  company_public
  company_internal
  department_financial

HR member
  company_public
  company_internal
  department_hr

Engineering member
  company_public
  company_internal
  department_engineering

C Level member
  company_public
  company_internal
  department_c_level
```

### Recommended Rollout Process

```text
1. Create a small pilot group.
2. Upload 3-5 non-sensitive docs per department.
3. Run leakage tests between departments.
4. Add real confidential docs after approval workflow is enabled.
5. Connect IDE/agent access through the gateway/MCP only.
```

## Features Available Now

```text
HTTPS public gateway on YOUR_SECOND_BRAIN_DOMAIN
API-key protected gateway endpoints
Department-separated LightRAG workspaces
Company public/internal workspaces
Text ingestion into LightRAG
Background LightRAG indexing
Graph + vector RAG retrieval
Ollama Cloud LLM
Local bge-m3 embeddings
Query fan-out only to allowed workspaces
Basic no-leakage behavior by workspace routing
Postgres metadata schema for users, groups, docs, workspaces, ACLs, audit
MinIO and Redis running for file/queue extensions
Honcho API key stored for memory integration
```

## Features To Build Next

### File Upload

Add:

```text
POST /api/documents/upload
```

Supported formats:

```text
.txt
.md
.pdf
.docx
.xlsx
.pptx
```

Pipeline:

```text
upload -> object storage -> parse -> classify -> approve -> index -> audit
```

### SSO And Real Auth

Replace request-body `groups` with verified identity:

```text
Google Workspace
Microsoft Entra ID
OIDC provider
SAML provider
```

Gateway should derive:

```text
user_id
email
groups
department
role
```

### MCP For IDE And Agents

Expose tools:

```text
company_search
company_read_document
company_ingest_text
company_upload_document
company_recall_memory
company_write_memory
```

IDE clients:

```text
Codex
Cursor
Claude Code
OpenCode
Hermes Agent
```

### Approval Workflow

Add statuses:

```text
draft
pending_classification
pending_approval
approved
indexed
rejected
archived
```

Required approvals:

```text
classification=confidential
classification=restricted
visibility=restricted
cross-department sharing
```

### Audit Dashboard

Track:

```text
who uploaded
who approved
who queried
which workspaces were searched
which document IDs were returned
allow/deny decisions
LLM/model cost
latency
```

### Project Workspaces

Add:

```text
project_<project_slug>
```

Useful for cross-functional projects where Marketing, Finance, HR, Engineering,
and C Level all need shared context without opening department-private archives.

### Restricted Workspaces

Add:

```text
restricted_<case_slug>
```

Use for board, legal, acquisition, incident, investigation, or payroll cases.

## Operations

### Health Checks

```bash
curl __PUBLIC_BASE_URL__/health
cd /opt/second-brain
docker compose ps
docker logs --tail=100 second-brain-gateway
docker logs --tail=100 second-brain-knowledge-api
docker logs --tail=100 second-brain-lightrag-hr
```

### Restart

```bash
cd /opt/second-brain
docker compose up -d
```

### Rebuild Gateway/API

```bash
cd /opt/second-brain
docker compose up -d --build knowledge-api company-ai-gateway
```

### Backup Targets

```text
Docker volumes:
  second-brain_postgres-data
  second-brain_lightrag-data
  second-brain_minio-data
  second-brain_redis-data
  second-brain_ollama-data

Config:
  /opt/second-brain/.env
  /opt/second-brain/docker-compose.yml
```

### Security Actions

```text
1. Rotate root password immediately.
2. Move SSH access to key-based deploy user.
3. Do not share GATEWAY_API_KEY with normal users.
4. Put user-facing access behind SSO before broad rollout.
5. Keep LightRAG internal-only.
6. Rotate OLLAMA_CLOUD_API_KEY and HONCHO_API_KEY if exposed.
```

## Runtime Upgrade Notes

The runtime includes three operational upgrades beyond the initial MVP:

1. Query fan-out is parallel inside `knowledge-api`.
   - `QUERY_WORKSPACE_CONCURRENCY` controls how many LightRAG workspaces are queried at once.
   - Default: `4`.
   - Parallel fan-out helps when LightRAG/LLM capacity is available. If the shared embedding service or remote LLM provider becomes the bottleneck, lower this value for more stable latency.
2. Document ingestion is asynchronous.
   - Gateway/admin upload returns `status: queued` and a `document_id`.
   - `knowledge-worker` consumes Redis queue `INGEST_QUEUE_NAME` and sends documents to the routed LightRAG workspace.
   - Document status values are `queued`, `indexing`, `indexed`, and `failed`.
   - Admin can check `GET /api/documents/{document_id}` and `GET /api/queue/status`.
3. Resource guardrails are configured in Compose.
   - LightRAG, gateway, API, worker, Postgres, Redis, MinIO, and Ollama embedding services have CPU/RAM/memswap settings.
   - Small VPS hosts should run `deploy/second-brain/scripts/configure-host-swap.sh 6G` once to create host swap.

Large-document rule of thumb:

- extracted text over 1MB
- source file over 10MB
- PDF/DOCX over 50 pages
- likely more than 200 chunks
- OCR-heavy, table-heavy, or repeated-boilerplate document

For large documents, extract and clean text first, split into stable sections,
then upload sections separately. Do not rely on one huge LightRAG insert for
production ingestion.
