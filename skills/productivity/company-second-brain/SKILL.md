---
name: company-second-brain
description: Use when querying, summarizing, or ingesting company knowledge through a Company Second Brain RAG gateway from Hermes, Codex, IDEs, or terminal workflows.
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [company-knowledge, rag, lightrag, honcho, second-brain, ide]
    category: productivity
---

# Company Second Brain

## Overview

Use the company second-brain gateway for company knowledge lookup. The current
MVP has two LightRAG workspaces:

- `company_public`: every employee token can query this.
- `department_c_level`: only admin tokens or the admin API key can query this.

The configured gateway is:

```text
__PUBLIC_BASE_URL__
```

The CLI wrapper inside this skill is:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain
```

For fresh sessions that only need quick query commands, use
`company-second-brain-start` first. Use this full skill for upload, sources,
token creation, analytics, and admin workflows.

## When To Use

Use this skill when the user asks to:

- Search company knowledge, second brain, internal docs, or C-Level docs.
- Generate setup instructions for Codex, Hermes, Cursor, Claude Code, or another IDE agent.
- Create a user/agent token as an admin.
- Ingest a plain-text document into the company RAG.
- Configure or manually scan Notion and public Google Drive document sources.
- Inspect admin analytics: top documents, user queries, workspace latency/errors,
  and repeated questions.

Do not use this skill for local codebase search. Use normal repository tools for
code files unless the user explicitly asks for company second-brain knowledge.

## Authentication Model

Admin authentication:

```text
X-API-Key: GATEWAY_API_KEY
```

User/agent/IDE authentication:

```text
Authorization: Bearer USER_TOKEN
```

Never reveal or share `GATEWAY_API_KEY` with normal users. Generate a user token
instead.

Valid token groups for this MVP:

```text
company_all
role_admin
```

Old department groups are ignored by the gateway for query routing.

## Agent User Runbook

Use this flow for a normal employee, agent, or IDE user. They only need the
gateway URL and their bearer token. Never ask them for the admin key.

If an admin pasted a setup prompt, extract:

```text
Gateway: __PUBLIC_BASE_URL__
Token: USER_TOKEN
```

The token may be raw JWT, `Bearer ...`, or `Authorization: Bearer ...`; the CLI
normalizes it.

Install/connect:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain connect \
  --base-url __PUBLIC_BASE_URL__ \
  --token "USER_TOKEN"
```

Verify:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain me
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain workspaces
```

Expected normal user workspace:

```text
company_public
```

Ask a question:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain query "QUESTION"
```

The CLI stores config at:

```text
~/.second-brain/config.json
```

For a non-Hermes agent or IDE, use the same auth contract:

```text
Base URL: __PUBLIC_BASE_URL__
Header: Authorization: Bearer USER_TOKEN
Endpoint: POST /api/query
```

## Query

Use:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain query "QUESTION"
```

For raw JSON:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain query "QUESTION" --raw
```

The gateway derives access from the bearer token. Do not ask the user to provide
`groups` manually.

## Admin Runbook

Use this flow for operators. Admin commands require `GATEWAY_API_KEY` or an
admin bearer token. Do not send `GATEWAY_API_KEY` to employees.

Admin-agent copy-paste setup should provide:

```text
Gateway: __PUBLIC_BASE_URL__
Token: ADMIN_TOKEN
```

Use the admin bearer token for C-Level query access. Use `GATEWAY_API_KEY` only
on trusted admin machines when creating tokens or changing system configuration.

Configure admin key:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain config \
  --base-url __PUBLIC_BASE_URL__ \
  --admin-key "GATEWAY_API_KEY"
```

Create a normal employee token:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain admin-token-create \
  --email user@company.local \
  --name "Company User" \
  --group company_all \
  --expires-days 30
```

Create an admin bearer token for agent/IDE admin work:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain admin-token-create \
  --email admin@company.local \
  --name "Admin" \
  --group role_admin \
  --admin \
  --expires-days 365
```

Check system status:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain queue-status
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain sources-list
```

Inspect analytics:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain analytics --days 30 --limit 20
```

Analytics includes:

- top documents/references surfaced by LightRAG
- users and recent query text
- repeated questions
- workspace latency/error counts

## Ingest Text

Admin only. Ingest is asynchronous: the API returns `queued`, then
`knowledge-worker` indexes the document into LightRAG in the background.
If the same normalized text already exists in the same workspace, the API
returns `duplicate` with the existing `document_id`; do not retry that upload.

Public document:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain ingest-text \
  --title "Company Handbook" \
  --target public \
  --file ./company-handbook.md
```

C-Level document:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain ingest-text \
  --title "Board Plan" \
  --target c_level \
  --classification restricted \
  --file ./board-plan.md
```

Check status after upload:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain document-status DOCUMENT_ID
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain queue-status
```

Treat a document as large when extracted text is over 1MB, source file is over
10MB, PDF/DOCX is over 50 pages, or it likely creates more than 200 chunks. For
large documents, extract and clean text first, split into stable sections, then
upload sections separately.

## Scheduled Sources

Admin only. Source scans are asynchronous: a source run fetches documents from
Notion or public Drive, then queues changed documents for LightRAG indexing.

Create a Notion source:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain source-create \
  --type notion \
  --name "Company Notion" \
  --notion-api-key "PASTE_NOTION_API_KEY" \
  --notion-page-url "https://www.notion.so/..." \
  --target public \
  --interval-minutes 360
```

Create a public Drive source:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain source-create \
  --type drive_public \
  --name "Public Drive Doc" \
  --drive-url "https://docs.google.com/document/d/.../edit" \
  --target public \
  --interval-minutes 720
```

Operate sources:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain sources-list
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain source-scan SOURCE_ID
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain source-runs SOURCE_ID
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain source-update SOURCE_ID --interval-minutes 1440 --reset-schedule
```

Drive public MVP supports direct public Docs/Sheets/Slides or file links. Public
folder listing needs Drive API/OAuth.

## Codex/Hermes Snippet

Generate instructions for an agent context file:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain codex-snippet
```

## Response Behavior

When answering from RAG results:

- Report which workspaces were searched.
- Prefer answers with references.
- Say clearly when no accessible workspace has context.
- Do not imply inaccessible C-Level data exists.
