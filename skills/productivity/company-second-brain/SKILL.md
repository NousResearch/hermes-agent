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

## When To Use

Use this skill when the user asks to:

- Search company knowledge, second brain, internal docs, or C-Level docs.
- Generate setup instructions for Codex, Hermes, Cursor, Claude Code, or another IDE agent.
- Create a user/agent token as an admin.
- Ingest a plain-text document into the company RAG.

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

## First-Time User Setup

Ask the user for their bearer token if it is not already configured.

Configure the CLI:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain connect \
  --base-url __PUBLIC_BASE_URL__ \
  --token "USER_TOKEN"
```

Verify identity and accessible workspaces:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain me
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain workspaces
```

The CLI stores config at:

```text
~/.second-brain/config.json
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

## Admin Token Generation

Configure admin key:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain config \
  --base-url __PUBLIC_BASE_URL__ \
  --admin-key "GATEWAY_API_KEY"
```

Generate a normal employee token:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain admin-token-create \
  --email user@company.local \
  --name "Company User" \
  --group company_all \
  --expires-days 30
```

Generate an admin bearer token:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain admin-token-create \
  --email admin@company.local \
  --name "Admin" \
  --group role_admin \
  --admin \
  --expires-days 365
```

## Ingest Text

Admin only. Ingest is asynchronous: the API returns `queued`, then
`knowledge-worker` indexes the document into LightRAG in the background.

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
