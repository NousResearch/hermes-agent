---
name: company-second-brain-start
description: Use when a user mentions Company Second Brain, LightRAG, company RAG, internal docs, company knowledge, C-Level docs, a configured second-brain gateway domain, or asks how to query or use the second-brain CLI in a fresh agent session.
---

# Company Second Brain Start

## Quick Start

Use the installed CLI first:

```bash
second-brain query "QUESTION"
```

If `second-brain` is not in `PATH`, use:

```bash
~/.hermes/skills/productivity/company-second-brain/scripts/second-brain query "QUESTION"
```

## Check Access

```bash
second-brain me
second-brain workspaces
```

Normal tokens can query `company_public`. Admin tokens can query both
`company_public` and `department_c_level`.

## Common Commands

```bash
# Ask company knowledge
second-brain query "tom tat tai lieu cong ty moi nhat"

# Raw JSON for another agent/tool
second-brain query "QUESTION" --raw

# Admin analytics
second-brain analytics --days 30 --limit 20
```

## If Config Is Missing

If the CLI says a token is missing, ask the user for their Company Second Brain
token and run:

```bash
second-brain connect --base-url __PUBLIC_BASE_URL__ --token "USER_OR_ADMIN_TOKEN"
```

The token may be raw JWT, `Bearer ...`, or `Authorization: Bearer ...`.

Do not ask for admin key unless the user explicitly wants to create tokens,
configure sources, or administer the VPS. Use the bearer token for normal query
work.

## Deeper Admin Work

For upload, Notion/Drive source setup, token creation, and detailed operations,
use the `company-second-brain` skill after this starter identifies the task.
