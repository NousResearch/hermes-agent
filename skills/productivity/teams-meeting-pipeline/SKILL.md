---
name: teams-meeting-pipeline
description: "Operate the Teams meeting summary pipeline via Hermes CLI."
version: 1.0.0
author: Hermes Agent
license: MIT
prerequisites:
  env_vars: [MSGRAPH_TENANT_ID, MSGRAPH_CLIENT_ID, MSGRAPH_CLIENT_SECRET]
  commands: [hermes]
metadata:
  hermes:
    tags: [Teams, Microsoft Graph, Meetings, Productivity, Operations]
---

# Teams Meeting Pipeline

Use this skill when the user asks to summarize a Teams meeting, extract action items, inspect pipeline status, replay a stored job, or validate Microsoft Graph meeting-ingest setup.

Prefer the Hermes CLI over ad hoc scripts. Route operator actions through the terminal tool with `hermes teams-pipeline ...`.

## When to use

- "Teams meeting ozetle"
- "action item cikar"
- "toplanti notu"
- "pipeline durumu"
- "replay job"

## Required environment

Set these in `~/.hermes/.env` before using the pipeline:

```bash
MSGRAPH_TENANT_ID=...
MSGRAPH_CLIENT_ID=...
MSGRAPH_CLIENT_SECRET=...
```

## Common commands

```bash
hermes teams-pipeline list
hermes teams-pipeline show <job-id>
hermes teams-pipeline replay <job-id>
hermes teams-pipeline fetch --meeting-id <meeting-id>
hermes teams-pipeline token-health
hermes teams-pipeline maintain-subscriptions
```

Start with `validate`, `list`, or `show` when the user asks for status. Use `replay` only when they explicitly want to rerun a stored job. Use `fetch` for dry-run artifact checks before changing pipeline config.
