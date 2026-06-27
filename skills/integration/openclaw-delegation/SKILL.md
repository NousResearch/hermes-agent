---
name: openclaw-delegation
description: Delegate approved dry-run tasks from Hermes to local OpenClaw through the openclaw_delegate tool; use ClawOps vocabulary when the user refers to the OpenClaw operations hub.
version: 1.0.0
author: local
license: MIT
metadata:
  hermes:
    tags: [Integration, OpenClaw, Delegation]
    requires_tools: [openclaw_delegate]
required_environment_variables:
  - name: OPENCLAW_GATEWAY_URL
    prompt: OpenClaw Gateway URL, for example http://127.0.0.1:1455
    required_for: Local OpenClaw bridge access
  - name: OPENCLAW_GATEWAY_TOKEN
    prompt: OpenClaw Gateway bearer token
    required_for: Gateway authentication
  - name: OPENCLAW_HERMES_BRIDGE_TOKEN
    prompt: OpenClaw Hermes bridge shared secret
    required_for: Plugin-local bridge authentication
---

# OpenClaw Delegation

## Relationship to ClawOps

ClawOps is the operating name for the OpenClaw multi-agent operations hub.
Hermes-Grace is the user-facing chief-of-staff assistant; ClawOps is the hub
Grace routes to when the user asks for coordinated OpenClaw agent work.

This `openclaw-delegation` skill describes the current safe bridge behavior:
Hermes can delegate approved OpenClaw dry-run requests through
`openclaw_delegate`. Do not confuse that with full ClawOps v1 runtime execution.
If the user asks for ClawOps responsibility or model assignments, use the
`clawops` skill. If the user asks to run a currently supported OpenClaw dry-run,
use this skill.

## When to Use

Use this skill when the user asks Hermes to have OpenClaw handle a local task and
explicitly says it should be a dry-run or should avoid external side effects.

## Procedure

For the v1 bridge, only delegate this approved task:

| User intent | taskId |
| --- | --- |
| Organize today's tasks as a dry-run | `tasks.organize_today` |
| Ask the OpenClaw agent team for dry-run analysis | `agents.ask_team` |

Call `openclaw_delegate` with:

```json
{
  "taskId": "tasks.organize_today",
  "intent": "<the user's original request>",
  "dryRun": true,
  "allowedTools": [],
  "input": {
    "request": "<the user's original request>"
  }
}
```

For OpenClaw agent team dry-runs, call `openclaw_delegate` with:

```json
{
  "taskId": "agents.ask_team",
  "intent": "<the user's original request>",
  "dryRun": true,
  "allowedTools": [],
  "input": {
    "team": "openclaw",
    "question": "<the user's question for the OpenClaw agent team>"
  }
}
```

After the tool returns, summarize the `summary` field and include the important
`auditLog` entries. Make clear that the task was a dry-run and that no external
side effects were performed.

## Safety Rules

- Do not request real email, calendar, messaging, trading, filesystem, shell,
  browser, node, cron, or Gateway-control execution through this bridge.
- Do not pass arbitrary OpenClaw tool names.
- Do not set `dryRun` to false.
- Do not add entries to `allowedTools`.
- If the user asks for real execution, explain that v1 supports dry-run only.
- Do not claim live OpenClaw agents were contacted when `agents.ask_team` returns a dry-run result.

## Verification

A successful v1 response includes:

```json
{
  "status": "succeeded",
  "summary": "Dry-run completed. No external side effects were performed."
}
```
