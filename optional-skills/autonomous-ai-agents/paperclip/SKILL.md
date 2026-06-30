---
name: paperclip
description: Use Paperclip as an external company-style control plane for teams of agents, budgets, heartbeats, and governance. Best when the user wants persistent multi-agent operations, not just one-off delegation.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Paperclip, Agent-Company, Budgets, Governance, Heartbeats, Dashboard]
    related_skills: [hermes-agent, subagent-driven-development, watchers]
---

# Paperclip

Use [Paperclip](https://github.com/paperclipai/paperclip) when the user wants a persistent operations layer for teams of agents: org charts, budgets, scheduled heartbeats, approvals, and dashboards. Hermes can help install, inspect, and integrate it, but Paperclip should remain an external system rather than be folded into Hermes core.

## When to Use

- The user wants long-lived agent teams with roles and reporting lines.
- Budgets, approvals, and governance matter.
- Agents should wake up on schedules or heartbeats.
- The user wants a web app to manage work at the company level.

Prefer Hermes `delegate_task` or cron for smaller local automation. Use Paperclip when the orchestration scope is organization-wide.

## Prerequisites

- Node.js 20+
- pnpm 9.15+
- Browser for the UI

## Quickstart

Paperclip's upstream quickstart is:

```bash
terminal(command="npx --registry https://registry.npmjs.org paperclipai onboard --yes", timeout=900, pty=true)
```

The explicit npm registry avoids failures on machines pointed at private registries.

For LAN or tailnet binding, upstream supports:

```bash
terminal(command="npx --registry https://registry.npmjs.org paperclipai onboard --yes --bind lan", timeout=900, pty=true)
terminal(command="npx --registry https://registry.npmjs.org paperclipai onboard --yes --bind tailnet", timeout=900, pty=true)
```

## Local Development

```bash
terminal(command="git clone https://github.com/paperclipai/paperclip.git", workdir="/tmp")
terminal(command="pnpm install", workdir="/tmp/paperclip", timeout=900)
terminal(command="pnpm dev", workdir="/tmp/paperclip", background=true, notify_on_complete=true)
```

## Hermes Adoption Target

Adopt the governance ideas selectively:

- budgets and approvals are useful design input for future workflow guardrails
- heartbeats and org-chart management should stay in Paperclip, not Hermes core
- Hermes can act as a worker, reviewer, or bootstrapper inside a Paperclip-managed system
- avoid duplicating Paperclip's company-control-plane concepts in native Hermes memory or cron primitives

## Verification

Upstream verification commands:

```bash
terminal(command="pnpm build", workdir="/path/to/paperclip", timeout=900)
terminal(command="pnpm typecheck", workdir="/path/to/paperclip", timeout=900)
terminal(command="pnpm test", workdir="/path/to/paperclip", timeout=900)
```

## Hermes Integration Shape

- Hermes can provision or inspect the Paperclip repo.
- Hermes can draft policies, role prompts, and operating procedures.
- Hermes should not mirror Paperclip's org chart, budget, or heartbeat engine into core memory/tools.
- If the user already runs Paperclip, Hermes can operate as one worker in that larger system.

## Pitfalls

- Paperclip is a full product, not a drop-in library.
- Budget/governance settings can create surprising no-op behavior if left unset.
- Treat mobile/web access and exposed binds as security-sensitive.
- Avoid duplicating the same schedules in both Paperclip and Hermes cron unless the owner is explicit.

## Related

- Upstream: https://github.com/paperclipai/paperclip
- Docs: https://github.com/paperclipai/paperclip-docs
