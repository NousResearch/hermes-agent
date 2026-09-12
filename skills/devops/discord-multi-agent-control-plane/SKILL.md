---
name: discord-multi-agent-control-plane
description: Use Discord as transparency/control plane for multi-agent systems while keeping agent logic and detailed state outside Discord.
---

# Discord multi-agent control plane

Use this when a user wants Discord integrated with Hermes/OpenClaw/Pluto for visibility, coordination, or lightweight human-in-the-loop control.

## Core rule
Discord is the UI/event layer, not the internal agent bus.

Recommended split:
1. Discord = alerts, task intake, handoffs, approvals, short summaries
2. Mission Control / internal services = status detail, logs, dashboards, run state
3. Internal queue/protocol/router = agent-to-agent communication
4. Obsidian = durable decisions, architecture notes, runbooks

## Why
- Prevents bot reply loops
- Keeps routing deterministic
- Improves security and auditability
- Avoids flooding Discord with machine chatter
- Makes human oversight simple

## Recommended architecture
1. User posts command/task in Discord or Telegram
2. Hermes gateway normalizes the event
3. Router chooses the right agent/system
4. Agents execute through internal tools/protocols
5. Only relevant summaries, alerts, and requests-for-input are posted back to Discord
6. Detailed progress stays in Mission Control

## Channel design pattern
- #inbox or #tasks: new requests
- #alerts: failures, escalations, important completions
- #agents: cross-agent summaries and handoffs
- #builds or #ops: deploy/build notifications
- #memory: extracted durable facts and decisions
- #study / #sales / domain channels: project-specific summaries

## Routing rules
- Prefer explicit routing by channel, thread, role, or mention
- One orchestrator decides what to fan out to specialist agents
- Agents should ignore messages not addressed to their route
- Never let agents auto-reply to each other broadly in the same channel

## What to post to Discord
Post:
- concise task status
- blockers
- escalations
- handoff summaries
- final outcome summaries

Do not post:
- verbose logs
- token-level reasoning
- repeated heartbeat/status spam
- unrestricted bot-to-bot chatter

## Implementation guidance
- Use Discord bot/webhook outputs for notifications and controlled commands
- Keep state outside Discord
- Link back to Mission Control for details
- Save final architecture/runbooks in Obsidian

## Decision heuristic
If the message helps a human notice, steer, approve, or understand a task quickly, Discord is a good destination.
If it is long-lived state, detailed telemetry, or internal machine coordination, keep it outside Discord.

## Pitfalls
- Feedback loops between bots
- Overloading one channel with every event
- Letting Discord become the only system of record
- Mixing human discussion, internal telemetry, and agent routing in one place

## Default recommendation
Choose: Discord as control/visibility layer + Mission Control as source of detail + internal router/protocol for agent execution.