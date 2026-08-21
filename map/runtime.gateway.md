---
id: runtime.gateway
kind: object
universe: runtime
name: gateway
summary: Messaging gateway backend serving Telegram, Discord, Slack, and others.
aliases: [gateway]
tags: [runtime, messaging]
shape: object
path: gateway/run.py
interface: [GATEWAY_KNOWN_COMMANDS, resolve_command]
depends_on: [repo:gateway.session.py]
---

# gateway

Messaging gateway backend serving Telegram, Discord, Slack, and others.

## Location

`gateway/run.py`

## Interface

- `GATEWAY_KNOWN_COMMANDS`
- `resolve_command`

## Dependencies

- `repo:gateway.session.py`
