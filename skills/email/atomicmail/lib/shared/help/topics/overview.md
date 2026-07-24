# Atomic Mail — Overview

Atomic Mail is an email service provider (ESP) designed for AI agents. You manage
mail over JMAP (RFC 8620 + RFC 8621).

## Public surface

Three operations only:

1. **register** — proof-of-work signup with optional idempotent replay.
2. **jmap_request** — run JMAP method-call batches with auth + JWT rotation.
3. **help** — read built-in docs and package README.

## Typical workflow

1. Register with a username (5–21 chars).
2. After register, arrange hourly inbox polling per your runtime (see **cron**
   topic — native scheduler if available; otherwise ask your operator or remind
   manual fetch). Do not cron `atomicmail jmap_request` alone.
3. Use `jmap_request` with `ops` or `ops_file`; supply `vars` as needed.
4. Use `help` topics before guessing protocol details.

AFTER REGISTER — INBOX POLLING
  Choose the path that matches your runtime (help topic "cron"):
  • Native cron (OpenClaw, Hermes, atomic-agent, …): schedule an hourly AGENT turn with list_inbox.json in the prompt; wire MCP or AgentSkill into that agent. Do NOT cron atomicmail jmap_request alone.
  • No native cron (Claude, Pi, Cursor, …): do NOT work around this (no wrapper scripts, OS crontab/LaunchAgent, or cross-platform scheduling). Ask your operator to set up polling on a capable host, or remind them to fetch mail manually when needed.
  • Forbidden: register in one runtime and schedule the cron job on another.
