# Hermes Agent Product Overview

Hermes is a personal AI agent runtime: a long-lived gateway process that connects
chat platforms (Discord first) to an agentic LLM loop with tools, skills, and
scheduled jobs. This fork (zealchaiwut/hermes-agent) additionally serves as the
**orchestration layer for the owner's personal automation stack** — journal,
perf-coach, and Commander — under the "Hermes MVP1" program.

## Core Features

- **Discord bot (gateway, not webhook)** — slash commands, interactive button
  components, threads, reactions; auth-checked interactions.
- **Skills system** — packaged instructions + scripts installed under
  `~/.hermes/skills/`, loadable per session or attached to cron jobs
  (e.g. the `commander-api` skill bridges to a local Commander dashboard).
- **Cron scheduling** — in-process scheduler (gateway ticks every 60s) with
  per-profile persisted jobs; jobs can load skills and deliver output to Discord.
- **CLI** — `hermes` / `hermes-agent` entry points; launchd service management
  keeps the gateway alive on macOS.

## MVP1 orchestration role (smart server, lean client)

Hermes is deliberately **boring**: a scheduler, a contract reader, a renderer
(to Discord), and a dispatcher. Leaf projects keep all domain intelligence and
expose thin surfaces:

- **journal** emits `journal_brief.latest.json` (reflection, todos, threads).
- **perf-coach** emits `perfcoach_brief.latest.json` (today/tomorrow sessions,
  CTL/ATL/TSB form, adherence wrap, advisories) and owns all training logic.
- **Commander** emits `commander_report.latest.json` after overnight sprint runs
  and accepts dispatch/backlog commands over its HTTP API (bearer token).

Hermes composes a four-section morning brief from those contracts (~06:00
Asia/Bangkok), gates Commander's overnight run behind an explicit Discord
confirmation (default: do nothing), routes journal dev-todos to Commander's
backlog **only on explicit approval**, and forwards `/rpe` training feedback
into perf-coach's feedback store without interpreting it.

## Design Principles

- Single-user, local-first (mac-mini host); secrets via env/config, never committed.
- No leaf-project logic in Hermes — training math, reflection logic, and sprint
  logic live in their own repos; push logic back to the leaf when in doubt.
- Degrade gracefully: a missing or stale contract renders as "unavailable,"
  never a crash.
- Overnight automation merges only to `develop`; `main`/PRD promotion is a
  human action. Mutating actions are idempotent and audit-logged.
