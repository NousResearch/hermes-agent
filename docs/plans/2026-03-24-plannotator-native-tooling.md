# Plannotator Native Tooling and Generic Exposure Integration

> For Hermes: this documents the design discussion with Philipp that led from the temporary skill-based Plannotator flow to native tools.

## What we discussed

We had already patched `send_message` so Hermes can send inline status updates back into the active Telegram conversation using the current session context:
- `target=current` / `target=origin`
- automatic chat/thread/message resolution from `HERMES_SESSION_*`
- `reply_to_current=true` for inline reply threading

That solved the messaging plumbing, but the Plannotator flow was still skill-driven.

Philipp asked for two next steps:
1. document the Plannotator and router design we discussed
2. build a native tool for Plannotator
3. make the routing / exposure integration generic rather than hardcoded to only one wildcard-router setup

## Product direction

### Native Plannotator tool

Hermes should be able to launch a live Plannotator session directly from a tool call instead of relying only on a skill.

Core use cases:
- review the current local diff
- review a PR/MR URL
- annotate a markdown artifact
- eventually annotate the last assistant message

The tool should return:
- live URL
- PID if available
- log path if available
- a short suggested Telegram message

### Generic exposure abstraction

We do not want Plannotator tied forever to only one exposure backend.

The exposure layer should support multiple strategies:
- `localhost` — for local/manual opening
- `cloud77` — wildcard router or other custom reverse proxy
- `tailscale-serve` — tailnet-only exposure
- `tailscale-funnel` — public exposure
- one-off `command` templates for arbitrary operators or future bridges

This allows Hermes to keep the user-facing concept stable (“give me a live review link”) while changing the transport/backend later.

## Key design choice

Separate these concerns:
- `plannotator_session` handles “start a Plannotator review/annotation session”
- `service_expose` handles “turn a local service into a usable URL”

Even if Philipp’s current bridge combines launch + exposure today, Hermes should still have a generic exposure tool so future flows can use:
- plain localhost URLs
- cloud77 wildcard routing
- Tailscale serve/funnel
- custom operator scripts

## Implementation approach

### 1. `plannotator_session` native tool

A command-template-backed native tool.

Why command templates:
- keeps Hermes generic
- works with Philipp’s existing bridge immediately
- supports later migration to other launchers without changing the tool schema

Default assumptions for Philipp’s current setup:
- `python3 ~/services/plannotator-bridge/start_session.py review ...`
- `python3 ~/services/plannotator-bridge/start_session.py annotate ...`
- optional `last` support if the bridge exposes it

Override mechanism:
- per-call `command_template`
- or env vars:
  - `HERMES_PLANNOTATOR_REVIEW_TEMPLATE`
  - `HERMES_PLANNOTATOR_ANNOTATE_TEMPLATE`
  - `HERMES_PLANNOTATOR_LAST_TEMPLATE`

Expected launcher output convention:
- `URL=...`
- `PID=...`
- `LOG=...`

### 2. `service_expose` native tool

A generic, backend-agnostic exposure tool.

Supported strategies:
- `localhost`
- `cloud77`
- `tailscale-serve`
- `tailscale-funnel`
- `command`

For non-localhost strategies, the tool runs operator-controlled command templates that should emit a `URL=...` line.

Environment variable hooks:
- `HERMES_SERVICE_EXPOSE_CLOUD77_TEMPLATE`
- `HERMES_SERVICE_EXPOSE_TAILSCALE_SERVE_TEMPLATE`
- `HERMES_SERVICE_EXPOSE_TAILSCALE_FUNNEL_TEMPLATE`

This keeps the repo generic while still making Philipp’s setup easy to plug in.

## Why this is better than only skills

Skills were useful to reconstruct and operationalize the workflow quickly.
But native tools provide:
- repeatable JSON outputs
- first-class tool selection by the model
- easier future automation
- simpler integration with inline Telegram status updates
- a path toward a later dedicated MCP server or richer backend

## Follow-up ideas

1. Add a dedicated `plannotator_last` path if the launcher contract is finalized.
2. Let `plannotator_session` call `service_expose` automatically when the local launcher returns only a port instead of a public URL.
3. Add a small config section in `~/.hermes/config.yaml` for named exposure profiles rather than relying only on env vars.
4. Add richer result parsing if future bridges emit JSON instead of `KEY=value` lines.
