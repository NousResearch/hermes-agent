# Plannotator Native Tooling and Generic Exposure Integration

## Goal

The goal is to integrate Plannotator review and annotate workflows as native
Hermes tools for interactive feedback on code changes and files.

That means Hermes should be able to:
- open a browser-based Plannotator session for a diff or artifact
- send the live review URL into chat
- wait for browser-submitted feedback when needed
- continue the conversation with the returned review result

This note captures the design direction that led from a temporary skill-based
workflow to native Hermes tooling.

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
- a short suggested chat message

### Generic exposure abstraction

Plannotator should not be tied to only one exposure backend.

The exposure layer should support multiple strategies:
- `localhost` — for local/manual opening
- `reverse-proxy` — operator-controlled routed exposure
- `tailscale-serve` — tailnet-only exposure
- `tailscale-funnel` — public exposure
- one-off `command` templates for arbitrary operators or future bridges

This lets Hermes keep the user-facing concept stable — “give me a live review link” — while changing the transport/backend later.

## Key design choice

Separate these concerns:
- `plannotator_session` handles “start a Plannotator review/annotation session”
- `service_expose` handles “turn a local service into a usable URL”

Even if a local bridge combines launch + exposure today, Hermes should still have a generic exposure tool so future flows can use:
- plain localhost URLs
- routed host exposure through a reverse proxy
- Tailscale serve/funnel
- custom operator scripts

## Implementation approach

### 1. `plannotator_session` native tool

A command-template-backed native tool.

Why command templates:
- keeps Hermes generic
- works with existing local bridges immediately
- supports later migration to other launchers without changing the tool schema

Default assumptions for a typical local setup:
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
- `reverse-proxy`
- `tailscale-serve`
- `tailscale-funnel`
- `command`

For non-localhost strategies, the tool runs operator-controlled command templates that should emit a `URL=...` line.

Environment variable hooks:
- `HERMES_SERVICE_EXPOSE_REVERSE_PROXY_TEMPLATE`
- `HERMES_SERVICE_EXPOSE_TAILSCALE_SERVE_TEMPLATE`
- `HERMES_SERVICE_EXPOSE_TAILSCALE_FUNNEL_TEMPLATE`

## Inline UX requirement

A single blocking Plannotator tool call is not enough for the best chat UX if Hermes must send the review URL before waiting for completion.

The cleaner architecture is:
1. pre-generate or reserve the host
2. send the URL immediately into the active conversation
3. launch Plannotator pinned to that exact host
4. wait for completion
5. return the final feedback

That design led to integrated inline actions in the Plannotator tool rather than relying on the model to manually compose prepare + send_message + wait steps.

## Why this is better than only skills

Skills were useful to reconstruct and operationalize the workflow quickly.
But native tools provide:
- repeatable JSON outputs
- first-class tool selection by the model
- easier future automation
- simpler integration with inline chat updates
- a path toward a later dedicated MCP server or richer backend

## Follow-up ideas

1. Add a dedicated `plannotator_last` path if the launcher contract is finalized.
2. Let `plannotator_session` call `service_expose` automatically when the local launcher returns only a port instead of a public URL.
3. Add a small config section in `~/.hermes/config.yaml` for named exposure profiles rather than relying only on env vars.
4. Add richer result parsing if future bridges emit JSON instead of `KEY=value` lines.
5. Consider a future Plannotator-side plugin/API if Hermes integration becomes a first-class product surface.
