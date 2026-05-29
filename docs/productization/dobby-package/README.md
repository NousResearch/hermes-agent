# Dobby/Hermes Productization Package

Stage 3A defines the safe foundation for a sellable Dobby-branded Hermes
package. This directory is not the package implementation yet; it is the
coordination surface for later workers to build the implementation without
touching live Hermes runtime state.

## Product Scope

Dobby/Hermes is a self-hosted, Discord-first operator package for Hermes Agent.
The customer brings:

- A Discord application and bot token.
- A model endpoint and API key compatible with Hermes provider configuration.
- A fresh `HERMES_HOME` for this package.
- Optional webhook senders that can sign payloads with HMAC.

The package must ship these product claims:

- Discord command center for operator control.
- Health, quota, and status reporting.
- Research scout with mockable/golden-testable data paths.
- Reminders and cron delivery through the existing Hermes scheduler.
- Explicit attachment review before file contents are read or summarized.
- Guarded repo helper that is read-only and propose-only by default.
- Signed webhook inbox with reject-by-default authentication.
- Native Hermes memory using `SOUL.md`, built-in memory files, and
  `session_search`, with consent, forget, export, and delete flows.

## Non-Goals

- Do not bundle Dobby model weights.
- Do not bundle or auto-run a Honcho server.
- Do not enable broad messaging integrations by default.
- Do not copy secrets, `~/.hermes`, session history, logs, or personal runtime
  state from any existing installation.
- Do not mutate any live customer host such as
  `<LIVE_REMOTE_USER>@<LIVE_REMOTE_HOST>`.
- Do not make repo-helper actions write, commit, push, or deploy by default.

## Target Operator Quickstart

Later implementation slices should make this shape real:

1. Create a fresh runtime home, for example `<FRESH_HERMES_HOME>`.
2. Create a Discord application, bot, and private server/channel.
3. Copy the package env example and replace placeholders only:

   ```env
   HERMES_HOME=<FRESH_HERMES_HOME>
   DISCORD_BOT_TOKEN=<DISCORD_BOT_TOKEN>
   DISCORD_HOME_CHANNEL=<DISCORD_CHANNEL_ID>
   OPENAI_BASE_URL=<MODEL_ENDPOINT_URL>
   OPENAI_API_KEY=<MODEL_API_KEY>
   WEBHOOK_SECRET=<WEBHOOK_HMAC_SECRET>
   HERMES_REDACT_SECRETS=true
   ```

4. Run preflight. It must fail on placeholders, missing Discord permissions,
   weak webhook secrets, unsafe `HERMES_HOME`, and accidental live paths.
5. Start the gateway with only Discord and the signed webhook inbox enabled.
6. Run the demo kit against mock fixtures before using real data.

The final quickstart must use exact commands verified by tests or dry-run
scripts. Until then, this file describes the intended operator experience.

## Architecture Summary

The package is a thin, safe-by-default product profile around existing Hermes
capabilities:

```text
Discord user
  -> gateway/platforms/discord.py
  -> Dobby command center and policy gates
  -> Hermes agent loop
  -> BYO model endpoint and approved local tools
  -> Discord response

Signed webhook sender
  -> gateway/platforms/webhook.py
  -> HMAC, size, route, and idempotency gates
  -> Dobby inbox prompt
  -> Discord or log delivery

Memory
  -> SOUL.md, memories/MEMORY.md, memories/USER.md, SessionDB/session_search
  -> consent, export, forget, and delete controls
```

## Package Artifact Boundaries

This package surface contains coordination docs and deterministic policy-contract
tests only. It does not add runtime implementation code, runtime config files,
real secrets, generated artifacts, or service commands. Later workers should use
`IMPLEMENTATION_SLICES.md` as the coordination map and `TRACEABILITY.md` as the
evidence checklist.
