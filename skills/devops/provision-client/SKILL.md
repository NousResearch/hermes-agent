---
name: provision-client
description: "Provision a new client bot (a Hermes profile) in-container, no host access. Use when asked to add/create/onboard a new client, tenant, or bot."
version: 1.0.0
platforms: [linux]
metadata:
  hermes:
    tags: [devops, provisioning, multi-tenant, profiles, onboarding]
    related_skills: []
---

# Provision a New Client Bot

## Overview

A new client bot is a new **Hermes profile**. Creating one is fully
in-container — no Docker, no host access — because `hermes profile create`
registers and starts a supervised per-profile gateway live. This skill drives
the one-command orchestrator that records the client, wires its token, and
launches it.

This procedure is deterministic and guard-railed, so it is safe to run on a
cheap model (e.g. DeepSeek).

## When to use

- "Add a new client / tenant / bot named X."
- "Onboard <person> with their own Telegram bot."

**Do NOT use this skill to** create a client's own *container* or split data
volumes (`client_split`, `compose_gen`). Those need host Docker and are a
separate, human-approved step. This skill only ever provisions **profiles in
the shared container** (soft isolation).

## Prerequisites

- The client registry path is set: `export HERMES_CLIENTS_REGISTRY=/opt/data/clients.yaml`
- A **Telegram bot token** for the new client (from @BotFather). If you don't
  have it yet, stage the client without activating (see step 3b) and ask for it.
- Optional: a **template profile** to clone config from (e.g. one preset to a
  cheap default model + the standard persona). Recommended so every new bot is
  consistent. Create it once with `hermes profile create cheap-template`, set
  its `config.yaml` model, then clone it for each client.

## Inputs

1. **name** — lowercase slug (letters, digits, dashes), e.g. `acme`. Becomes the
   profile and the registry key.
2. **token** — the Telegram bot token.
3. **template** (optional) — profile to clone config from.
4. **model** (optional) — model slug to record for this client (default to a
   cheap model like `deepseek-v4-flash` unless told otherwise).

## Workflow

### 1. Validate
- Confirm the name is a clean slug. Reject spaces, uppercase, or symbols.
- Check it isn't already a live profile: `hermes profile list`. If it exists,
  the orchestrator will no-op the creation — that's fine for reconciliation.

### 2. Provision (one command)
```bash
python -m hermes_cli.provision_client <name> \
  --env prod \
  --token "<telegram-token>" \
  --model deepseek-v4-flash \
  --clone-from cheap-template
```
This will, in order: record the registry entry; refuse to continue if no token
is available; run `hermes profile create <name> --clone --clone-from cheap-template`
(registers + starts the gateway); write the token into the **profile's own**
`.env` as `TELEGRAM_BOT_TOKEN` — which is exactly what a per-profile gateway
reads (it runs with `HERMES_HOME` = its profile dir), overriding any token
cloned from the template; then `hermes gateway restart --profile <name>` so the
gateway picks it up.

> Token location: a profile bot reads `TELEGRAM_BOT_TOKEN` from
> `/opt/data/profiles/<name>/.env`. The registry's separate
> `/opt/data/secrets/<name>.env` (`<NAME>_TG_TOKEN`) is only for the
> container-isolation path and is NOT read by a profile gateway.

### 3. Handle the two common cases
- **3a. You have the token** → run the command above. Done.
- **3b. Token not available yet** → stage without a working gateway:
  ```bash
  python -m hermes_cli.provision_client <name> --env prod --allow-empty-token --model deepseek-v4-flash
  ```
  When you get the token, re-run step 2 with `--token` (idempotent: it skips
  re-creating the profile, writes the token into the profile's `.env`, and
  restarts the gateway).

## Verification

After provisioning, confirm the bot is actually live:
1. `hermes profile list` shows `<name>`.
2. `hermes gateway status` (or the per-profile gateway state) shows its gateway
   running.
3. Send a test message to the client's Telegram bot and confirm a reply.

If the gateway is registered but the bot doesn't answer, check
`/opt/data/profiles/<name>/.env` actually contains a valid `TELEGRAM_BOT_TOKEN`
and that you restarted the gateway after writing it (`hermes gateway restart
--profile <name>`). Also confirm the profile's model/provider is set (cloning a
known-good template avoids this).

## Idempotency & safety

- Re-running is safe: the registry entry and profile are not duplicated; an
  already-created profile is left untouched.
- The orchestrator refuses to *activate* a bot whose token is still empty, so a
  cheap model can't spin up a dead bot.
- Secrets are 0600 and never leave the host volume; the registry never stores
  token values, only the env-var name.

## Out of scope (escalate to a human)

- Creating a dedicated **container** for a client (`isolation: container` →
  `compose_gen` → `docker compose up`).
- Creating sibling **data volumes** (`client_split` into `/opt/data-prod` etc.).
- Anything requiring the Docker socket or host root.
