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
This will, in order: add the registry entry, write the token to
`/opt/data/secrets/<name>.env` (0600), refuse to continue if the token is
empty, then run `hermes profile create <name> --clone --clone-from cheap-template`
which registers and starts the gateway.

### 3. Handle the two common cases
- **3a. You have the token** → run the command above. Done.
- **3b. Token not available yet** → stage without starting the gateway:
  ```bash
  python -m hermes_cli.provision_client <name> --env prod --allow-empty-token --model deepseek-v4-flash
  ```
  Tell the user to put the token in `/opt/data/secrets/<name>.env`, then re-run
  step 2 (it's idempotent — it only completes the missing activation).

## Verification

After provisioning, confirm the bot is actually live:
1. `hermes profile list` shows `<name>`.
2. `hermes gateway status` (or the per-profile gateway state) shows its gateway
   running.
3. Send a test message to the client's Telegram bot and confirm a reply.

If the gateway is registered but the bot doesn't answer, the usual cause is the
profile's `config.yaml` not picking up the token — verify how this deployment
maps `<NAME>_TG_TOKEN` / the secret into the profile's Telegram config, and that
the profile's model/provider is set (cloning a known-good template avoids this).

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
