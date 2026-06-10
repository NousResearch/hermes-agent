---
name: fleet-ops
description: "Operate the bot fleet from the host: spin up / tear down / restart / inspect a client's ISOLATED container, and regenerate the compose from the registry. Use when asked to graduate a client to its own container, start/stop/restart a containerized bot, or check fleet container status. For creating a SHARED-container profile bot, use provision-client instead."
version: 1.0.0
platforms: [linux]
metadata:
  hermes:
    tags: [devops, fleet, orchestration, containers, multi-tenant, isolation]
    related_skills: [provision-client]
---

# Fleet Ops — Hermes' host-level control of the bot fleet

## Overview

Hermes owns the fleet. Creating a **shared-container** bot is fully in-container
(`provision-client`). Giving a client its **own container** — and starting,
stopping, restarting, or inspecting that container — is a *host* operation,
because containers live on the host's Docker daemon, not inside Hermes.

Hermes performs these host operations through a **bounded broker**: an
unprivileged `hermes-ops` host user whose SSH key can run ONLY an allowlisted
set of `docker`/`compose` actions (`scripts/fleet_broker.py`) — never an
arbitrary shell. The `hermes fleet` client (`python -m hermes_cli.fleet`) is the
in-container handle; it reaches the broker over loopback SSH. The broker and key
are provisioned automatically by the deploy — there is nothing to set up by hand.

## When to use

- "Graduate <client> to its own container" / "isolate <client>."
- "Start / stop / restart <client>'s container."
- "Is <client>'s container running?" / "show fleet status" / "tail <client> logs."

**Use `provision-client` instead** to create a normal shared-container bot.
**This skill does not edit code** — for that, edit the repo and push (the deploy
pipeline builds + ships, gated by the health and proxy-lock checks).

## Division of labor (important)

- **Registry + compose generation happen in-container** (here), where the
  tested, registry-aware `compose_gen` lives:
  `python -m hermes_cli.fleet generate` reads `clients.yaml` and writes
  `docker-compose.clients.yml` + `isolated.list` onto the shared volume.
- **The host broker only runs `docker`/`compose`** against those generated
  artifacts. It never parses the registry or invents services.

## Workflow: graduate a client to its own container

1. Mark the client isolated in the registry (`isolation: container` in
   `$HERMES_CLIENTS_REGISTRY`), and ensure its per-env data volume exists
   (`client_split` if you are splitting prod/dev data).
2. Regenerate the compose artifacts from the registry:
   ```
   python -m hermes_cli.fleet generate --data-root /opt/data --env prod
   ```
3. Bring the client's container up:
   ```
   python -m hermes_cli.fleet up <client>
   ```
   (or `apply` to converge the whole fleet to the generated compose at once).
4. Verify: `python -m hermes_cli.fleet status <client>`.

The boot registry-gate already excludes an `isolation: container` client from
the shared gateway, so graduating a client does not double-run its bot.

## Commands

| Command | What it does | Where it runs |
|---|---|---|
| `fleet generate [--env E] [--data-root D]` | regen `docker-compose.clients.yml` + `isolated.list` from the registry | in-container |
| `fleet apply` | `compose up -d --remove-orphans` the generated fleet | host broker |
| `fleet up <client>` | start one client's isolated container | host broker |
| `fleet down <client>` | stop + remove one client's container | host broker |
| `fleet restart <client>` | restart one client's container | host broker |
| `fleet status <client>` | container running/restarting/status | host broker |
| `fleet logs <client> [lines]` | tail container logs (1..2000, default 100) | host broker |
| `fleet ps` | list all fleet containers | host broker |

## Guardrails (do not try to work around these)

- The broker accepts only the subcommands above; client names must be lowercase
  slugs (`[a-z0-9][a-z0-9-]{0,30}`). Anything else is refused and audited
  (`/opt/hermes/fleet/audit.log` on the host).
- The broker key cannot open a shell, forward ports, or run arbitrary commands —
  by design. If you need a genuinely new host capability, add a new allowlisted
  subcommand to `scripts/fleet_broker.py` in the repo and push (it ships through
  the normal gated deploy and the next deploy reinstalls the broker).
- If `hermes fleet` reports the broker is unreachable, the host bootstrap hasn't
  completed (or failed) — check the latest deploy's `fleet:` log lines. Bots are
  unaffected either way; fleet ops are additive.
