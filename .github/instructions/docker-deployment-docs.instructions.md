---
description: "Use when editing docker-compose.yml, INSTALL.md, docker deploy scripts, or deployment comments for the Ollama/OpenSpace stack. Keeps Docker docs aligned with the actual image source, state-preserving update commands, and mounted local customizations."
name: "Docker Deployment Docs"
applyTo:
  - "docker-compose.yml"
  - "docker-compose.upstream.yml"
  - "INSTALL.md"
  - "docker/*.sh"
  - "docker/*.yaml"
---

# Docker Deployment Docs

- Keep deployment comments and install docs aligned with the real compose behavior. If the stack uses upstream images, the docs should prefer `docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans`, not `--build`.
- Do not describe Hermes upstream images as GHCR-backed unless you have re-verified a public GHCR tag. The validated upstream publish target in this repo is `nousresearch/hermes-agent`.
- Preserve mention of fork-local overlays when they exist. In this repo, [docker/hermes-config.yaml](../../docker/hermes-config.yaml) stays bind-mounted in the default compose lane, while the default stack uses the upstream image's baked-in entrypoint instead of mounting [docker/entrypoint.sh](../../docker/entrypoint.sh).
- Treat [docker-compose.upstream.yml](../../docker-compose.upstream.yml) as the preferred pulled-image refresh lane for this fork's routine updates against the published upstream image.
- If `8644` is documented, be explicit about whether the webhook platform is actually enabled by default or whether the port is only pre-published for later use.
- Keep `docker-compose.yml` and `docker-compose.upstream.yml` port mappings in sync. The canonical gateway ports are `8642` (API server), `8789` (health), `8644` (webhook). If one compose file exposes a port the other doesn't, treat `docker-compose.upstream.yml` as the reference and sync the other. See [api-gateway-ports.instructions.md](api-gateway-ports.instructions.md) for the full port map and env var reference.
- The INSTALL.md Ports Reference table must list all exposed host ports: `9119` (dashboard), `8642` (API server), `8789` (health), `8644` (webhook), `5433→5432` (PostgreSQL). Update the table whenever a port binding changes.
- Treat `data/.env`, `data/config.yaml`, and the PostgreSQL volume as persistent state in both docs and commands. Avoid suggesting `docker compose down -v` except for explicit reset instructions.
- When changing update instructions, include a smoke-test step that checks `docker compose ps`, the dashboard status endpoint, and the API endpoint used by OpenSpace.