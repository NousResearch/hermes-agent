---
name: fork-sync-review
description: "Review fork divergence, sync origin/main with upstream/main, keep this fork aligned as a deployment wrapper around the published upstream nousresearch/hermes-agent image, preserve data/.env and Postgres state, and produce a smoke-test plan for deployment updates."
argument-hint: "Describe the fork sync, merge, or image migration question"
user-invocable: true
---

# Fork Sync Review

Use this skill when the task is to inspect how far this fork has drifted from upstream, decide whether deployment changes should stay fork-local, or keep the fork aligned as a deployment wrapper around the published upstream image without losing runtime state.

## Procedure

1. Inspect remotes and divergence with `git remote -v`, `git fetch upstream --prune`, `git fetch origin --prune`, and `git rev-list --left-right --count origin/main...upstream/main`.
2. Review fork-only deployment files before changing the running stack: [docker-compose.yml](../../../docker-compose.yml), [INSTALL.md](../../../INSTALL.md), [docker/deploy.sh](../../../docker/deploy.sh), and [docker/hermes-config.yaml](../../../docker/hermes-config.yaml). Read [docker/entrypoint.sh](../../../docker/entrypoint.sh) when startup behavior or entrypoint drift is part of the question.
3. Check the currently running containers and images with `docker compose ps` and compare them to the documented upstream image flow in [website/docs/user-guide/docker.md](../../../website/docs/user-guide/docker.md) and [docker publish workflow](../../../.github/workflows/docker-publish.yml).
4. Preserve local state. Never delete `data/.env`, `data/config.yaml`, `data/memories/`, `data/sessions/`, or the PostgreSQL volume unless the user explicitly asks.
5. Prefer `nousresearch/hermes-agent:latest` for upstream image pulls unless a different official tag has been explicitly verified. Do not assume a GHCR package exists without checking.
6. After any deployment edit, validate with `docker compose config`, then use `docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans` for upstream-image stacks, followed by health checks on the dashboard and API endpoints.

## Output Expectations

- Report fork divergence as `ahead/behind` counts.
- Separate fork-only deployment customizations from upstream changes that can be adopted safely.
- Call out anything that would change credentials, mounted config, or persistent data before doing it.
- End with the concrete commands needed to smoke test the updated stack.