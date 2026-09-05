---
description: "Use when syncing this fork with upstream main, comparing fork divergence, keeping the deployment wrapper aligned with the published upstream nousresearch/hermes-agent image, preserving data/.env credentials and Postgres state, or smoke-testing container updates."
name: "Fork Maintenance"
applyTo:
  - "docker-compose.yml"
  - "INSTALL.md"
  - "docker/*.sh"
  - "docker/*.yaml"
  - ".github/workflows/docker-publish.yml"
---

# Fork Maintenance

## Start From Current State

- Verify remotes before planning merges. The usual layout in this fork is `origin` for the personal fork and `upstream` for `NousResearch/hermes-agent`, but always confirm with `git remote -v`.
- Read [docker-compose.yml](../../docker-compose.yml), [INSTALL.md](../../INSTALL.md), [docker/hermes-config.yaml](../../docker/hermes-config.yaml), and [docker publish workflow](../workflows/docker-publish.yml) before proposing Docker changes. Read [docker/entrypoint.sh](../../docker/entrypoint.sh) when startup behavior or image-entrypoint drift is part of the task.
- The upstream container reference is [website/docs/user-guide/docker.md](../../website/docs/user-guide/docker.md).

## Safety Rules

- Preserve local runtime state: never overwrite, regenerate, or delete `data/.env`, `data/config.yaml`, `data/SOUL.md`, `data/memories/`, `data/sessions/`, or the `postgres_data` volume unless the user explicitly asks.
- Avoid destructive reset commands like `docker compose down -v` or deleting `data/` during sync or image-migration work.
- Do not push fork-specific deployment changes to `upstream`. Merge or rebase locally, then push only to `origin` unless the user explicitly wants an upstream contribution.
- Treat bind-mounted files as part of the deployment contract. By default [docker/hermes-config.yaml](../../docker/hermes-config.yaml) stays mounted, so upstream image updates can still be partially shaped by local config. If a task reintroduces [docker/entrypoint.sh](../../docker/entrypoint.sh) or another overlay, call out that additional pinning risk explicitly.

## Evaluating Upstream Images

- [docker-compose.yml](../../docker-compose.yml) keeps this fork as a deployment wrapper around the fork's published GHCR image `ghcr.io/jzkk720/hermes-agent:latest` rather than a local `hermes-agent:local` build.
- The fork publishes its own GHCR image via [fork-ghcr-publish workflow](../workflows/fork-ghcr-publish.yml), which builds from the fork's source (upstream + Weixin QR onboarding + dashboard auth) on every push to main.
- For routine updates in this fork, prefer `docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans`; do not switch to `--build` or fork-owned image tags unless the user is explicitly testing image contents.
- Treat [docker-compose.upstream.yml](../../docker-compose.upstream.yml) as the preferred pulled-image refresh lane for this fork when the goal is to stay on the published upstream image while keeping fork-local mounts and state.
- Re-check whether the default [docker/hermes-config.yaml](../../docker/hermes-config.yaml) mount and any optional overlays are still necessary; they preserve the fork wrapper but can also block future upstream behavior changes.

## Merge and Divergence Checks

- Inspect divergence with `git fetch upstream --prune`, `git fetch origin --prune`, and `git rev-list --left-right --count origin/main...upstream/main` before discussing merge strategy.
- If the fork carries local-only deployment commits, keep them on `origin/main` or a dedicated branch. Do not merge `upstream/main` into the fork unless the user explicitly asks for that sync path.
- If the user wants to review upstream without integrating it yet, compare commits and diffs without rewriting remotes or force-pushing.

## Smoke Test Expectations

- After editing compose or deployment docs, validate with `docker compose config`.
- For upstream-image stacks, use `docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans`.
- Check `docker compose ps`, the dashboard health endpoint `http://127.0.0.1:9119/api/status`, and the agent API `http://127.0.0.1:8789/v1/models` when available.
- Report any step that would change credentials, rebuild volumes, or replace mounted config before doing it.