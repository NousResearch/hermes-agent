---
name: homelab-service-onboarding
description: "Use when adding a new service to Justin's homelab from upstream Docker Compose docs through a homelab GitHub repo PR, Portainer Git stack deployment, optional Nginx Proxy Manager exposure, optional Authentik login, and optional API-key setup."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [homelab, docker-compose, portainer, gitops, nginx-proxy-manager, authentik, secrets]
    related_skills: [devops-infrastructure-operations, github-pr-workflow]
---

# Homelab Service Onboarding

## Overview

Use this skill for the full lifecycle of adding a new service to Justin's homelab: find the upstream Docker Compose source, add a clean stack definition to the homelab GitHub repository, open a PR, then deploy it as a Git-backed Portainer stack with the right environment variables and persistent storage. Optional layers include Nginx Proxy Manager exposure, Authentik login, and creating a scoped API key for Hermes or MCP/tool use.

Core model:

- **Upstream docs are discovery.** Do not blindly paste examples into production.
- **The homelab GitHub repo is source of truth.** Compose should be reviewed in Git before deployment.
- **Portainer Git stacks are deployment mechanism.** Prefer Git-backed stacks with Git updates enabled.
- **NPM, Authentik, and API keys are integration layers.** Add them only when requested or clearly needed.

## When to Use

Use when the user asks to:

- Add, deploy, install, self-host, or onboard a new homelab service.
- Convert an app's Docker Compose into a managed Portainer stack.
- Add a service to the homelab GitHub repo.
- Put a service behind Nginx Proxy Manager or Authentik.
- Create an API key so Hermes can use a homelab service.

Do not use for:

- Throwaway container experiments that should not become managed homelab services.
- Pure application code changes unrelated to homelab deployment.
- Destructive migrations, data moves, or host-level changes unless explicitly scoped and confirmed.

## Non-Negotiable Safety Rules

1. **Revalidate before side effects.** Before SSH, sudo, Portainer stack creation/redeploy, NPM changes, Authentik changes, DNS changes, or API-key creation, restate the exact target and get confirmation.
2. **PRs are allowed; deploys require confirmation.** Justin allows homelab repo PRs without asking. Deployments and remote infra changes still need confirmation.
3. **No secrets in Git, chat, or logs.** Use placeholders in Compose and `.env.example`; store actual values in Portainer variables, `pass`, or a secrets manager.
4. **Use Git-backed Portainer stacks.** Do not deploy Portainer stacks from local Compose files unless Justin explicitly overrides this.
5. **Persistent state belongs on stable host paths.** Avoid repository-relative bind mounts for data. Portainer clones Git repos under internal paths such as `/data/compose/<stack-id>/`; that is not durable app state.
6. **Keep public repo diffs generic.** Do not commit private domains, LAN IPs, endpoint names, tokens, or host topology unless the repo is private and Justin explicitly approves.
7. **Prefer least privilege.** API keys for Hermes should be scoped/read-only when possible.

## Workflow Summary

1. Find the official Docker Compose or deployment documentation.
2. Inventory ports, volumes, environment variables, secrets, auth, and resource needs.
3. Recommend the target homelab machine with rationale and tradeoffs.
4. Add Compose, `.env.example`, and notes to the homelab GitHub repo.
5. Open a PR with validation notes and follow-up deployment instructions.
6. After confirmation, create the Portainer stack from the Git repo with Git updates enabled.
7. Optionally expose through Nginx Proxy Manager.
8. Optionally add Authentik native OIDC/SAML/OAuth or forward-auth.
9. Optionally create and store an API key for Hermes.
10. Verify health, mounts, env, access paths, and rollback notes.

## Step 1 — Find the Docker Compose Source

Start with upstream sources. Search official docs and the project's canonical GitHub/GitLab repository first. Prefer, in order:

1. Official `compose.yaml` / `docker-compose.yml`.
2. Official install docs with Compose snippets.
3. Official Docker run docs that can be converted to Compose.
4. Maintained examples from the project org.
5. Community examples only when upstream has no usable container guidance.

Capture these facts before writing Compose:

- Upstream source URL(s).
- Image names and tags.
- Required CPU architecture: amd64, arm64, or multi-arch.
- Published ports and internal container ports.
- Required and optional environment variables.
- Secrets and how they are generated.
- Volumes and persistence paths.
- Database/cache dependencies.
- Health/readiness endpoint.
- Built-in authentication model.
- Reverse proxy requirements: websockets, SSE, trusted proxy headers, base URL, upload limits.
- Resource profile: CPU, RAM, disk, GPU, database size, background workers.
- Backup/restore notes.

If upstream examples use `latest`, repository-relative state paths, or weak demo defaults, adapt them for a managed homelab deployment.

## Step 2 — Recommend the Target Machine

Advise on placement before deployment. Inspect current state when available, but do not SSH or alter remote hosts without confirmation.

Use this decision model:

- **Reverse-proxy / edge host:** lightweight networking utilities, services that benefit from proximity to NPM, VPN, or ingress.
- **App platform host:** general web apps, dashboards, and services that match existing application-hosting patterns.
- **NAS / storage-heavy host:** media, backups, document stores, databases with large persistent volumes, or services needing NAS-local bind mounts.
- **GPU / high-CPU host:** transcription, AI/ML, indexing, OCR, video processing, or other compute-heavy tasks.
- **Dedicated or isolated host:** sensitive services, high-risk internet-facing apps, or services with unusual network/storage requirements.

Check or ask about:

- Architecture compatibility.
- Port conflicts.
- Storage path and backup needs.
- Public vs LAN-only access.
- Data sensitivity.
- Dependency locality: database, NPM, Authentik, GPU, NAS paths.
- Existing host load and reliability expectations.

Give a concise recommendation:

```text
Recommended host: <machine or role>
Reason:
- <why this host fits>
- <why alternatives are weaker>
Tradeoffs:
- <main cost/risk>
Needs confirmation before deployment: yes
```

## Step 3 — Add the Service to the Homelab GitHub Repo

Follow the repository's existing structure. If there is no established convention, prefer:

```text
services/<service-name>/
  compose.yaml
  .env.example
  README.md
```

Use `services/` because it describes what the thing is, not which tool deploys it. Portainer is an implementation detail.

### Compose Guidelines

- Use `compose.yaml` unless the repo already standardizes on `docker-compose.yml`.
- Use stable absolute host paths for persistent data, parameterized through stack variables:

```yaml
services:
  app:
    image: vendor/service:1.2.3
    restart: unless-stopped
    volumes:
      - ${DATA_DIR:-/opt/stacks/service/data}:/app/data
```

- Avoid relative bind mounts for persistent state.
- Pin image tags when practical; avoid `latest` unless upstream only supports it or auto-update is desired and documented.
- Include `restart: unless-stopped` unless upstream says otherwise.
- Add a healthcheck when upstream provides a reliable endpoint or CLI probe.
- Keep networks minimal. Only add shared proxy networks when the repo already uses that pattern.
- Do not commit actual secrets.
- Prefer explicit environment variables over relying on Portainer interpolation magic.
- If Portainer stack variables supply values, be careful with `env_file:`; it may not behave like container runtime environment injection.

### `.env.example` Guidelines

Include all required variables with safe placeholders:

```env
DATA_DIR=/opt/stacks/service
SERVICE_BASE_URL=https://service.example.com
SERVICE_SECRET_KEY=change-me
DATABASE_PASSWORD=change-me
```

Add comments for generation commands when useful, but never include generated real values.

### README Guidelines

Include:

- What the service does.
- Upstream docs/Compose links.
- Recommended host and rationale.
- Required Portainer stack variables.
- Required persistent paths.
- Initial admin/setup steps.
- Optional NPM/Auth/API-key steps.
- Validation commands or endpoints.
- Backup notes when relevant.

### Validation Before PR

Run what applies:

```bash
docker compose -f services/<service>/compose.yaml config
git diff --check
```

Also inspect the diff for secrets and topology leaks:

```bash
git diff -- services/<service>/
```

If Docker is unavailable, at least YAML-parse the Compose file and explain the limitation in the PR.

## Step 4 — Open the Homelab Repo PR

Use the GitHub PR workflow. Branch names should be descriptive:

```text
feat/add-<service>-stack
```

PR body should include:

```markdown
## Summary
- Add <service> Compose stack under `services/<service>/`
- Add `.env.example` and deployment notes

## Upstream sources
- <official docs URL>
- <official compose URL, if any>

## Host recommendation
Recommended host: <host/role>
Reason:
- ...
Tradeoffs:
- ...

## Portainer deployment notes
- Stack name: <service>
- Compose path: `services/<service>/compose.yaml`
- Required stack variables:
  - `DATA_DIR`
  - `...`

## Optional follow-ups
- [ ] Nginx Proxy Manager hostname: <if requested>
- [ ] Authentik provider/forward-auth: <if requested>
- [ ] API key for Hermes: <if requested>

## Validation
- [ ] `docker compose -f services/<service>/compose.yaml config`
- [ ] `git diff --check`
- [ ] Diff reviewed for secrets/topology
```

After opening the PR, summarize the PR URL and the exact deployment steps that will follow after approval/merge.

## Step 5 — Deploy as a Portainer Git Stack

Only proceed after Justin confirms deployment scope.

Confirm:

- Target Portainer instance and endpoint/machine.
- Stack name.
- Homelab repo URL.
- Branch/ref, normally `main` after PR merge.
- Compose path, e.g. `services/<service>/compose.yaml`.
- Required Portainer stack variables.
- Whether Git updates / auto updates should be enabled.

Deployment requirements:

- Create the stack from the Git repository, not a pasted local Compose file.
- Enable Git updates when supported by the target Portainer setup.
- Supply non-secret config and secret values as stack variables/environment through Portainer.
- Keep persistent data on stable host paths such as `/opt/stacks/<service>` or the host's established equivalent.
- Do not print variable values that are secrets.

Verification after deploy:

- Stack is Git-backed and points to the expected repo/ref/path.
- Containers are running and healthy.
- Rendered mounts point to stable host paths.
- Required environment variables are present without exposing values.
- Local endpoint responds from the target network.
- Logs show no obvious boot errors.
- Data directory exists and has plausible ownership.

If deployment fails, explain the exact failing layer: Compose render, image pull, volume permissions, app boot, healthcheck, network, proxy, or auth.

## Optional Step — Nginx Proxy Manager Exposure

Only add NPM when public or friendly-hostname access is requested.

Clarify:

- LAN-only or public?
- Desired hostname.
- Upstream scheme and port.
- Websocket/SSE needs.
- Upload/body size limits.
- Whether Authentik should gate the service.

Implementation notes:

- Use an explicit DNS record; avoid wildcard split-DNS one-liners.
- Use the upstream target reachable from NPM.
- Enable SSL and force HTTPS for public services.
- Add websocket support if the app requires it.
- Scope changes to the target proxy host; avoid global NPM regeneration unless required.

Verify:

- Public and internal DNS resolve as expected.
- NPM proxy host points to the correct target.
- HTTPS certificate is valid.
- Browser access works.
- Logs do not show proxy loop, 502, or websocket errors.

## Optional Step — Authentik Login

Prefer service-native SSO when the app supports it. Use forward-auth when the service lacks suitable native auth or the frontend is not token-aware.

### Native OIDC/OAuth/SAML

Collect:

- Public base URL.
- Redirect/callback URI.
- Required scopes and claims.
- Group/role mapping needs.
- Admin bootstrap flow.

Verify:

- Authentik application/provider exists.
- Redirect URI exactly matches the public hostname and callback path.
- Login and logout work.
- User/group claims map correctly.

### NPM Forward-Auth

Use when the service cannot safely do native auth.

Verify:

- Forward-auth is applied only to the target service.
- The Authentik outpost/provider list is scoped correctly.
- Redirects use the public IdP hostname, not an internal service URL.
- Existing applications behind the same outpost still work.

## Optional Step — API Key for Hermes

Only create an API key when Justin asks or when a requested automation requires it.

Flow:

1. Identify API capabilities and the minimum permissions Hermes needs.
2. Prefer read-only or narrowly scoped tokens.
3. Create the token through the service UI/API after confirmation.
4. Store it in `pass` using Justin's conventions:

```bash
pass insert infra/<service>/api-token
# or, for MCP/tool credentials:
pass insert mcp/<service>/api-token
```

5. If a Hermes MCP/tool integration needs an env var, document the variable name and config path.
6. Verify with a minimal read-only API call while redacting token output.

Warnings:

- Do not paste API keys into chat.
- Do not commit API keys to Git or Compose.
- If the service only offers full-admin tokens, stop and ask Justin before storing or using one.

## Rollback / Cleanup Checklist

If the deployment is abandoned or fails publicly, inventory and remove only the target service resources:

- Portainer stack.
- Dedicated host data directory, if safe and confirmed.
- NPM proxy host.
- Authentik application/provider/outpost entry.
- Dedicated DNS record.
- Stored API token in `pass`, if created.
- Repo branch/PR, if no longer wanted.

Do not remove shared DNS, shared NPM config, shared Authentik outposts, or common networks without explicit confirmation.

## Common Pitfalls

1. **Deploying before the PR is merged.** Use the PR for review first unless Justin explicitly approves deploying a branch.
2. **Putting app data in Portainer's Git checkout.** `/data/compose/<stack-id>` is not a durable data home.
3. **Committing secrets in `.env` or README.** Use `.env.example` and Portainer/pass for real values.
4. **Hardcoding private topology in public Git.** Keep hostnames, LAN IPs, and machine names out of repo files unless explicitly allowed.
5. **Choosing the closest host instead of the right host.** Match workload to storage, CPU/GPU, network, architecture, and exposure requirements.
6. **Assuming upstream demo Compose is production-ready.** Fix persistence, secrets, tags, restart policy, and healthchecks.
7. **Mixing native auth and forward-auth blindly.** Choose one primary auth model and test redirects carefully.
8. **Letting NPM/Auth changes bleed across apps.** Scope proxy and provider changes to the target service.
9. **Printing secrets while verifying.** Redact values; verify presence and behavior, not raw content.
10. **Skipping rollback notes.** Public exposure and auth integrations need cleanup paths, not just container removal.

## Verification Checklist

- [ ] Upstream Compose/docs found and cited.
- [ ] Required images, ports, volumes, env vars, secrets, and health checks identified.
- [ ] Target machine recommended with rationale and tradeoffs.
- [ ] Homelab repo structure followed.
- [ ] Compose uses stable persistent paths and no committed secrets.
- [ ] `.env.example` documents required variables safely.
- [ ] README includes upstream links, host recommendation, stack variables, and deployment notes.
- [ ] Compose/YAML validation completed or limitation documented.
- [ ] Diff checked for secrets and topology leaks.
- [ ] PR opened with validation and Portainer deployment notes.
- [ ] Justin confirmed deployment scope before Portainer/SSH/NPM/Auth/API-key changes.
- [ ] Portainer stack is Git-backed with Git updates enabled.
- [ ] Stack variables are configured without leaking secrets.
- [ ] Service is healthy locally.
- [ ] NPM exposure works, if requested.
- [ ] Authentik login works, if requested.
- [ ] API key is least-privilege, stored in `pass`, and minimally tested, if requested.
- [ ] Rollback/cleanup path is documented.
