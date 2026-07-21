---
name: devopsellence
description: "Deploy apps with the devopsellence CLI — initialize solo or shared workspaces, manage nodes and secrets, deploy, and inspect status."
version: 1.0.0
author: devopsellence
license: MIT
homepage: https://www.devopsellence.com
metadata:
  hermes:
    tags: [devops, deployment, cli, infrastructure, ssh, docker, paas]
    category: devops
    requires_toolsets: [terminal]
---

# devopsellence

devopsellence is an agent-first deployment CLI for shipping applications, managing runtime nodes, storing deployment secrets, and inspecting deployment status from terminal workflows.

Use this skill when the user wants to deploy an app with devopsellence, initialize a devopsellence workspace, choose between solo and shared mode, register or create nodes, manage deployment secrets, inspect deployment status, or troubleshoot a devopsellence deployment.

All commands in this skill use the **terminal tool**.

## When to Use

- The user asks to deploy, redeploy, roll back, or inspect an application with devopsellence
- The user asks about `devopsellence.yml`, devopsellence workspace setup, or devopsellence modes
- The user wants an SSH-first/single-operator deployment workflow
- The user wants a hosted/team workflow with org/project/environment context
- The user wants to manage devopsellence secrets, nodes, logs, or status

## Prerequisites

The `devopsellence` CLI must be installed and available in `PATH`.

Check first:

```bash
command -v devopsellence && devopsellence --version
```

If it is not installed, do not guess or run arbitrary installer scripts. Point the user to the official devopsellence documentation or installation instructions they trust, then continue after the CLI is available.

For app deployments, the local app directory should normally be a git checkout with a clean or intentional working tree. Docker may be required for local image builds, depending on the deployment mode and command options.

## Workspace Modes

devopsellence supports two workspace modes. Do not silently choose a mode for a fresh workspace unless the user already made the mode obvious.

- **`solo`** — SSH-first, single-operator workflow for existing or provider-created VMs. Uses local workspace state, local secrets, and direct deployment coordination from the operator machine.
- **`shared`** — hosted/team workflow with sign-in, org/project/environment context, shared encrypted secrets, and control-plane-managed collaboration.

Before initializing, inspect any existing mode/context:

```bash
devopsellence mode show || true
devopsellence context show || true
```

If a mode already exists, use it. If no mode exists, recommend a mode and ask for confirmation before running `devopsellence init`.

Recommend **solo** when the user mentions their own VM/server, SSH, single-operator usage, local secrets, direct image streaming, or avoiding hosted/team workflows.

Recommend **shared** when the user mentions teams, org/project/environment context, browser sign-in, hosted control plane, shared encrypted secrets, managed nodes, auditability, or collaboration.

## Quick Reference

| Task | Command |
|------|---------|
| Show mode | `devopsellence mode show` |
| Show context | `devopsellence context show` |
| Initialize solo workspace | `devopsellence init --mode solo` |
| Initialize shared workspace | `devopsellence init --mode shared` |
| Check health/configuration | `devopsellence doctor` |
| Deploy current app | `devopsellence deploy` |
| Deploy existing image | `devopsellence deploy --image registry.example.com/app@sha256:...` |
| Inspect status | `devopsellence status` |
| List secrets | `devopsellence secret list` |
| Set secret from stdin | `printf '%s' "$VALUE" \| devopsellence secret set NAME --service web --stdin` |
| List nodes | `devopsellence node list` |
| View app logs | `devopsellence logs --lines 100` |
| Diagnose a node | `devopsellence node diagnose <node>` |

## Default Deployment Procedure

### 1. Move to the app directory

Work from the root of the app the user wants to deploy:

```bash
pwd
git status --short --branch || true
```

If the directory is not the intended app, stop and clarify before initializing or deploying.

### 2. Verify the CLI

```bash
command -v devopsellence && devopsellence --version
```

If the CLI is missing, stop and ask the user to install it from official devopsellence instructions.

### 3. Inspect existing workspace state

```bash
devopsellence mode show || true
devopsellence context show || true
```

If the workspace is already initialized, do not reinitialize unless the user asks.

### 4. Initialize if needed

Solo:

```bash
devopsellence init --mode solo
```

Shared:

```bash
devopsellence auth whoami || devopsellence auth login
devopsellence init --mode shared
```

When the shared workspace is known, prefer explicit context flags if supported by the installed CLI:

```bash
devopsellence init --mode shared --org <org> --project <project> --env <environment>
```

After init, review generated configuration before deploying:

```bash
git diff -- devopsellence.yml || true
```

### 5. Run preflight checks

```bash
devopsellence doctor
```

Resolve reported blockers before deploying. Common blockers include missing Docker, missing auth, missing node attachment, invalid config, DNS/ingress issues, or unreachable SSH hosts.

### 6. Deploy

Default deploy:

```bash
devopsellence deploy
```

Deploy an already-built image when appropriate:

```bash
devopsellence deploy --image registry.example.com/app@sha256:<digest>
```

### 7. Verify completion

```bash
devopsellence status
```

Treat CLI status as the primary control-plane/runtime signal, but verify important public endpoints directly when URLs are reported:

```bash
curl -fsS <public-url>/up
```

For web apps, prefer the app's real health path from `devopsellence.yml` over guessing.

## Secrets

Keep secret values out of chat, logs, command arguments, and shell history whenever possible. Prefer stdin:

```bash
printf '%s' "$VALUE" | devopsellence secret set NAME --service web --stdin
devopsellence secret list
devopsellence secret delete NAME --service web
```

If setting secrets for a non-default environment, pass the environment selector before any workload command separator and verify the command output targeted the intended environment:

```bash
printf '%s' "$VALUE" | devopsellence secret set NAME --service web --env staging --stdin
```

Never print secret values back to the user.

## Nodes

### Existing SSH node in solo mode

Use this when the user wants SSH-first deployment to a VM they already control:

```bash
devopsellence init --mode solo
devopsellence node create prod-1 --host <ip-or-hostname> --user <ssh-user> --ssh-key <path-to-private-key>
devopsellence agent install prod-1
devopsellence node attach prod-1
devopsellence doctor
devopsellence deploy
```

### Provider-created node

Creating cloud infrastructure can incur cost. Get explicit user confirmation before running provider-backed create commands unless the user already clearly authorized provisioning.

Typical flow:

```bash
printf '%s' "$PROVIDER_TOKEN" | devopsellence provider login <provider> --stdin
devopsellence node create prod-1 --provider <provider>
devopsellence node list
```

If a provider create fails with quota/capacity errors, do not retry blindly. Inspect existing nodes/resources and ask whether to reuse, remove, or request more quota.

## Output Contract

The devopsellence CLI is designed for agent-primary terminal workflows. Prefer structured stdout over scraping prose.

General rules:

- Successful bounded commands should emit one JSON document on stdout when structured output is supported by the installed CLI.
- Long-running commands may emit newline-delimited JSON events.
- Failure output may use structured envelopes with `ok: false` and an `error` object.
- When present, check `schema_version` before relying on command-specific fields.
- Prefer explaining failures from structured `error.code`, `error.message`, evidence fields, and suggested next actions.
- Treat stderr progress/diagnostics as supplemental unless diagnosing CLI/runtime failure.
- Tolerate unknown fields and avoid assuming undocumented fields are stable.

## Troubleshooting

### CLI missing

```bash
command -v devopsellence
```

If missing, ask the user to install the official CLI, then retry.

### Docker missing or not running

```bash
docker --version
docker info
```

If local builds are required, Docker must be available. If the user has a pushed image digest, consider `devopsellence deploy --image ...` instead.

### Not authenticated in shared mode

```bash
devopsellence auth whoami || devopsellence auth login
```

### SSH node unreachable

Check host, user, key path, and direct SSH connectivity before retrying devopsellence node operations:

```bash
ssh -i <path-to-private-key> <user>@<host> 'uname -a'
```

### Deployment not healthy

Collect status and logs without exposing secrets:

```bash
devopsellence status
devopsellence logs --lines 100
devopsellence node list
devopsellence node diagnose <node>
```

Use the app's configured healthcheck path and reported public URLs for endpoint checks.

## Safety Rules

- Do not initialize, deploy, provision paid infrastructure, delete nodes, delete secrets, or roll back production unless the user's request clearly authorizes that action.
- Ask before creating provider-backed nodes or making changes likely to incur cloud cost.
- Keep secret values out of command arguments, logs, PRs, and chat output.
- If the workspace has unexpected uncommitted changes, pause and summarize them before overwriting config or deploying.
- Prefer `devopsellence doctor` before deploy and `devopsellence status` after deploy.

## Verification Checklist

Before telling the user a deployment is complete, verify as much as applies:

- `devopsellence doctor` has no unresolved blockers
- `devopsellence deploy` finished successfully
- `devopsellence status` reports the intended environment/service/node as healthy or settled
- Reported public URLs respond on the app's health path
- Logs do not show fresh rollout, routing, or healthcheck failures
- The final response includes what changed, what was verified, and any remaining caveats
