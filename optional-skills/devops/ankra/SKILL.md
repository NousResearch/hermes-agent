---
name: ankra
description: "Manage Ankra Kubernetes clusters via its MCP or CLI."
version: 2.0.0
author: "Mark Shine (@CodeStaple), Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
requires: "Ankra account; MCP token (mcp:read+) or ankra CLI login"
metadata:
  hermes:
    tags: [kubernetes, devops, cloud, cluster-management, mcp, hetzner, helm, infrastructure]
    category: devops
    related_skills: []
---

# Ankra

[Ankra](https://ankra.io) is a Kubernetes cluster management platform. It offers
two surfaces, and this skill covers both:

- **MCP server** (`https://platform.ankra.app/api/v1/mcp`) — the preferred
  agent surface. Scope-gated, org-isolated, and audit-logged. Best for
  interactive, permissioned inspection and changes.
- **`ankra` CLI** — covers platform operations the MCP surface does not expose
  (deprovisioning, credential and token management, node groups, kubeconfig
  export). Best for scripting and CI/CD.

Prefer MCP when a token is configured; use the CLI for the operations listed
under Surface 2.

## When to Use

Use when the user wants to:

- Inspect clusters, workloads, logs, events, cost, or security reports
- Provision a new cluster, or install/sync/roll back add-ons
- Work with stacks (inspect, apply draft changes, delete)
- Apply, patch, scale, restart, or delete Kubernetes resources
- Query Prometheus, or drive GitHub CI/CD flows
- Manage credentials, tokens, or deprovision clusters (CLI)

Do not use for: in-cluster operations on a cluster Ankra does not manage (use
`kubectl` via `terminal`).

---

# Surface 1: MCP server (preferred)

## Prerequisites

### One-time: create an MCP token

1. Open the [Ankra portal](https://platform.ankra.app), go to
   **Profile → API Tokens**, and select **Add Token**.
2. Name it (e.g. `hermes-mcp`) and choose `mcp:read`. Add `mcp:write` only if
   Hermes should apply changes. Ankra shows **no** server-side confirmation
   before a write tool runs — only grant write to clients you trust.
3. Copy the token immediately — Ankra shows it once.

The token is bound to the organisation chosen at creation and cannot call
Ankra's REST API or cross organisations.

### One-time: connect Hermes

```bash
export ANKRA_MCP_TOKEN=<token>
hermes mcp install ankra \
  --url https://platform.ankra.app/api/v1/mcp \
  --header "Authorization: Bearer $ANKRA_MCP_TOKEN"
```

**OAuth alternative** — Claude and other clients that support remote connectors
can add `https://platform.ankra.app/api/v1/mcp` as a custom connector and
complete Ankra's OAuth 2.1 PKCE flow (sign in → choose org → approve scope).
Ankra issues a seven-day MCP-only token; reconnect when it expires (no refresh
token is issued).

Confirm the connection:

```bash
hermes mcp list          # ankra  connected  <N> tools
```

Tools register under the `mcp_ankra_*` prefix (Hermes namespaces MCP tools by
server name). On the wire and in `tools/list` they use the bare names below.
If tools are absent, verify the token and run `hermes mcp reconnect ankra`.

## Scope gate

`tools/list` returns only what your token permits:

| Scope | Reaches |
|---|---|
| `mcp:read` | Read-only tools **plus** ask-mode "safe creations" (ephemeral workspace pods, repo clone/search, throwaway PR demos) |
| `mcp:write` | The read surface **plus** all mutating tools |

`mcp:write` implies read. Two structural rules from the server:

- **Draft-building tools are not exposed over MCP at all** — stack drafts are a
  chat-lane concept. Over MCP you inspect stacks and `apply_stack_changes`, but
  you do not assemble drafts step by step.
- **`save_memory` is write-only** even though it reads as read-only — content it
  stores is injected into future prompts, so a read token cannot reach it.

Cluster-scoped tools require a `cluster_id`; the server injects it into the
schema and rejects any cluster outside the token's organisation. Resolve IDs
with `list_clusters` first.

## Tools

The read surface is large (100+ tools); the write surface is ~20. The tables
below are the load-bearing subset — call `tools/list` for the authoritative
per-token catalogue. Names are exact.

### Read (mcp:read)

| Tool | Purpose |
|---|---|
| `list_clusters` | Clusters in the org with state, environment, K8s version |
| `get_cluster_details` | Detailed info for one cluster by name or ID |
| `get_cluster_status` | Cluster health overview |
| `get_cluster_cost` / `get_stack_cost` | Cost breakdowns |
| `get_security_reports` | Cluster security findings |
| `list_recent_executions` | Recent deployments / write operations |
| `get_execution_details` | Steps, statuses, and errors for one execution |
| `get_pods` / `get_pod_logs` | Pod status and logs |
| `get_deployments` / `get_services` / `get_events` / `get_nodes` | Core K8s reads |
| `get_secrets` | Secret **metadata** (values are redacted) |
| `describe_resource` | Full YAML/JSON for a resource |
| `list_addons` / `get_addon_details` / `get_addon_history` | Add-on state |
| `list_stacks` / `get_stack_details` | Stack inventory and detail |
| `list_available_charts` / `get_chart_schema` | Helm chart catalogue |
| `list_provider_credentials` / `list_instance_types` | Provisioning inputs |
| `query_prometheus` / `query_prometheus_range` | PromQL queries |
| `github_list_workflow_runs` / `github_get_job_logs` | CI inspection |

### Write (mcp:write) — every entry requires confirmation

| Tool | Risk | Purpose |
|---|---|---|
| `restart_deployment` | low | Rolling restart of a deployment |
| `scale_deployment` / `scale_statefulset` | low/med | Change replica count |
| `addon_install` | med | Install an add-on into a stack |
| `sync_addon` | med | Apply pending add-on changes from source |
| `patch_resource` / `delete_pod` | med | Patch a resource / delete a pod |
| `apply_manifest` | high | Create/update K8s resources from YAML |
| `delete_resource` | high | Delete any K8s resource (incl. CRDs) |
| `secret_rotation` | high | Rotate a Secret and restart referencers |
| `rollback_addon` / `helm_release_rollback` | high | Roll back to a prior version |
| `apply_stack_changes` | high | Apply the current stack draft to the cluster |
| `delete_stack` | high | Delete a stack and all its addons/manifests |
| `uninstall_addon_from_stack` | high | Mark an add-on for uninstallation |
| `create_hetzner_cluster` | high | Provision a new cluster (also `_ovh_`, `_upcloud_`, `_digitalocean_`) |
| `add_helm_registry` | med | Add a chart registry to the org catalogue |
| `github_commit_files` / `github_create_pull_request` | med | Write to GitHub |

### Not available over MCP — use the CLI

Deprovisioning a cluster, cancelling an operation, and **all credential and API
token management** have no MCP tool. Do them via the `ankra` CLI (Surface 2).

## Operating Discipline (MCP)

1. **Start with `list_clusters`.** Confirm the target exists; capture its
   `cluster_id`. Never hard-code it.
2. **Read before you write.** Inspect state (`get_cluster_status`, `list_addons`,
   `get_stack_details`) before any mutating call. Ankra runs no second
   confirmation server-side — the MCP client owns that, so you are the guard
   rail.
3. **Tell the user what will change before any write**, especially the `high`
   risk tools (`apply_manifest`, `delete_resource`, `delete_stack`,
   `create_*_cluster`, `secret_rotation`). There is no server-side undo.
4. **After a write, confirm via `list_recent_executions` / `get_execution_details`**
   before the next change on the same cluster.
5. **Never pass literal secrets as arguments** — the server refuses a mutating
   call whose parameter contains one. Reference existing Secrets instead.
6. **When anything misbehaves, check `get_cluster_status` first** — an offline
   agent cascades failures across every cluster-scoped tool.

Calls are rate-limited per token (writes also hit a fail-closed platform write
limit); wait and retry on a limit. Every resolved call — including refused ones
— is written to the org audit log.

## Pitfalls (MCP)

| Symptom | Cause | Fix |
|---|---|---|
| `401 Invalid or missing bearer token` | Token missing, expired, or revoked | Recreate token; `hermes mcp configure ankra` |
| `403 Token has no MCP scope` | Token is REST-only | Recreate with `mcp:read` or `mcp:write` |
| `403 Origin not allowed` | Browser client on a disallowed origin | Ask the Ankra admin to review allowed origins |
| A tool reports it needs `mcp:write` | Read-only token hit a mutating tool | Confirm with user; recreate token with write |
| Expected tool missing from `tools/list` | Read scope, or it's a draft/token/deprovision tool | Check the scope gate; use the CLI for token/credential/deprovision ops |
| A tool reports a rate limit | Too many calls/min | Wait 60 s; reduce polling |

---

# Surface 2: `ankra` CLI

Use the CLI for what MCP does not expose — deprovisioning, credentials, API
tokens, node groups, kubeconfig, version selection — and for scripting/CI.

## Prerequisites

- `ankra` CLI installed — see https://ankra.io/docs/cli
- Authenticated: `ankra login`, or set `ANKRA_API_TOKEN`
- Config at `~/.ankra.yaml` (auto-created on first login)

Global flags on every command: `--base-url <url>`, `--token <token>`,
`--config <path>`. Run CLI commands through the `terminal` tool.

## Quick Reference

| Task | Command |
|---|---|
| Login | `ankra login` |
| List / switch orgs | `ankra org list` · `ankra org switch <org>` |
| List clusters | `ankra cluster list` |
| Select cluster | `ankra cluster select <name>` |
| Cluster info | `ankra cluster info` |
| Provision (Hetzner) | `ankra hetzner cluster create` |
| Deprovision | `ankra cluster deprovision` |
| Kubeconfig | `ankra cluster kubeconfig` |
| Agent status | `ankra cluster agent status` |
| Add-ons | `ankra cluster addons list` |
| Stacks | `ankra cluster stacks list` |
| Helm releases | `ankra cluster helm releases` |
| Credentials | `ankra credentials list` |
| Tokens | `ankra tokens list` |
| Operations | `ankra cluster operations list` |
| AI chat | `ankra chat` |

## Command Groups

**Clusters**
```bash
ankra cluster list
ankra cluster select <name>
ankra cluster info
ankra cluster deprovision                 # MCP has no deprovision tool
ankra cluster kubeconfig
ankra cluster agent status
ankra cluster operations list
ankra cluster operations cancel <operation-id>
```

**Add-ons and stacks** (also available over MCP)
```bash
ankra cluster addons available
ankra cluster addons list
ankra cluster addons settings <addon>
ankra cluster addons update <addon>
ankra cluster addons uninstall <addon>
ankra cluster stacks list
ankra cluster stacks clone <source> <dest>
ankra cluster stacks history <name>
ankra cluster stacks delete <name>
```

**Credentials** — org-scoped; **not on the MCP surface**
```bash
ankra credentials list
ankra credentials hetzner --name <n> --token <t>
ankra credentials validate <name>
ankra credentials delete <name>
```

**API tokens** — **not on the MCP surface**
```bash
ankra tokens list
ankra tokens create --name <descriptive-name>
ankra tokens revoke <id>    # soft-delete, keeps audit trail
ankra tokens delete <id>    # hard-delete
```

### Token rotation for CI/CD (CLI only)

MCP exposes no token tools, so rotation is always a CLI flow:

```bash
ankra tokens create --name ci-prod-deploy-v2   # 1. create replacement
# 2. update the secret in your CI system
ankra tokens revoke <old-id>                    # 3. only then revoke the old
```

Update the CI secret **before** revoking the old token — revoking first breaks
pipelines with no safe rollback.

## Pitfalls (CLI)

- **Confirm the target before destructive commands** — `ankra cluster info`
  proves which cluster is selected.
- **Agent must be healthy** — timeouts or odd errors → `ankra cluster agent
  status` first.
- **Never hard-code `ANKRA_API_TOKEN`** in CI — inject it via the environment.

---

## Verification Checklist

- [ ] Connection healthy — `list_clusters` (MCP) or `ankra cluster list` (CLI)
      returns clusters
- [ ] `cluster_id` / selected cluster resolved before any cluster-scoped action
- [ ] Current state read before any write / destructive command
- [ ] User informed of what will change before any mutating / high-risk call
- [ ] Write confirmed via `list_recent_executions` before the next change
- [ ] No literal secrets passed as tool arguments
- [ ] Token rotation done via CLI: new token active in CI before old is revoked
