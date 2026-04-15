---
name: ankra-cli
description: Manage Kubernetes clusters with the Ankra platform CLI. Covers auth, org management, cluster provisioning/deprovisioning/reconcile, add-ons, stacks, Helm, Hetzner nodes, credentials, and API tokens.
version: 1.0.0
author: Ankra
license: MIT
prerequisites:
  commands: [ankra]
metadata:
  hermes:
    tags: [Kubernetes, DevOps, Cloud, Cluster-Management, CLI, Hetzner, Helm, Infrastructure]
    related_skills: [webhook-subscriptions]
    requires_toolsets: [terminal]
---

# Ankra CLI Skill

Ankra is a Kubernetes cluster management platform. This skill enables Hermes to manage clusters, add-ons, stacks, Helm releases, credentials, and API tokens via the `ankra` CLI.

## Prerequisites

- `ankra` CLI installed (see https://ankra.io/docs/cli)
- Authenticated: `ankra login` or `ANKRA_API_TOKEN` environment variable set
- Config at `~/.ankra.yaml` (auto-created on first login)

## Configuration

```bash
# Env vars (prefer for automation)
export ANKRA_BASE_URL=https://api.ankra.io   # optional override
export ANKRA_API_TOKEN=***

# Config file
# ~/.ankra.yaml  —  written by `ankra login`
```

Global flags available on every command:
- `--base-url <url>` — Override API base URL
- `--token <token>` — Provide API token inline
- `--config <path>` — Alternate config file path

## Quick Reference

| Task | Command |
|------|---------|
| Login | `ankra login` |
| List clusters | `ankra cluster list` |
| Select cluster | `ankra cluster select <name>` |
| Cluster info | `ankra cluster info` |
| Provision cluster | `ankra cluster provision` |
| Deprovision cluster | `ankra cluster deprovision` |
| Reconcile cluster | `ankra cluster reconcile` |
| List add-ons | `ankra cluster addons list` |
| List stacks | `ankra cluster stacks list` |
| Helm releases | `ankra cluster helm releases` |
| List credentials | `ankra credentials list` |
| List tokens | `ankra tokens list` |
| AI chat | `ankra chat` |

## When to Use This Skill

Use this skill when the user wants to:
- Provision, deprovision, or inspect Kubernetes clusters via Ankra
- Manage add-ons installed on a cluster
- Create, delete, rename, clone, or roll back application stacks
- Manage Helm releases and chart registries
- Handle cloud provider credentials (Hetzner, OVH, UpCloud)
- Manage Ankra API tokens for CI/CD pipelines
- Troubleshoot cluster issues via `ankra chat`

## Procedure

### 1. Verify authentication

```bash
ankra cluster list
# If error: run `ankra login` or set ANKRA_API_TOKEN
```

### 2. Select the target organisation (if needed)

```bash
ankra org list
ankra org switch <org-name>
```

### 3. Select the target cluster

```bash
ankra cluster list
ankra cluster select <cluster-name>
ankra cluster info
```

### 4. Perform the requested operation

**Provision a new cluster (Hetzner)**
```bash
ankra cluster hetzner create
ankra cluster select <new-cluster-name>
ankra cluster agent status
```

**Manage add-ons**
```bash
ankra cluster addons available          # see what can be installed
ankra cluster addons list               # see what is installed
ankra cluster addons settings <addon>   # view current settings
ankra cluster addons update <addon>     # apply settings changes
ankra cluster addons uninstall <addon>  # remove an add-on
```

**Manage stacks**
```bash
ankra cluster stacks list
ankra cluster stacks create <name>
ankra cluster stacks clone <source> <dest>   # preferred for replication
ankra cluster stacks history <name>          # inspect change history
ankra cluster stacks delete <name>
```

**Manage Helm releases**
```bash
ankra cluster helm releases
ankra cluster helm uninstall <release>
ankra helm registries
ankra helm credentials
```

**Manage credentials**
```bash
ankra credentials list
ankra credentials hetzner --name <n> --token <t>
ankra credentials validate <name>
ankra credentials delete <name>
```

**Manage API tokens**
```bash
ankra tokens list
ankra tokens create --name <descriptive-name>
ankra tokens revoke <id>     # soft-delete (keeps audit trail)
ankra tokens delete <id>     # hard-delete
```

**Monitor operations**
```bash
ankra cluster operations list
ankra cluster operations cancel <operation-id>
ankra cluster manifests list
```

### 5. Reconcile and verify

```bash
ankra cluster reconcile
ankra cluster info
ankra cluster agent status
```

## Pitfalls

- **Always run `ankra cluster info` before any destructive command** — confirm you are targeting the right cluster.
- **Reconcile after config changes** — `ankra cluster reconcile` syncs desired state; skipping it may leave the cluster out of sync.
- **Agent must be healthy** — If commands time out or return unexpected errors, check `ankra cluster agent status` first.
- **Token rotation** — When rotating CI tokens, create the new token and update the secret before revoking the old one.
- **Credentials scope** — Cloud provider credentials (`ankra credentials hetzner/ovh/upcloud`) are org-scoped; ensure the right org is active before creating them.
- **ANKRA_API_TOKEN in CI** — Never hard-code tokens; always inject via environment variable.

## Verification

After operations, verify with:

```bash
ankra cluster info                  # Cluster status and metadata
ankra cluster agent status          # Agent connectivity
ankra cluster addons list           # Installed add-ons
ankra cluster stacks list           # Running stacks
ankra cluster operations list       # No stuck operations
```

## Troubleshooting

If you encounter unexpected behaviour:

1. Check agent: `ankra cluster agent status`
2. Check operations: `ankra cluster operations list`
3. Try reconcile: `ankra cluster reconcile`
4. Use AI assistant: `ankra chat`
5. Check cluster events: `ankra cluster info`
