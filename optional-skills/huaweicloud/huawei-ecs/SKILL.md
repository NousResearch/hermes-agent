---
name: huawei-ecs
description: 'Use when creating, configuring, managing, or troubleshooting ECS instances on Huawei Cloud. Covers instance creation (hcloud ECS CreateServers), flavor selection, image management, security groups, EIP binding, disk attachment, auto-scaling (AS), and troubleshooting. Triggers on: ECS, instance, flavor, image, security group, EIP, EVS, auto-scaling. NOT for: CCE container workloads (use huawei-cce), BMS bare metal servers.'
version: 1
---

# Huawei Cloud ECS

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud <Service> <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for Huawei Cloud Elastic Cloud Server (ECS). Covers instance lifecycle, flavor selection, image management, networking, storage attachment, auto-scaling, and troubleshooting.

## Prerequisites

Before creating an ECS instance from scratch, you MUST have:

- A VPC (see `huawei-vpc`)
- A subnet with DNS configured (see `huawei-vpc`)
- A security group with application ports open (see `huawei-vpc`)

If these do not exist, load the `huawei-vpc` skill first and create them before returning here.

## Critical Warnings

| Trap                          | Why                                                                         |
| ----------------------------- | --------------------------------------------------------------------------- |
| Flavor not in region          | Not all flavors available everywhere. Check with ECS ListFlavors first      |
| Security group denies all     | New SGs deny ALL inbound. Must explicitly add rules                         |
| EIP bills when idle           | Unattached EIP still incurs charges                                         |
| Stopped instance still bills  | Pay-per-use instances bill when stopped (unless shutdown-no-billing flavor) |
| Disk survives instance delete | Deleting instance does NOT delete system disk by default                    |

## Flavor Selection Guide

Flavor families are **region-dependent**. Always run `hcloud ECS ListFlavors --cli-region=<r>` to discover available flavors before recommending.

| Scenario                | Family                         | What to look for          |
| ----------------------- | ------------------------------ | ------------------------- |
| Web app / microservices | General-purpose (ac, s, sn, c) | 2-4 vCPU, 4-8 GB RAM      |
| Database / big data     | Memory-optimized (m, r)        | 4-8 vCPU, 16-64 GB RAM    |
| AI inference            | GPU (g, p)                     | 8+ vCPU, 64+ GB RAM + GPU |
| HPC                     | High-IO (h, ir, i)             | 8+ vCPU, local SSD        |

> See `references/flavors.md` for discovery workflow. **Do not hardcode flavor names** — availability changes by region and over time.

## Common Workflows

| Task                | Command                                                                                                                                                  | Steps                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| List flavors        | hcloud ECS ListFlavors --cli-region=<region>                                                                                                             | references/flavors.md                                                                    |
| Create instance     | hcloud ECS CreateServers --server.name=<n> --server.flavorRef=<id> --server.imageRef=<id> --server.nics.1.subnet_id=<id> --server.availability_zone=<az> | references/create-instance.md                                                            |
| Find by name        | See "How to search for instances" below                                                                                                                  |                                                                                          |
| Bind EIP            | hcloud EIP AssociatePublicips --publicip_id=<id> --publicip.associate_instance_id=<port-id> --publicip.associate_instance_type=PORT                      | Get port ID from `hcloud ECS ListServersDetails --server_id=<id>` → `OS-EXT-IPS:port_id` |
| Security group rule | hcloud VPC CreateSecurityGroupRule --security_group_id=<id> --direction=<direction> --protocol=<protocol>                                                | references/sg.md                                                                         |
| Attach disk         | hcloud EVS AttachVolume --volume_id=<id> --server_id=<id>                                                                                                | references/evs.md                                                                        |
| Delete instance     | hcloud ECS DeleteServers --servers.1.id=<id> --delete_publicip=true --delete_volume=true                                                                 | references/create-instance.md                                                            |
| Reboot instance     | hcloud ECS BatchRebootServers --reboot.servers.1.id=<id> --reboot.type=SOFT                                                                              | NOT `RebootServer` — that operation does not exist                                       |

## How to Search for Instances

`ListServersDetails` supports `--name` for exact match only. For fuzzy search:

1. List all instances: `hcloud ECS ListServersDetails --cli-region=<region> --limit=100`
2. Filter client-side by name substring, tag, or status
3. Use `--server_tags` filter if instances are tagged: `hcloud ECS ListServersDetails --server_tags.1.key=Project`

> **Return structure**: `ListServersDetails` returns `{"servers": [...]}` (array), NOT `{"server": ...}`. Parse as `.servers[0].status`, NOT `.server.status`.

Abort if the result set is larger than `--limit` and ask the user to narrow the search.

## Instance Status Polling

ECS creation is asynchronous. Status transitions: `BUILD` → `ACTIVE` (or `ERROR`). Wait times vary widely (20s–3min), never use fixed sleep.

Poll strategy:

```bash
for i in $(seq 1 30); do
  status=$(hcloud ECS ListServersDetails --cli-region=<region> --server_id=<id> --cli-output=json | jq -r '.servers[0].status')
  if [ "$status" = "ACTIVE" ]; then break; fi
  if [ "$status" = "ERROR" ]; then echo "Creation failed"; exit 1; fi
  sleep 10
done
```

- Poll interval: 10 seconds
- Maximum wait: 5 minutes (30 iterations)
- Check for `ERROR` status to detect creation failures early

## SSH Connection Verification

After ECS is ACTIVE and EIP is bound, verify SSH connectivity:

```bash
hcloud ECS ListServersDetails --cli-region=<region> --server_id=<id>
# → addresses.<vpc-id>[].OS-EXT-IPS:addr

ssh -o StrictHostKeyChecking=accept-new -i <path-to-private-key> root@<eip-address>
```

> **SSH verification checklist**: EIP bound → security group has port 22 ingress → keypair private key saved locally → known_hosts handled with `StrictHostKeyChecking=accept-new` (or delete stale entries with `ssh-keygen -R <ip>`).

### Running Commands Inside the Instance

Use SSH for any in-instance operations (install software, check logs, start services):

```bash
ssh -o StrictHostKeyChecking=accept-new -i <key> root@<eip> 'command'
```

Example — re-run failed cloud-init setup manually:

```bash
ssh -o StrictHostKeyChecking=accept-new -i <key> root@<eip> 'dnf install -y nginx && systemctl enable --now nginx'
```

> If SCP blocks SSH, use cloud-init `--server.user_data` for full deployment instead. See `references/create-instance.md` §Bootstrap.

## Deleting Instances

- Show the user the exact command and get explicit approval before running
- By default, `--delete_publicip` and `--delete_volume` are **false** — public IP and system disk survive deletion
- Set both to `true` to avoid orphaned resources and unexpected billing
- Data disks ARE deleted by default (unlike system disk)

## Troubleshooting

| Error                          | Root Cause -> Fix                                                                                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cannot SSH                     | SG missing port 22 or no EIP -> Add ingress rule / Bind EIP                                                                                                                     |
| Flavor unavailable             | Region limitation -> ListFlavors in target region                                                                                                                               |
| Insufficient resources         | Stock depleted -> Change flavor or AZ                                                                                                                                           |
| AuthFailure                    | Expired AK/SK -> re-run `npx huaweicloud-devkit auth init`                                                                                                                      |
| APIGW.0802 / region permission | IAM user has no access to this region -> IAM console → User → Permissions → add region, or switch to another region                                                             |
| Cannot SSH (port 22 open)      | SCP policy may be blocking SSH. Check `SYS.0403` errors in command output -> Use cloud-init/user_data for initial setup instead. See `references/create-instance.md` §Bootstrap |

## Security Considerations

- MUST use security groups, not iptables
- MUST store SSH keys in DEW, never in user-data
- SHOULD enable CTS audit logging
- MUST NOT open 0.0.0.0/0 for SSH

## MCP Tools

Prefer these tools over raw hcloud CLI — they enforce safety policies:

- huaweicloud_list_operations service=ECS
- huaweicloud_run_readonly_command for discovery (auto-redacts output)
- huaweicloud_plan_cli_command for command planning (returns command text + safety classification)
- huaweicloud_run_approved_command for writes (requires exact command approval)
- huaweicloud_check_cli to verify hcloud is available

> **approvedCommand trap**: `huaweicloud_run_approved_command` validates that `approvedCommand` matches the planned command EXACTLY (including `<redacted>` placeholders). Always use the `command` field value returned by `huaweicloud_plan_cli_command` verbatim — never reconstruct or retype it. Mismatches cause rejection with "approvedCommand must exactly match the planned hcloud command."

## Without MCP (Fallback)

If MCP tools are NOT available (new install, session not restarted):

- Raw hcloud commands WILL appear in shell history — passwords and secrets are at risk
- Always use key_name instead of adminPass
- Verify safety manually: no secret value reads, no credential file access
- Restart session as soon as possible to enable safety policies

## Without MCP

Fall back to hcloud CLI. State: "MCP unavailable, using local hcloud CLI."

## Flexus (Lightweight ECS)

Flexus is the lightweight ECS family. Two variants:

| Variant      | API                        |                 CLI                 | Billing             |
| ------------ | -------------------------- | :---------------------------------: | ------------------- |
| **Flexus L** | HCSS (hcss:lightInstances) |   ❌ No hcloud — Python SDK only    | Prepaid only        |
| **Flexus X** | Standard ECS API           | ✅ `hcloud ECS` with `x1.*` flavors | On-demand & prepaid |

For Flexus X, use standard ECS CreateServers flow with `x1.*` flavors. For Flexus L, manual console provisioning is recommended — no KooCLI path exists.

## References

- ECS Docs: https://support.huaweicloud.com/ecs/
- Flavor specs: references/flavors.md
- Create instance: references/create-instance.md
