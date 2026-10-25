---
name: huawei-apig
description: 'Use when creating or managing API Gateway (APIG). Covers API creation, throttling, auth (IAM/APP/basic), CORS, publishing, instance management. Triggers: APIG, API gateway, throttling, publish API. NOT for: FunctionGraph triggers (use huawei-functiongraph).'
version: 1
---

# Huawei Cloud APIG

**STOP - Do not answer from general knowledge.** Follow the procedure below.

## Overview

Domain expertise for Huawei Cloud API Gateway (APIG). Covers instance lifecycle, API group/API creation, publishing, and FunctionGraph trigger integration.

Always discover parameters with `hcloud APIG <Operation> --help` before executing.

## Critical Warnings

| Trap                              | Why                                                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| API group region-locked           | Cannot move across regions                                                                                                      |
| Throttling per-API default        | Use app-level quotas for per-user limits                                                                                        |
| CORS must be explicit             | OPTIONS preflight fails until configured                                                                                        |
| `BASIC` spec has no public IP     | Use `PROFESSIONAL` + `elb` provider for public access                                                                           |
| Instance creation takes 5-15min   | Long-running async operation. State is **Running** (NOT "SUCCESS"). Poll with `ListInstancesV2`, wait for `status == "Running"` |
| `sl_domain` is from API **Group** | NOT from Instance. Get it from `CreateApiGroupV2` or `ListApiGroupsV2` response                                                 |
| API name must NOT have hyphens    | `[a-zA-Z0-9_]+` only. Hyphens cause regex validation failure                                                                    |
| VPC params need prefix            | `--vpc.name=<n>` / `--subnet.vpc_id=<id>` / `--security_group.name=<n>` with KooCLI 7.x                                         |

## Instance Management

### Check Existing Instances

```bash
hcloud APIG ListInstancesV2 --cli-region=<r>
```

### Create Instance

```bash
hcloud APIG CreateInstanceV2 --help
```

Key gotchas when creating:

| Param                     | Note                                                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--spec_id`               | `BASIC` (no public access), `PROFESSIONAL` (requires `--loadbalancer_provider`)                                   |
| `--loadbalancer_provider` | `elb` for public access (supports `AddIngressEipV2`), `lvs` for internal only (supports `AddEipV2`)               |
| `--enterprise_project_id` | **Required** for enterprise accounts. Use `"0"` for default project                                               |
| `--available_zone_ids`    | Use AZ code like `ap-southeast-3a`, NOT UUID from `ListAvailableZonesV2`                                          |
| `--vpc_id`, `--subnet_id` | Must exist in the target region                                                                                   |
| `--security_group_id`     | **Required**. Create a security group first via `VPC CreateSecurityGroup` (VPC v3 API — no `vpc_id` param needed) |

### Add Public Access (ELB Provider Only)

`BASIC` instances have no public IP. Use `PROFESSIONAL` + `elb` provider for public access (`lvs` is internal-only).

```bash
hcloud APIG AddIngressEipV2 \
  --instance_id=<id> \
  --bandwidth_charging_mode=bandwidth \
  --bandwidth_size=<size>
```

To use an existing EIP instead of creating a new one:

```bash
hcloud APIG AddIngressEipV2 \
  --instance_id=<id> \
  --eip_id=<existing-eip-id>
```

> `AddIngressEipV2` only works with `elb` provider. `AddEipV2` (without "Ingress") requires `lvs` provider. Bandwidth minimum is 5 Mbps.

**Verification** — After binding, poll the instance to confirm EIP is active:

```bash
hcloud APIG ListInstancesV2 --cli-region=<region> --instance_id=<id> --cli-output=json | jq '.instances[0].eip_address'
```

Wait for `eip_address` to show a valid IP (may take up to 2 minutes). If `eip_address` is still `null` after 2 minutes, the EIP may not have been assigned.

> **Trap**: `sl_domain` from `CreateApiGroupV2` is an **internal-only** domain (e.g., `*.apic.cn-north-4.huaweicloudapis.com`). It may resolve to internal IP only (NXDOMAIN from public internet). **For public access, always use the `eip_address` from the instance**, not the `sl_domain` or trigger `invoke_url`.

## API Group

```bash
hcloud APIG CreateApiGroupV2 --instance_id=<id> --name=<n>
```

| Param      | Note                   |
| ---------- | ---------------------- |
| `--name`   | Group name (required)  |
| `--remark` | Description (optional) |

## API Management

```bash
hcloud APIG CreateApiV2 --help
```

## Publishing

```bash
hcloud APIG BatchPublishOrOfflineApiV2 \
  --instance_id=<id> \
  --action=online \
  --env_id=<env-id> \
  --apis.1=<api-id>
```

> The operation is `BatchPublishOrOfflineApiV2`, NOT `PublishApiV2`. The parameter is `apis.1` (1-based array), NOT `api_ids`. For multiple APIs, use `--apis.1`, `--apis.2`, etc.

## Throttling

```bash
hcloud APIG CreateThrottlingPolicyV2 --name=<n> --api_call_limits=1000
```

## Common Workflows

| Task             | Operation                                    |
| ---------------- | -------------------------------------------- |
| List instances   | `ListInstancesV2`                            |
| Create instance  | `CreateInstanceV2`                           |
| Delete instance  | `DeleteInstancesV2`                          |
| Add public EIP   | `AddIngressEipV2`                            |
| Create API group | `CreateApiGroupV2`                           |
| Create API       | `CreateApiV2`                                |
| Update API       | `UpdateApiV2` — change auth mode, path, etc. |
| List APIs        | `ListApisV2`                                 |
| Publish          | `BatchPublishOrOfflineApiV2`                 |
| List APIs        | `ListApisV2`                                 |

## FunctionGraph Integration

To expose a FunctionGraph function via HTTP, you need the complete chain:

```
APIG Instance → API Group → DEDICATEDGATEWAY Trigger → Publish
```

After the trigger is created on FunctionGraph side, publish the API in APIG:

```bash
hcloud APIG BatchPublishOrOfflineApiV2 \
  --instance_id=<apig-instance-id> \
  --action=online \
  --env_id=<env-id> \
  --apis.1=<api-id-from-trigger>
```

See `huawei-functiongraph` skill → `references/deploy-workflow.md` for the full end-to-end workflow.

## Cross-Skill References

- **VPC setup**: See `huawei-vpc` for VPC, subnet, security group creation
- **EIP setup**: See `huawei-vpc` for EIP creation and binding
- **FunctionGraph**: See `huawei-functiongraph` for DEDICATEDGATEWAY trigger creation

## References

- APIG Docs: https://support.huaweicloud.com/apig/
