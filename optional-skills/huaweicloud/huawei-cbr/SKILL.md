---
name: huawei-cbr
description: 'Use when creating or managing Cloud Backup and Recovery (CBR) vaults, backups, and restore operations on Huawei Cloud. Triggers: CBR, backup, restore, vault, snapshot, disaster recovery. NOT for: OBS object versioning (use huawei-obs), RDS automated backups (use huawei-rds).'
version: 1
---

# Huawei Cloud CBR

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud CBR <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for Cloud Backup and Recovery. Covers vaults, backups, restore operations, policies, and cross-region replication.

## Critical Warnings

| Trap                             | Why                                                              |
| -------------------------------- | ---------------------------------------------------------------- |
| Vault binds single resource type | Server vault != disk vault != file system vault                  |
| Restore needs instance stop      | Most VM restore requires the target instance to be stopped first |
| Backup storage incurs charges    | Retention policies affect storage costs                          |

## Common Workflows

| Task                       | Operation                                                    |
| -------------------------- | ------------------------------------------------------------ |
| List vaults                | `ListVaults --cli-region=<r> --project_id=<p>`               |
| Create vault               | `CreateVault --cli-region=<r> --project_id=<p>`              |
| Create backup policy       | `CreatePolicy --cli-region=<r> --project_id=<p>`             |
| Create checkpoint          | `CreateCheckpoint --cli-region=<r> --project_id=<p>`         |
| Add resource to vault      | `AddVaultResource --cli-region=<r> --project_id=<p>`         |
| Batch update vault         | `BatchUpdateVault --cli-region=<r> --project_id=<p>`         |
| Create organization policy | `CreateOrganizationPolicy --cli-region=<r> --project_id=<p>` |

Discover exact parameters with `--help` before executing any command.

## Troubleshooting

| Error                 | Fix                                                |
| --------------------- | -------------------------------------------------- |
| Backup creation fails | Check instance is running and agent is online      |
| Restore fails         | Ensure target instance is stopped before restore   |
| Vault full            | Increase vault capacity or adjust retention policy |

## Cross-Skill References

- **ECS**: See `huawei-ecs` for backing up instances
- **EVS**: See `huawei-ecs` references/evs.md for disk backup
