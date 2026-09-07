---
name: huawei-gaussdb
description: 'Use when creating or managing GaussDB distributed database on Huawei Cloud. Covers GaussDB(for MySQL), GaussDB(for openGauss), sharding, HTAP. Triggers: GaussDB, distributed database, sharding, openGauss, HTAP. NOT for: single RDS (use huawei-rds).'
version: 1
---

# Huawei Cloud GaussDB

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud GaussDB <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for GaussDB. Covers instance lifecycle, databases, backups, configurations, and shard/readonly node management.

## Critical Warnings

| Trap                  | Why                                                          |
| --------------------- | ------------------------------------------------------------ |
| Shard key permanent   | Once set, shard key cannot be changed                        |
| Minimum nodes         | Distributed GaussDB requires at least 3 nodes for production |
| Engine version pinned | MySQL-compatible vs openGauss are separate products          |

## Common Workflows

| Task              | Operation                                                  |
| ----------------- | ---------------------------------------------------------- |
| List instances    | `ListInstances --cli-region=<r> --project_id=<p>`          |
| Show instance     | `ShowInstance --cli-region=<r> --project_id=<p>`           |
| List flavors      | `ListFlavors --cli-region=<r> --project_id=<p>`            |
| Create instance   | `CreateInstance --cli-region=<r> --project_id=<p>`         |
| Delete instance   | `DeleteInstance --cli-region=<r> --project_id=<p>`         |
| Create backup     | `CreateGaussMySqlBackup --cli-region=<r> --project_id=<p>` |
| Add readonly node | `AddReadonlyNode --cli-region=<r> --project_id=<p>`        |
| Add shard         | `AddShardingNode --cli-region=<r> --project_id=<p>`        |
| Manage database   | `AddDatabasePermission --cli-region=<r> --project_id=<p>`  |

Discover exact parameters with `--help` before executing any command.

## Troubleshooting

| Error                   | Fix                                               |
| ----------------------- | ------------------------------------------------- |
| Instance creation fails | Check VPC/subnet availability and flavor capacity |
| Connection refused      | SG missing database port                          |

## Security

- MUST use security groups for database access
- MUST enable SSL for connections

## Cross-Skill References

- **VPC/Subnet/Security Group**: See `huawei-vpc`
- **Backup/CBR**: See `huawei-cbr`
