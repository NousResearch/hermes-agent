---
name: huawei-dds-dcs
description: 'Use when creating or managing Document Database Service (DDS/MongoDB-compatible) or Distributed Cache Service (DCS/Redis/Memcached) on Huawei Cloud. Triggers: DDS, DCS, MongoDB, Redis, Memcached, document DB, cache, replica set, sharding. NOT for: RDS (use huawei-rds), GaussDB (use huawei-gaussdb).'
version: 1
---

# Huawei Cloud DDS / DCS

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud DDS <Operation> --help` or `hcloud DCS <Operation> --help` before constructing commands.

## DDS (Document Database Service — MongoDB Compatible)

## DDS Critical Warnings

| Trap                     | Why                                         |
| ------------------------ | ------------------------------------------- |
| Requires 3-node replica  | Minimum production deployment needs 3 nodes |
| Shard key permanent      | Once selected, shard key cannot be changed  |
| Storage auto-scaling off | Enable before storage fills                 |

### Common Workflows

| Task              | Operation                                           |
| ----------------- | --------------------------------------------------- |
| List instances    | `ListInstances --cli-region=<r> --project_id=<p>`   |
| Create instance   | `CreateInstance --cli-region=<r> --project_id=<p>`  |
| Delete instance   | `DeleteInstance --cli-region=<r> --project_id=<p>`  |
| Add readonly node | `AddReadonlyNode --cli-region=<r> --project_id=<p>` |
| Add shard         | `AddShardingNode --cli-region=<r> --project_id=<p>` |
| Create backup     | `CreateBackup --cli-region=<r> --project_id=<p>`    |

Discover exact parameters with `--help` before executing any command.

## DCS (Distributed Cache Service — Redis/Memcached)

## DCS Critical Warnings

| Trap                    | Why                                                |
| ----------------------- | -------------------------------------------------- |
| Redis password required | Cannot create Redis instance without password      |
| Memcached no password   | Memcached has no auth — only accessible within VPC |
| Cache persistence costs | AOF/RDB persistence uses additional storage        |

### Common Workflows

| Task                   | Operation                                                     |
| ---------------------- | ------------------------------------------------------------- |
| List instances         | `ListInstances --cli-region=<r> --project_id=<p>`             |
| Create Redis instance  | `CreateInstance --cli-region=<r> --project_id=<p>`            |
| Delete instance        | `DeleteInstances --cli-region=<r> --project_id=<p>`           |
| Restart instance       | `RestartInstance --cli-region=<r> --project_id=<p>`           |
| Show node information  | `BatchShowNodesInformation --cli-region=<r> --project_id=<p>` |
| Create custom template | `CreateCustomTemplate --cli-region=<r> --project_id=<p>`      |

Discover exact parameters with `--help` before executing any command.

## Troubleshooting

| Error                       | Fix                                                                 |
| --------------------------- | ------------------------------------------------------------------- |
| DDS instance creation fails | Check 3-node minimum, VPC/subnet availability                       |
| DCS connection refused      | Redis instances need password. Memcached only accessible within VPC |
| DCS auto-expire not working | Enable auto-expire scan task for Redis                              |

## Cross-Skill References

- **VPC/Subnet**: See `huawei-vpc` for instance networking
- **Backup/CBR**: See `huawei-cbr` for backup management
