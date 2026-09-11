---
name: huawei-rds
description: 'Use when creating, configuring, managing, or troubleshooting RDS instances on Huawei Cloud. Covers MySQL, PostgreSQL, SQL Server. Triggers: RDS, MySQL, PostgreSQL, database instance, backup, read replica. NOT for: GaussDB (use huawei-gaussdb), DDS (use huawei-dds-dcs).'
version: 1
---

# Huawei Cloud RDS

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud RDS <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Prerequisites

Before creating an RDS instance, you MUST have:

- A VPC and subnet (see `huawei-vpc`)
- A security group with database port open (MySQL=3306, PostgreSQL=5432, SQL Server=1433)
- Run `hcloud RDS ListFlavors --database_name=<engine> --cli-region=<r>` to get spec codes

## KooCLI Command Format

```
hcloud RDS <Operation> --cli-region=<region> [--key=value ...]
```

| Rule           | Detail                                                                            |
| -------------- | --------------------------------------------------------------------------------- |
| Service name   | `RDS` (uppercase)                                                                 |
| Operation      | PascalCase: `ListInstances`, `CreateManualBackup`                                 |
| Params         | `--key=value` format. JSON params: `--key='{"k":"v"}'`                            |
| Array params   | 1-based: `--instance_ids.1=xxx`                                                   |
| Password param | Conflicts with KooCLI system param; use `--cli-jsonInput` (see Critical Warnings) |

## Critical Warnings

| Trap                                | Why                                                                                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Engine version immutable            | Cannot change MySQL to PostgreSQL in-place                                                                                                   |
| Automated backups use OBS           | Backup storage incurs separate charges. Set retention period explicitly                                                                      |
| Storage auto-scaling off by default | Enable before storage runs out or instance goes read-only                                                                                    |
| `--password` conflicts with KooCLI  | Use `--cli-jsonInput=<file>` with JSON file (see `--cli-jsonInput` section below). The `printf "b\n"` workaround is broken in KooCLI 7.2.12+ |
| Volume type must match flavor       | General→CLOUDSSD; Dedicated→CLOUDSSD\|ESSD; ARM→ULTRAHIGH                                                                                    |
| Flavor not in region                | Always `ListFlavors` first. Spec codes vary by region                                                                                        |
| `database_name` is case-sensitive   | Use `MySQL` / `PostgreSQL` / `SQLServer` / `MariaDB` — NOT lower-case `mysql`                                                                |
| Instance creation takes 3–8 min     | Status: BUILD→MODIFYING→ACTIVE. Poll every 15s: `hcloud RDS ListInstances --cli-region=<r> --instance_id=<id> \| jq '.instances[0].status'`  |
| Body `--region` is required         | CreateInstance body requires `--region=<r>` (same as `--cli-region`) or `DBS.280243`                                                         |
| Restore creates new instance        | No in-place restore. Verify target flavor before restoring                                                                                   |

## Instance Management

### List

```bash
hcloud RDS ListInstances --cli-region=<r>
hcloud RDS ListFlavors --database_name=<engine> --cli-region=<r>
hcloud RDS ListDatastores --database_name=<engine> --cli-region=<r>
hcloud RDS ListEngineFlavors --instance_id=<id> --cli-region=<r>
```

### Create Instance

```bash
hcloud RDS CreateInstance --cli-region=<r> \
  --region=<r> \
  --name=<name> \
  --datastore.type=<engine> \
  --datastore.version=<version> \
  --flavor_ref=<flavor-id> \
  --volume.type=<vol-type> \
  --volume.size=<size> \
  --vpc_id=<vpc-id> \
  --subnet_id=<subnet-id> \
  --security_group_id=<sg-id> \
  --availability_zone=<az> \
  --charge_info.charge_mode=<mode>
```

> `--password` conflicts with a KooCLI system param. Use `--cli-jsonInput` (see below). The `printf "b\n"` pipe workaround is broken in KooCLI 7.2.12+.

### `--cli-jsonInput` for CreateInstance

```json
{
  "path": {
    "project_id": "<project-id>"
  },
  "body": {
    "name": "<name>",
    "region": "<region>",
    "datastore": { "type": "<engine>", "version": "<version>" },
    "ha": { "mode": "single", "replication_mode": "semisync" },
    "flavor_ref": "<flavor-id>",
    "volume": { "type": "<vol-type>", "size": <gb> },
    "vpc_id": "<vpc-id>",
    "subnet_id": "<subnet-id>",
    "security_group_id": "<sg-id>",
    "availability_zone": "<az>",
    "password": "<pw>",
    "charge_info": { "charge_mode": "postPaid" }
  }
}
```

> Save as `rds-create.json` then: `hcloud RDS CreateInstance --cli-jsonInput=rds-create.json`. For `project_id`, run `hcloud IAM KeystoneListProjects`.

| Param                 | Required | Note                                                                                         |
| --------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `--name`              | Yes      | Instance name                                                                                |
| `--datastore`         | Yes      | `--datastore.type=<engine> --datastore.version=<version>`                                    |
| `--flavor_ref`        | Yes      | From `ListFlavors` output                                                                    |
| `--volume`            | Yes      | Type matching: General→CLOUDSSD, Dedicated→CLOUDSSD\|ESSD, ARM→ULTRAHIGH                     |
| `--vpc_id`            | Yes      | Must exist in target region                                                                  |
| `--subnet_id`         | Yes      | Must exist in target region                                                                  |
| `--security_group_id` | Yes      | Must have DB port open                                                                       |
| `--availability_zone` | Yes      | Use AZ code from `NovaListAvailabilityZones`                                                 |
| `--charge_info`       | No       | `--charge_info.charge_mode=<mode>`. Default postPaid (pay-per-use). Run `--help` for options |
| `--password`          | Yes      | 8-32 chars, uppercase+lowercase+digit+special                                                |
| `--port`              | No       | Default 3306 (MySQL) / 5432 (PG) / 1433 (SQL Server)                                         |

### Modify / Resize

```bash
hcloud RDS StartInstanceRestartAction --instance_id=<id> --cli-region=<r>
hcloud RDS StartResizeFlavorAction --instance_id=<id> --flavor_ref=<new-id> --cli-region=<r>
hcloud RDS StartInstanceEnlargeVolumeAction --instance_id=<id> --volume_size=<gb> --cli-region=<r>
hcloud RDS StartFailover --instance_id=<id> --cli-region=<r>
hcloud RDS UpdateInstanceAlias --instance_id=<id> --alias=<new-name> --cli-region=<r>
```

### Delete

```bash
hcloud RDS DeleteInstance --instance_id=<id> --cli-region=<r>
```

## Backup & Recovery

```bash
hcloud RDS ShowBackupPolicy --instance_id=<id> --cli-region=<r>
hcloud RDS SetBackupPolicy --instance_id=<id> --backup_policy='{"keep_days":7,"period":"1,2,3,4,5,6,7","start_time":"00:00-01:00"}' --cli-region=<r>
hcloud RDS ListBackups --instance_id=<id> --cli-region=<r>
hcloud RDS CreateManualBackup --instance_id=<id> --name=<name> --cli-region=<r>
hcloud RDS ShowRecoveryTimeWindow --instance_id=<id> --cli-region=<r>
hcloud RDS CreateRestoreInstance --instance_id=<id> --backup_id=<id> --name=<new-name> --flavor_ref=<id> --cli-region=<r>
```

> Manual backup deletion is irreversible. Verify backup ID before deleting.

## Read Replicas

```bash
hcloud RDS CreateReadReplica --replica_of_id=<primary-id> --name=<name> --flavor_ref=<id> --volume.type=<vol-type> --volume.size=<size> --cli-region=<r>
```

## Fault Diagnosis

```bash
hcloud RDS ListErrorLogs --instance_id=<id> --start_date=2024-01-01T00:00:00Z --end_date=2024-01-31T23:59:59Z --cli-region=<r>
hcloud RDS ListSlowLogs --instance_id=<id> --start_date=... --end_date=... --cli-region=<r>
hcloud RDS ShowReplicationStatus --instance_id=<id> --cli-region=<r>
hcloud RDS ListInstanceDiagnosis --cli-region=<r>
```

## Connecting

```bash
hcloud RDS ListInstances --cli-region=<r>  # get private_ips + port
mysql -h <private_ip> -P 3306 -u root -p --ssl-mode=REQUIRED
psql -h <private_ip> -p 5432 -U root -d postgres
```

> hcloud does NOT support SQL execution. Use a database client.
> For public access, bind an EIP (see `huawei-vpc`).

### Database Client Acquisition

If a database client is not installed on the agent's machine, install one:

| Platform         | MySQL                                                        | PostgreSQL                                                     |
| ---------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| **Linux**        | `apt install mysql-client` / `yum install mysql`             | `apt install postgresql-client`                                |
| **macOS**        | `brew install mysql-client`                                  | `brew install libpq`                                           |
| **Windows**      | Download MySQL Workbench or `winget install Oracle.MySQL`    | Download pgAdmin or `winget install PostgreSQL.PostgreSQL`     |
| **Python (any)** | `pip install pymysql` then `python -c "import pymysql; ..."` | `pip install psycopg2` then `python -c "import psycopg2; ..."` |
| **Docker (any)** | `docker run -it --rm mysql:8 mysql -h <ip> -u root -p`       | `docker run -it --rm postgres:16 psql -h <ip> -U root`         |

> If no client can be installed, use the Huawei Cloud **Data Studio** console: https://console.huaweicloud.com/dms/

## Mutating Operations (Require Approval)

| Operation                          | Effect                               |
| ---------------------------------- | ------------------------------------ |
| `CreateInstance`                   | New instance (billing starts)        |
| `DeleteInstance`                   | Irreversible data loss               |
| `StartResizeFlavorAction`          | Brief interruption during switchover |
| `StartInstanceEnlargeVolumeAction` | Online, but irreversible             |
| `StartFailover`                    | Primary-standby switchover           |
| `CreateRestoreInstance`            | Creates new instance from backup     |
| `CreateManualBackup`               | Incurs OBS storage charges           |
| `DeleteManualBackup`               | Irreversible                         |

## Troubleshooting

| Error                           | Root Cause -> Fix                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Connection refused              | SG missing DB port. Add ingress rule                                                                        |
| Storage full                    | Manual resize or enable auto-scaling                                                                        |
| DBS.280241 Invalid storage type | Volume type doesn't match flavor group                                                                      |
| DBS.280448 Sold out             | Try different volume type or AZ                                                                             |
| Replication lag                 | Check `ShowReplicationStatus`. Consider read replica                                                        |
| Instance stuck BUILDING         | Check task status: `hcloud RDS ListTasks`                                                                   |
| SYS.0403                        | SCP policy denies this operation (e.g. security group rules, EIP binding). See `huawei-vpc` troubleshooting |
| EIP.7905                        | EIP quota exceeded — cannot create new EIP for public access. See `huawei-vpc` troubleshooting              |

## Security

- MUST use security groups, not open 0.0.0.0/0
- MUST enable SSL for connections
- MUST store passwords in DEW/CSMS (see `huawei-dew`)
- SHOULD enable audit logs for compliance
- SHOULD set backup policy with >= 7 day retention

## Cross-Skill References

- VPC/Subnet/Security Group: `huawei-vpc`
- DEW secrets: `huawei-dew`
- EIP for public access: `huawei-vpc`
- OBS for backup storage: `huawei-obs`
