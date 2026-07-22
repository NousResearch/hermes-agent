# DATA_RETENTION_POLICY

## Purpose
Define how long Hermes-managed data is retained, archived, or removed without exceeding the 8 GB Windows host capacity or weakening auditability.

## Default lifecycle

| Data type | Active retention | Archive threshold | Minimum legal hold | Deletion policy |
|-----------|-----------------|-------------------|--------------------|-----------------|
| Synthetic BlackGold deals/tasks/docs | 12 months | 90 days inactive | None applicable | Human approval once Arthur authorizes live mode |
| Document metadata | 7 years | 2 years inactive | Contract/legal | Human approval |
| Audit logs | 12 months | 90 days cold | Security/compliance | Human approval |
| Backup manifests | 6 months | 6 months | Disaster recovery | Automated delete |
| Restriction markers | Permanent | N/A | Compliance | Never auto-deleted |

## Enforcement rules
- Restricted/Secret data: never auto-deleted; human-approval gate required.
- Approved synthetic/test fixtures may apply shortened retention.
- Quarantine-hold outputs older than 30 days require Marcus/Naomi review.

## Restoration freeze exception
Do not execute live recovery verification against production data until Arthur expressly authorizes.
