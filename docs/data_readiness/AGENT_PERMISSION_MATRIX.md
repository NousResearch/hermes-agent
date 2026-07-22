# AGENT_PERMISSION_MATRIX

## Purpose
Define per-agent tool scope, data permissions, review boundaries, and approval steps. No agent receives DBA/owner/admin privileges over production systems.

## Core rules
- System of record writes are delegated to a service account with append-only DB permissions; agents never own credentials.
- Human approval required for any access to *restricted* data.
- Audit log append is permitted to no agent except operational monitoring.
- Self-modification of operational systems is limited to nonproduction artifacts under version control.

## Tiered permissions

| Agent | System of record | Document vault | Reports | Config | Approvals |
|-------|-----------------------|----------------|---------|--------|-----------|
| Marcus | admin | admin | none | read | approvals |
| Grant | append | read | read | none | approvable |
| Malcolm | append | read | read | read | approvable |
| Julian | read | append | read | none | approvable |
| Elijah | none | read | read | none | approvable |
| Jordan | read | read | read | none | approvable |
| Miles | admin | read | read | none | approvable |
| Naomi | read | read | read | admin | approvable |
| Olivia | append | append | none | none | approvable |
| Grace | none | none | none | none | approvable |
| Caleb | read | read | read | none | approvable |
| David | read | read | read | read | approvable |
| Sophia | none | none | read | none | approvable |
| Victor | read | read | read | read | approvable |
| System services | none | none | none | none | approvals |

## Notes
- "approvable" = can request via HUMAN_APPROVAL_GATES.
- Restricted/Secret data always requires Arthur endorsement.
- Approval requests must log required evidence docs.
