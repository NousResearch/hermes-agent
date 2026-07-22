# ENVIRONMENT DEFINITIONS

## Principle
Separate write authority, data exposure, and backup targets across environments.

## Environments

### Lagoon
- Purpose: experiment and code generation only
- Scope: design docs, synthetic schema, test fixtures, validation matrix
- Data: synthetic only
- Services: interactive local dev profile
- Backup: None
- Authorization: auto for synthetic artifacts only
- Logging: verbose allowed

### Harbor
- Purpose: integration validation
- Scope: policy binaries, staging schema, backup/restore drills
- Data: synthetic plus approved test datasets
- Services: Hermes gateway, backup scheduler, network monitoring
- Backup: daily, 30-day retention
- Authorization: Arthur approval required for promotion to Sandbox
- Logging: normal audit level

### Anchor
- Purpose: production-only authoritative state
- Scope: validated BlackGold data, approved documents, audit trail
- Data: only after Arthur authorization
- Services: canonical gateway, audit append, monitoring
- Backup: daily + weekly, 1 year, off-box copy preferred
- Authorization: Arthur approval required; Naomi for restricted data
- Logging: full audit + tamper-evident hash chain

## Environment promotion standards
- Every environment must pass hermetic validation before promotion.
- Approval id and evidence required.
- Harbor cannot automatically touch Anchor data.
