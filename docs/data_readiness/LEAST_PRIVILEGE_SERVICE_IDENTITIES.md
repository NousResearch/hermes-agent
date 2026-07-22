# LEAST_PRIVILEGE_SERVICE_IDENTITIES

## Goal
Ensure agents and services operate with minimum required authority.

## Identity model
- Service identities are separate from user/Hermes config.
- Each integration gets a named service account with explicit capability list.
- DBA/admin privilege never granted to agents under any bundle.

## Service identities

### hermes-backup
- Credential type: local filesystem only
- Permissions: read system of record, write backup dir, append audit
- No outbound network access

### hermes-audit-writer
- Credential type: local filesystem only
- Permissions: append audit log only
- No data read beyond metadata

### ingestion-validator
- Credential type: internal service token
- Permissions: read quarantine, write parsed, append audit
- No document vault write

### analytics-reporter
- Credential type: internal service token
- Permissions: read-only to system of record + metadata
- No write access to validated data

## PATH prevention
- Each service identity configured at service invocation boundary.
- No interactive agent gifted worker-identity credentials.

## Credential storage
- Central secrets store encrypted at rest.
- No plaintext credentials in `config.yaml`, `.env`, or docs.

## Rotation
- Service token cadence quarterly for Harbor/Anchor.
- Rotation audit maintained by Naomi.
