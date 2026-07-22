# DATA_READINESS_PROPOSED_ARCHITECTURE
**Environment target:** Windows 11 Lite, 8 GB RAM  
**Constraint:** local-first, minimal additional services, synthetic-only until Arthur expressly authorizes production ingestion  
**Status:** proposed

## 1. Primary store: SQLite-first system of record
Use SQLite FTS5 as the primary structured system of record for Hermes-managed data.

Design choices:
- Single-tenant, local NTFS
- WAL mode enabled
- AES-256 encrypted at rest with Windows EFS or BitLocker
- One schema per bounded concern (deals, documents, contacts, tasks)
- Size target currently small: <1 GB dataset in phase 1

Why SQLite:
- Fast enough for single-user BlackGold workflows
- Native Hermes backup tooling already supports SQLite backups
- No extra Windows service or network binding

PostgreSQL only after:
- dataset exceeds 2 GB or concurrent-writer needs above sqlite concurrency
- Arthur explicitly authorizes additional service
- backup/restore validated in isolated restore env

## 2. Encrypted document repository
Documents are kept in a canonical encrypted directory tree:
- Root: `C:\Users\13464\AppData\Local\hermes\data\vault`
- Subdirs: `quarantine/`, `staging/`, `validated/`, `archive/`
- Encryption: BitLocker or EFS; application layer stores only metadata in SQLite, never raw secrets
- Filename scheme: `SHA256(sha256(content) || salt)` plus retention tag
- Content-addressable to prevent duplication and leaks
- Separate quota bucket for binary attachments

## 3. Analytics/reporting layer
- Hermes-generated Python reporting scripts under `ops/analytics/`
- Read-only access to system-of-record SQLite + vault metadata
- Outputs: markdown summaries, CSV exports, rendered HTMLs saved to `ops/reports/`
- Approved agents only; no direct writers to store

## 4. Optional vector retrieval
- Optional lightweight vector/keyword index in SQLite via sqlite-vec if available
- If unavailable, fallback to keyword/BM25 via FTS5
- No mandatory external vector service in phase 1

## 5. Services inventory for 8 GB Windows
| Service | Required? | RAM budget | Disk budget | Notes |
|---------|-----------|------------|-------------|-------|
| Hermes gateway | yes | 200 MB | 100 MB | already in-tree |
| Hermes CLI/agent | yes | 300 MB | 50 MB | runtime |
| SQLite databases | yes | 50 MB | up to 1 GB | system of record |
| Document vault | yes | 50 MB | 200 MB | encrypted NTFS |
| Reports worker | optional | 120 MB | minimal | on-demand |
| Vector index | optional | 100 MB | 200 MB | optional |

Total possible additional RAM: <1 GB. That fits on 8 GB if background services are constrained.

## 6. Failure modes and mitigations
| Failure | Mitigation |
|---------|-----------|
| BitLocker unavailable | EFS fallback; quarantine unclassified data |
| SQLite locking | bounded writer, single process ownership |
| Disk full | retention policy + staging quota |
| Gateway crash | canonical restart script + gateway_state.json |
| Agent overreach | approval boundaries + read-only unless authorized |

## 7. Authorization gate
No production network or external DB will be provisioned without Arthur's express authorization.
