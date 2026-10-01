# Phase 6.3 — Authentication storage and source lifecycle

Branch: `refactor/phase6-auth-credentials`, independently based on One Gateway `bc1b572e2c`.

## Ownership cut

| Canonical owner | Responsibility |
| --- | --- |
| `auth/store.py` | Current/profile and global-root store paths, fail-closed reads, corruption recovery, path-keyed reentrant process locks, private atomic JSON writes and cache invalidation |
| `auth/store_migrations.py` | Existing legacy authentication JSON compatibility and stale Nous portal URL migration |
| `auth/provider_state.py` | Provider-state reads, active-provider state, transactions and write-through to the store owning a rotating grant |
| `auth/pool_persistence.py` | Durable pool snapshots, concurrent additions/removals, token-generation adoption and cooldown/reset merging |
| `auth/persistence.py` | Borrowed-credential disk sanitization and fingerprints; moved directly from agent |
| `auth/sources.py` | Source suppression and coordinated environment credential save/removal |
| `auth/source_removal.py` | The existing single source-removal registry, including owned OAuth files, borrowed sources and Copilot duplicate-source suppression |

The old storage/state/suppression implementations have been removed from
`hermes_cli.auth`. `agent/credential_persistence.py` and
`hermes_cli/credential_lifecycle.py` are deleted. Their internal consumers and
regression-test patch points now use the canonical owners. No old-path storage
facade, second store, second pool or migration fallback was introduced.

## Configuration boundary

Applications explicitly supply `CredentialEnvironment` to credential lifecycle
operations. It binds the existing profile scope and supplies environment reads,
saves/removal, raw configuration mirror reconciliation, provider metadata,
cache invalidation and pool seeding callbacks. Operations reject a foreign
profile boundary before any mutation.

`hermes_cli/config_credentials.py` supplies these collaborators. The existing
configuration loaders, keys, writer and persistence behavior are unchanged.
Configuration mirrors still go through `atomic_config_write`; auth has no CLI
imports. Cache and pool collaborators are loaded only when invoked, preserving
best-effort failure isolation and lazy provider discovery. TUI handlers import
the boundary locally because their globals are rebound onto the server.

## Persisted and plugin compatibility

Authentication file names, paths, JSON version and schema, grant provenance,
root/profile shadowing, suppression values, file permissions and lock ordering
are preserved. Shared rotating grants still write back to their owning store.

The demonstrated external `agent.credential_sources.register` obligation is
retained only inside its marked plugin-compat block. It refers to the one registry
in `auth.source_removal`; internal callers do not use the old pointer. The existing
compatibility manifests now identify the canonical target. Provider auth-handler
and refresh-hook contracts are unchanged.

## Verification

All tests use `scripts/run_tests.sh` with the hermetic test interpreter.

- Storage, pool, OAuth consumers and provider registration: 52 files, **417 passed**, 6 skipped on this Windows host.
- Ten completed affected consumer files: **129 passed**, 1 platform skip. These cover Gateway notices, dashboard destructive profile scopes and credential lifecycle, anonymous authentication, inventory, model-provider persistence, Codex write-through, TUI free-tier RPC and Photon.
- Focused TUI credential-save/profile tests: **2 passed**. The full large TUI file was interrupted and narrowed to the changed handlers; full integration remains a 6.8 gate.
- Final auth-package, packaging metadata and external-login opt-out run: 5 files, **39 passed**. Includes the new storage-import guard blocking CLI imports and provider discovery.
- Canonical-module lint passes. A broad check of touched code found only the 56 existing baseline diagnostics in three files; comparison to the base introduced no new diagnostics.
- Direct ownership audit confirms the moved definitions have canonical owners and internal imports of the retired storage/lifecycle paths are absent.
- Configuration YAML writer check passes across 1,904 files.

- Final compatibility-target and profile-isolation rerun: 2 files, **34 passed**.
- UTF-8 compatibility-pointer scan passes: no in-tree dependency on the 2,084 external-only pointers. The initial scan reached its success print but failed on the Windows console encoding; rerunning with Python UTF-8 mode passed without changing the checker.

Whole-repository builds, Linux/POSIX-only permission coverage and the complete
regression/integration gate remain scheduled for 6.8. Changes are uncommitted.

## Next slice

6.4 moves the existing pool implementation, administration, rotation, model
cooldowns, source readers and borrowed-login policy into auth. Pool orchestration
and built-in OAuth protocol implementations retain their current owners until
those declared cuts; the store does not create a replacement manager.
