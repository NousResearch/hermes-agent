# Phase 6.6 — Runtime authentication consumers

Completed on the independent `refactor/phase6-auth-credentials` branch, following
6.5 commit `d02c166ebe`.

## Ownership and consumers

`auth/secret_validation.py` owns shared secret validation and declared-prefix
policy. `auth/failure_policy.py` owns runtime quota classification.
`auth/api_keys.py` owns API-key source precedence: model credential pointers,
profile-aware environment reads, provider declarations and the existing pool
fallback. It requires explicit scoped settings and a supplied secret reader.
No second credential manager or source registry was introduced.

`auth/keepalive.py` owns the existing process-wide Nous keepalive lifecycle,
selected-pool refresh, singleton refresh and expiry horizon. Application startup
supplies the configuration factory and scope context. Both lifetime observation
and refresh execute within that context; retained settings fail before credential
reads in another profile. The old CLI implementation is deleted.

Agent, Gateway, auxiliary inference, TUI profile cloning, service tools and provider
plugins now import these shared operations directly from their canonical owners.
Spotify availability reads canonical provider status; xAI no longer imports the
CLI authentication module for its credential path. CLI setup wizards use the same
canonical API-key source resolver with explicit settings.

Provider identities, alias tables, provider/model selection, endpoint policy and
presentation retain their existing owners. Existing API-key route materialization
and external-process route bodies now live in the application routing module
`hermes_cli/runtime_provider_credentials.py`; Actual endpoint normalization lives
in `hermes_cli/route_identity.py`. These are routing responsibilities, not shared
authentication implementations. Configuration ownership and persisted formats
are unchanged. Documented external plugin contracts remain intact.

## Verification

Affected consumer regression runs and verified follow-ups cover 90 distinct test
files: 1,420 tests passing, two platform skips, no unresolved failures. Coverage
includes agent and auxiliary inference, Gateway startup, pool cooldown and
recovery, concurrent refresh, OAuth profile-fork protection, provider boundaries,
plugin discovery, CLI authentication, Bedrock setup, credential removal, Spotify
and xAI clients, and quota handling. All runs used `scripts/run_tests.sh -j 8`.

New contract tests exercise real profile files across A→B→A, rejection of retained
settings before source/store callbacks, and profile-specific Spotify availability.
Independent imports reject reverse CLI dependencies in the new canonical modules.
The structural guard prevents static and lazy imports of shared authentication
operations from CLI owners, including source-selection calls at the CLI edge.

All 75 changed/new Python modules pass syntax checks and add no F821/F811
diagnostics relative to the 6.5 checkpoint. The compatibility audit found no
in-tree dependencies on any of the 2,084 plugin compatibility pointers. The
final diff whitespace check passed.

6.7 authentication presentation integration and 6.8 full repository build,
packaging, compatibility and integration closeout remain pending.
