# Phase 6.7 — Authentication presentation boundary

Completed on the independent `refactor/phase6-auth-credentials` branch after
6.6 commit `5c619083ce`.

## Final ownership

The genuine authentication commands, including the legacy login notice and logout
command, live in `hermes_cli/auth_commands.py`. Commands invoke the defining
provider presentation modules and canonical authentication operations directly.
The routing module no longer exports login helpers, command bodies, OAuth grant
management, refresh implementations, or error-rendering functions for internal use.

`auth/provider_status.py` owns read-only Codex, xAI, MiniMax and registered-plugin
OAuth observations. Applications supply scoped settings explicitly. Snapshots
retain pool-first observation, expiry and refresh-needed semantics without
spending grants or rotating pool order. Plugin registration remains with its
existing owner; the CLI reads that owner's live mirrored-provider set and renders
the existing sign-in hint. Codex route presentation stays with the current
provider/model routing owner rather than becoming authentication policy.

`auth/provider_state.py` owns provider activation/restoration, service-login
persistence, logout adoption-source cleanup and locked metadata updates.
`auth/pool_persistence.py` owns legacy custom-pool key migration.
`auth/sources.py` owns source re-enabling. The source-removal and refresh paths
continue using the existing canonical pool and store. No CLI authentication
module implements an auth-store write transaction.

`auth/api_keys.py` also owns the remaining Anthropic declared-environment key
lookup, preserving dotenv precedence and requiring explicit settings.
`auth/failure_policy.py` owns fallback quota classification; genuine error
rendering lives in `hermes_cli/auth_error_copy.py`.

Prompts, browser launching, interactive login orchestration, command arguments,
status formatting and `ProviderProfile.auth_handler(action, args)` remain at the
presentation boundary. Runtime refresh hooks remain under authentication ownership.
Configuration formats, credential locations, saved sessions and plugin contracts
are unchanged. Configuration callers were rewired without changing configuration
loading, keys or persistence.

## Subsequent nous_cli integration

This independent One Gateway baseline has no `nous_cli`. As recorded in the
Phase 6 baseline specification, creating a second CLI or depending on Phase 0/5
would break branch independence. The independent 6.7 cut therefore makes the
existing presentation call canonical owners directly.

When Phase 0 CLI integration is available, relocate the actual authentication
presentation bodies and argument/parser dispatch into `nous_cli`: command
implementation, device/browser interaction, built-in interactive login modules,
error copy and the plugin CLI-action dispatcher. Update application entry points,
setup/model flows and tests to those defining modules in the same hard cut.
Retain plugin `auth_handler(action, args)` at that edge; keep runtime refresh,
stores, pools and source policy in `auth/`. Do not add `hermes_cli` forwarding
modules or alter provider/model routing as part of that cut.

## Verification

Affected regression runs and verified follow-ups cover 89 distinct test files:
1,009 tests passing, 10 platform skips, no unresolved failures. All runs used
`scripts/run_tests.sh -j 8`. Coverage includes CLI authentication commands,
browser and device flows, read-only pool observations, model setup, provider
plugins and PKCE hooks, concurrent refresh, profile isolation, credential
lifecycle, auxiliary inference, runtime routing and desktop OAuth cards.

All 78 changed/new Python modules compile and add no F821/F811 diagnostics
relative to the 6.6 checkpoint. The final internal ownership audit finds no
imports or literal patch bindings through removed authentication exports.
The compatibility audit reports no in-tree dependency on any of the 2,084
existing plugin pointers.

Structural checks reject reverse CLI imports, obsolete internal authentication
imports and auth-store transactions implemented by CLI authentication modules.
Fresh-process import checks include the new status owner. Compatibility target
identity and the supported provider PKCE hook contract are verified.

6.8 integration, build, packaging, compatibility and independent PR closeout
are recorded in `PHASE6_AUTH_CLOSEOUT.md`, including baseline failures and the
full TUI validation limitation.
