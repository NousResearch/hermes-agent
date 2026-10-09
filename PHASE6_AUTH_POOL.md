# Phase 6.4 — Credential pool ownership

Branch: `refactor/phase6-auth-credentials`. Independent One Gateway baseline:
`bc1b572e2c0cdefb9a9324198df3155815fae500`. This slice does not depend on Phase 5.

## Ownership cut

The existing pool is now implemented in `auth/credential_pool.py`. Its selection,
rotation, cooldown, administration, persistence and source semantics are retained;
there is no second pool, credential registry or store.

| Canonical owner | Responsibility |
| --- | --- |
| `auth/credential_pool.py` | Pool construction, selection strategies, leases and failover orchestration |
| `auth/credential_pool_admin.py` | Add/remove/reset administration |
| `auth/credential_pool_model_cooldowns.py` | Existing credential/model cooldown policy |
| `auth/pool_sources.py` | Existing singleton, environment and custom-provider ingestion |
| `auth/pool_refresh.py` | Refresh synchronization, locking and durable rotation commits |
| `auth/credential_pool_plugin.py`, `auth/plugin_hooks.py` | Registered ProviderProfile refresh invocation and result merge/recovery |
| `auth/source_policy.py` | Existing borrowed Codex/Claude adoption policy |
| `auth/oauth_grants.py` | Single-use grant clone stripping, fork healing and associated caches |
| `auth/token_validation.py`, `auth/errors.py` | Shared token validity primitives and the existing AuthError class |

The retired agent pool administration/cooldown/plugin modules and
`hermes_cli/auth_oauth_grants.py` are deleted. Agent, Gateway, auxiliary clients,
tools, scheduler, TUI and CLI pool consumers use the canonical owner directly.
The old CLI plugin-refresh lookup body is removed in favor of the auth owner.
Generic profile secret-scope infrastructure remains in agent.

## Explicit application boundary

`load_pool(provider, *, environment)` and pool construction require a
`PoolEnvironment`. The application composition factory in
`hermes_cli/config_credentials.py` captures the active profile and supplies
the existing configuration owner, secret-source provenance, provider metadata,
endpoint normalization and provider-specific protocol callbacks.

A pool refuses operations in another active profile, including selection, leases,
administration and cooldown changes. Loading rejects a mismatched profile before
grant healing, storage or source discovery. A returning A → B → A execution can
use A's original pool again. Existing custom-provider endpoint keys, legacy display
aliases and endpoint vetoes remain the underlying isolation implementation.

The pool owns refresh invocation. Built-in protocols remain in their existing
modules until 6.5 and are supplied through callbacks; auth imports no CLI module.
Registered external provider refresh hooks still come from the existing
ProviderProfile registry. Configuration formats, keys, writers and credential
locations are unchanged.

The 6.2 credential-request/result contracts are retained. Wiring all runtime
authentication paths to those contracts belongs to the later consumer cut; this
slice moves the existing pool rather than wrapping it in another manager.

## Demonstrated external obligations

The pre-migration model-provider guide demonstrates external imports of
`AUTH_TYPE_OAUTH`, `PooledCredential` and `load_pool` from
`agent.credential_pool`. That file now contains only those public exports and
an application-edge environment adaptation for existing plugins. It contains
no selection, source, refresh or persistence implementation. Internal consumers
cannot import it, enforced by the structural gate.

The original `hermes_cli.auth_constants.AuthError` import still denotes the
same class, now defined in `auth.errors`. Existing
`ProviderProfile.auth_handler(action, args)` remains at the presentation
boundary and `refresh_credential(entry)` retains its protocol contract.

These demonstrated public imports are deliberately not added to the September
deprecation manifest. Its loader treats every entry as deprecated and disables
affected plugins after 2026-09-14; adding these obligations would break existing
plugins immediately. The external regression checks both import behavior and
the scanner's absence of deprecated-import hits. Existing manifest entries whose
implementation owner moved are updated to the canonical target.

## Verification

- Module gate: 54 files, **448 passed, 0 failed, 4 platform skips**, using
  `scripts/run_tests.sh -j 4`. Includes credential pool selection/leases,
  concurrent Nous and Anthropic refresh, profile grant-fork isolation, custom
  provider boundaries, persistence failure, lifecycle removal, CLI authentication,
  plugin hooks, multiplex credential clients, packaging metadata and compatibility.
- Consumer follow-up after fixing the custom runtime environment propagation:
  11 files, **107 passed, 0 failed**. Includes auth package contracts/structural
  guards, switch/restore, keyed and legacy custom pool lookup, voice resolution,
  provider hooks and packaging/compatibility.
- Remaining direct pool consumers: 37 files, **777 passed, 7 initially failed,
  1 platform skip**. Five migration failures were stale owner/signature test
  seams; after migration, the six-file follow-up passed **62 tests, 0 failed,
  1 platform skip**. The other two failures in
  `test_plugin_provider_picker_residue.py` reproduce unchanged on the untouched
  One Gateway baseline (1 passed, 2 failed), using the same POSIX executable
  fixture pattern on Windows.
- The earlier expanded consumer gate covers Gateway session credentials,
  multiplexing, auxiliary fallback, endpoint vetoes, cooldown recovery and TUI.
  Its migration regressions were corrected and verified in the follow-up.
- `test_plugin_provider_picker_admission.py` has one Windows failure in its
  POSIX shell executable fixture. The identical failure was reproduced on the
  untouched One Gateway baseline in an isolated worktree (4 passed, 1 failed);
  it is not suppressed or counted as a passing migration check.

Final structural checks passed: syntax for 291 changed/new Python files; no new
F821/F811 diagnostics (121 pre-existing findings checked against HEAD); no pool
call sites missing explicit environment; no dependencies on the 2084 deprecated
plugin pointers; configuration-writer audit for 1900 files; configured auth/adapter
lint; and Git whitespace checks.

The complete repository integration/build and platform lanes remain the 6.8
gate. Built-in OAuth protocol ownership moves in 6.5; remaining runtime
authentication and presentation separation follow in 6.6–6.7.
