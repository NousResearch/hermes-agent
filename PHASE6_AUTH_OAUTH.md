# Phase 6.5 — OAuth lifecycle and provider authentication

Completed on the independent `refactor/phase6-auth-credentials` branch.

## Ownership

`auth/oauth.py` owns shared PKCE, loopback callback and device-flow primitives,
OAuth failure classification and lifecycle safeguards. `auth/providers/` owns the
built-in runtime protocols, refresh operations and provider credential resolution.
Codex transport, quota, browser and device protocols have separate modules; Nous
runtime, store and status responsibilities are likewise separated.

The canonical pool invokes registered provider refresh hooks. Provider identity,
discovery, routing, model selection and account entitlement policy retain their
existing owners. There is no second credential registry or authentication store.

Application startup supplies `PoolEnvironment`, including scope, configuration
callbacks and presentation-owned entitlement/user-agent callbacks. Canonical
operations validate the current scope before accessing credentials. Authentication
modules do not import `hermes_cli`, `nous_cli` or CLI configuration.

CLI modules retain interactive login, prompts, browser launching, status rendering
and the public plugin `auth_handler(action, args)` contract. This baseline has no
`nous_cli`; CLI namespace integration remains separate. Documented PKCE plugin
configuration and refresh-factory exports, AuthError identity and existing plugin
compatibility targets remain supported at their actual external boundaries.

The old Anthropic runtime implementation and Qwen CLI runtime module are deleted.
Credential formats, locations, refresh rotation, persistence failure handling,
external-login policy and profile isolation are preserved.

## Verification

The final affected regression gate ran through `scripts/run_tests.sh -j 8`:
185 files, 2,469 tests passed, zero failed, 31 skipped on the Windows host.
It includes auth storage, pools, concurrent refresh, grant protection, profile
isolation, provider plugins, agent, auxiliary clients, Gateway, TUI, CLI and desktop
authentication coverage. Platform-specific skipped tests remain for their CI lanes.

All 291 changed/new Python modules passed syntax verification. Compared with the
6.4 checkpoint there are no new F821/F811 diagnostics. Auth import-boundary and
new explicit-scope contract tests passed. The compatibility audit verifies all
2,084 existing plugin pointers without internal dependence on those pointers.

6.6–6.8 remain pending, including the full repository integration/build gate and
independent Phase 6 PR closeout.
