## Summary

Fixes multi-profile death for xAI Grok OAuth (issue #65394): xAI issues **single-use rotating refresh tokens**, so when each Hermes profile forks its own copy into `auth.json`, the first profile to refresh **revokes** every other copy (`invalid_grant`). This PR introduces an **opt-in canonical shared store** that all profiles read and refresh under one shared lock — one grant, one refresher, no per-profile forking.

**Activation is config.yaml (not a user-facing env var).** This addresses the `env-var-for-config` closure of #67243 (AGENTS.md: non-secret feature flags belong in `config.yaml`; bridge to an internal env var if the mechanism needs one — same pattern as `terminal.cwd` → `TERMINAL_CWD`). The shared-store **engine** (election/sweep/marker/lock/consumer-routing) is unchanged; only the on/off switch moved.

## Problem

Today each profile (and the root `~/.hermes/auth.json`) can hold its own xAI OAuth grant. xAI rotates the refresh token on every successful refresh:

1. Profile A refreshes → new RT written to A's store; old RT revoked at xAI.
2. Profile B still holds the old RT → next refresh gets `invalid_grant`.
3. Concurrent gateways / cron / desktop / CLI workers multiply the race.

This is the multi-profile fork/death path described in #65394. Write-through helpers and pool entries that also persist raw refresh tokens make the blast radius worse.

## Solution

A **canonical sole-owner store** (not a Nous-style convenience layer) at:

```text
${HERMES_SHARED_AUTH_DIR}/xai_oauth.json   # default: ~/.hermes/shared/xai_oauth.json
${HERMES_SHARED_AUTH_DIR}/xai_oauth.lock   # cross-process advisory lock
```

All profiles **read / refresh / persist** through one resolver under that lock. Profile and root stores keep only a non-secret `source: shared:xai-oauth` reference (no raw RT forking).

### Opt-in gate (config.yaml user-facing; internal env bridge)

Shared xAI mode does **not** activate just because `HERMES_SHARED_AUTH_DIR` is set (that directory is already used for Nous shared auth).

**User-facing activation** (non-secret feature flag in config.yaml):

```yaml
# ~/.hermes/config.yaml  (or a profile's config.yaml)
shared_auth:
  providers: [xai-oauth]
  # optional: dir: ~/.hermes/shared
```

```bash
hermes auth xai enable-shared    # writes the config key + bridges for this process
hermes auth xai disable-shared   # removes xai-oauth from shared_auth.providers
```

At process startup (CLI, gateway, cron, dashboard/serve, TUI), Hermes **force-exports** the internal bridge targets from that config (AGENTS.md env-var-for-config / `terminal.cwd` → `TERMINAL_CWD` precedent):

- `HERMES_XAI_SHARED_AUTH=1` when providers includes an xAI alias
- `HERMES_SHARED_AUTH_PROVIDERS=<comma-joined list>`
- `HERMES_SHARED_AUTH_DIR` only when `shared_auth.dir` is set

The engine gate (`_xai_shared_auth_enabled()`) still **reads only those env vars** — they remain the internal mechanism so tests and power-user overrides keep working. **Gate-off** (absent/empty `shared_auth` in config) does not set or modify those env vars → byte-identical legacy behavior.

### Key properties

| Property | Behavior |
|----------|----------|
| **Canonical sole-owner** | One durable RT in the shared store; profiles never hold a forked secret copy |
| **Atomic multi-store election + sweep** | Under all-store locks: elect a winner, write shared, strip residual RTs across profile/root/pool/manual rows |
| **Fail-closed shapes** | Unreadable or wrong-shape stores (list, string, number) fail election / strip / sole-owner audit — never silent first-wins |
| **Fleet sole-owner marker** | Verifiable digest over inventory; stale/existence-only markers never skip re-audit |
| **Fail-loud persistence** | Durable write + parent-dir fsync; poison/partial write leaves non-promotable state |
| **Quarantine compare-and-clear** | Dead grants quarantined by generation; concurrent winner is adopted, not double-rotated |
| **Full consumer routing** | Runtime, credential pool, auxiliary client, proxy adapter, x_search / image / video / tts / stt, availability probes all go through the canonical resolver |
| **Codex / Nous isolation** | Shared xAI path does not couple into Codex or Nous shared-auth ownership |
| **Config-not-env activation** | User-facing switch is `shared_auth.providers`; env vars are internal bridge targets only |

### Enablement + migration

```bash
# 1) Opt in (writes config.yaml; restart gateway/cron/desktop so they reload)
hermes auth xai enable-shared

# 2) One-time fleet migrate (or a single device-code login seeds the shared store)
hermes auth xai migrate-shared --source auto
# --source profile | --source root
# --force overwrites an existing shared grant

# 3) Login (if no legacy grant to migrate)
hermes auth add xai-oauth
```

**Logout semantics under shared mode:**

- `hermes auth xai disable-shared` — removes `xai-oauth` from `shared_auth.providers` in config.yaml (canonical grant stays on disk)
- `hermes logout --provider xai-oauth` — per-profile disable marker while shared mode remains on; canonical grant stays for other profiles
- `hermes logout --provider xai-oauth --global` — deletes the grant for **every** profile (intentionally noisy)

Requires a **local** filesystem with reliable advisory locking (not NFS/SMB). Every gateway/cron/desktop process must load the same config.yaml (shell-only exports are not the activation path).

## Testing

Extensive coverage (adversarial review rounds R1–R7 on the engine) plus config-bridge tests:

- `tests/hermes_cli/test_shared_auth_config_bridge.py` — config → env bridge, gate-off byte-identical, power-user env override preserved, enable/disable write config.yaml
- `tests/hermes_cli/test_xai_shared_auth_store.py` — gate off/on, generation bump, fail-loud persist, concurrent waiters adopt winner, quarantine compare-and-clear, migrate/strip sole-owner, election fail-closed on unreadable/wrong-shape stores, concurrent logout vs promote, fleet marker digest races, no-resurrection after quarantine, gate-off byte-identical legacy
- `tests/agent/test_auxiliary_xai_shared_recovery.py` — aux auth-error recovery with rejected bearer / generation
- `tests/agent/test_credential_pool_oauth_writethrough.py` / `test_credential_sources_xai_remove.py` — pool write-through + source removal under shared mode
- `tests/hermes_cli/test_xai_oauth_writethrough.py` — OAuth write-through boundaries
- `tests/tools/test_xai_http_shared_mode.py` — tool HTTP path uses canonical resolver
- `tests/plugins/video_gen/test_xai_plugin.py` — plugin routing under shared mode

Includes deterministic concurrency (election→commit race, strip→inventory race), fail-closed invalid shapes, gate-off byte-identical, and no-resurrection guarantees.

## Docs

- `website/docs/guides/xai-grok-oauth.md` — Shared-store mode uses config.yaml / `enable-shared`
- `website/docs/user-guide/configuration.md` — `shared_auth` section
- `website/docs/reference/environment-variables.md` — `HERMES_XAI_SHARED_AUTH` / `HERMES_SHARED_AUTH_PROVIDERS` labeled **internal bridge targets** (do not set by hand)

## Scope notes

- **Opt-in only** — default behavior unchanged until `shared_auth.providers` includes an xAI alias (or a power-user sets the internal env var).
- **Engine untouched** — `_xai_shared_auth_enabled()` body still reads env only; election/sweep/marker/lock/routing unchanged.
- Does not change Codex or Nous shared-auth semantics beyond co-using the directory path convention.
- Closes / addresses #65394; re-scopes activation to satisfy the policy that closed #67243 (`env-var-for-config`).

## Checklist

- [x] Gate-off preserves legacy path (proven by bridge + store tests)
- [x] User-facing activation is config.yaml (not a new `HERMES_*` .env flag)
- [x] Internal env bridge at CLI / gateway / cron / dashboard / TUI startup
- [x] All xAI OAuth consumers routed through canonical resolver
- [x] Fail-closed election / strip / audit on bad store shapes
- [x] Docs for enablement + migration + logout
- [x] Adversarial concurrency + no-fork tests
