# OAuth Broker (loopback Codex account multiplexer)

Opt-in per-machine daemon that owns one Keychain-backed Codex OAuth refresh
chain per account alias (`A`, `B`, `C`) and transparently forwards Codex
Responses/usage traffic for local Hermes profiles. Profiles stop holding
OpenAI tokens entirely — they persist only a `keychain://` reference to the
broker's local client key.

Design spec: `docs/design/oauth-broker.md`. Every command below is verified
by `tests/hermes_cli/test_oauth_broker_help.py` against the real CLI.

## Threat model in one paragraph

The broker binds `127.0.0.1` only (`127.0.0.1`/`::1` are the only accepted
bind literals; anything else refuses to start). Callers must present the
local client key (constant-time compared bearer); the key and all OAuth
grants live exclusively in the macOS Keychain, accessed through
Security.framework via ctypes — never the `security` CLI, never argv/env,
never ordinary files. Exactly two proxy routes exist per account plus two
health routes; the upstream origin is pinned to `https://chatgpt.com`
(no userinfo, no explicit port, no host variants, no path/query/fragment).
Loopback `http://127.0.0.1:<port>` upstreams require a non-production,
in-process test gate that is not exposed by the CLI, launchd plist, or normal
runtime configuration.
Forwarded request headers use a minimal allowlist (`accept`, `content-type`,
`user-agent`, `originator`, `openai-beta`, `session_id`,
`x-client-request-id`); everything else is dropped and `Authorization` /
`ChatGPT-Account-Id` are always broker-inserted. Bodies and header values
are never logged — log lines carry request id, alias, route kind, status,
and duration only. Failure is fail-closed: no legacy OAuth fallback.

## HTTP surface

```text
GET  /health                                        liveness only, no auth
GET  /health/detailed                               requires client key
POST /accounts/{A|B|C}/backend-api/codex/responses  SSE-transparent proxy
GET  /accounts/{A|B|C}/backend-api/wham/usage       usage passthrough
```

`/health/detailed` reports per alias: `healthy`, `expires_at`,
`last_refresh_result`, `persistence_degraded`. Both health routes bypass the
bounded proxy-request semaphore so a long-lived SSE stream cannot starve
liveness/readiness; proxy routes remain bounded. A `persistence_degraded: true`
account is serving a rotated grant from memory because the Keychain write
keeps failing. Persistence retries are coalesced and rate-limited. Before the
write, the broker durably records only a SHA-256 fingerprint of the consumed
Keychain refresh token (never the token itself). A restarted broker refuses to
reuse a matching stale token and reports `persistence_recovery_required`;
complete a new `auth login`. If another process or a human has already stored
a different grant, that external grant wins and the obsolete marker is
removed. A live degraded process also retains the old refresh fingerprint in
memory and compares Keychain both before persistence retries and before any
later forced/expired refresh. A different external grant wins and terminates the
old generation's request even for bare `force_refresh`; an external logout
clears the in-memory pending grant and returns retryable `not_found`, so
neither path can be overwritten or resurrected. Per-alias lock acquisition is
cancellation-safe: `KeyboardInterrupt` and cancelled async waiters close every
FD and cannot leave an orphaned flock.

## Keychain layout

```text
service ai.hermes.oauth-broker.openai-codex   accounts A | B | C   (OAuth grants)
service ai.hermes.oauth-broker.client         account  local       (client key)
```

Each OAuth grant payload has an exact versioned JSON schema. Duplicate keys,
unknown fields, boolean/float schema versions, and non-finite expiries
(`NaN`, `Infinity`) are rejected; serialization uses standard JSON with
`allow_nan=False`. `keychain://` references require full-segment matches and
reject whitespace/control characters, including final newlines.

Profile pool entries reference the client key as
`secret_source: keychain://ai.hermes.oauth-broker.client/local` with
`source: keychain_reference`; the raw value is resolved at request time and
stripped at every disk boundary.

## Commands

```bash
hermes oauth-broker install [--port 17880] [--apply]   # render plist + ensure client key
hermes oauth-broker run [--host 127.0.0.1] [--port N]  # foreground broker (loopback only)
hermes oauth-broker status [--port N]                  # GET /health
hermes oauth-broker doctor [--port N]                  # PASS/FAIL checklist
hermes oauth-broker uninstall [--apply]                # bootout argv (grants untouched)
hermes oauth-broker auth login  A|B|C                  # device-code login → Keychain
hermes oauth-broker auth status [A|B|C] [--port N]     # present/expiring/healthy booleans
hermes oauth-broker auth logout A|B|C --yes            # delete one grant (explicit only)
hermes oauth-broker migrate --profiles-root DIR --groups FILE --snapshot FILE [--apply]
hermes oauth-broker rollback --profiles-root DIR --snapshot FILE --yes
```

Notes:

- `install`/`uninstall` are render-only by default; `--apply` asks for
  interactive confirmation before any `launchctl` action runs
  (`launchctl bootstrap|bootout|kickstart gui/$UID/...`). launchd label:
  `ai.hermes.oauth-broker`; logs: `~/.hermes/logs/oauth-broker[.error].log`
  (directory created owner-only, 0700); the plist restarts the broker on
  crashes (`KeepAlive.SuccessfulExit=false`) with `ThrottleInterval=5` to
  damp crash loops.
- `run` preloads and validates all three A/B/C Keychain grants before
  binding and fails closed if any is missing.
- `status` authenticates via `/health/detailed` (client key) rather than
  trusting the unauthenticated liveness route; `/health/detailed` reports
  `present`, and an account with no loaded grant is `healthy: false`.
- `uninstall` never deletes grants; `auth logout` is separate and requires
  `--yes`.
- `auth login` asks for confirmation before storing the completed login
  under the selected alias, showing only the alias and a one-way account-id
  fingerprint. Device authorization completes before locking; final Keychain
  reconciliation, `auth logout`, and broker refresh share the same hardened
  per-alias process lock, so an old refresh cannot overwrite a concurrent
  human re-auth. Declining stores nothing.
- `migrate --apply` asks for interactive confirmation; declining keeps the
  dry-run snapshot and migrates nothing. `rollback` requires `--yes`.
- Rollback preflights every profile (each `broker-*` entry must match the
  exact migration identity: source, base_url, secret_source, priority) and
  fails closed on drift. Snapshot schema v2 includes a non-secret canonical
  store hash in addition to the raw-file hash, so restart recovery can verify
  that secret-bearing fields are unchanged without storing their values.
  Apply and rollback write an owner-only, snapshot-bound durable journal before
  the first profile write. Normal exceptions and `KeyboardInterrupt` compensate
  the full fleet immediately; an `os._exit`, kill, or power-loss mixed state is
  accepted on the next invocation only when every profile is either the exact
  canonical original or the exact broker-migrated identity. Any third state
  fails closed. Snapshots retain exact legacy `disabled`/`priority` presence and
  values, so restore reproduces the original logical shape.
- Migration/snapshot/journal writes are durable-atomic (exclusive random
  staging file, fchmod 0600, full write, file fsync, `os.replace`,
  parent-directory fsync); symlinked profile
  dirs/auth.json and profile names escaping the root are rejected; ports
  must be 1..65535. Launchd install recovery also covers
  `KeyboardInterrupt`/`SystemExit`: a post-bootstrap kickstart interruption
  boots the service out and restores the previous plist transactionally.
- The client key is generated with `secrets.token_urlsafe(32)` during
  `install`, stored in the Keychain, and never printed.
- Real OAuth logins (nine total: A/B/C on three machines) are a human-only
  R3 gate — agents must not execute or read them.

## Migration and rollback

`migrate` is a dry run unless `--apply` is passed. It discovers the complete
Hermes topology: the root `auth.json` is the `default` profile and named
profiles live under `profiles/<name>/auth.json`; a simultaneous root default
and `profiles/default` collision is rejected. It validates that the groups
file (`{"profile-name": "A|B|C"}`) exactly matches that discovered profile
set, that no profile already holds broker entries, and writes a redacted
schema-v2 snapshot (entry ids, labels, sources, priorities, status timestamps,
raw-file hashes and canonical hashes — never token values). Apply re-verifies
every file hash (any drift aborts before writing), then per profile inserts
`broker-A/B/C` references in the group's cyclic priority order and marks the
legacy entries `disabled: true` without deleting them or touching their secret
fields; disabled entries drop out of pool rotation entirely. Both apply and
rollback use snapshot-bound durable journals. A normal write failure restores
or re-applies the full batch, while a process-kill/power-loss mixed state is
strictly recognized and completed on the next invocation. `rollback` removes
the broker references, verifies canonical secret-bearing state, and restores
legacy priority/enabled state by stable entry id; then only the affected
machine's gateway needs a restart.

## Failure semantics

- Upstream `429` passes through verbatim (status, `Retry-After`,
  rate-limit/reset headers, content encoding, and wire bytes); account switching
  stays in the Hermes credential pool, never inside the broker. The shared
  upstream session uses a `DummyCookieJar` and disables automatic decompression,
  preventing cookies from crossing A/B/C account slots.
- The local broker bearer is checked before alias lookup, so an unauthenticated
  loopback process cannot distinguish configured from unconfigured aliases.
- Upstream `401` triggers exactly one locked, deduplicated refresh and one
  replay; a second `401` flows back unchanged. A delayed `401` for an older
  access-token generation reuses the already-rotated grant instead of
  consuming another refresh token.
- `invalid_grant` / `refresh_token_reused` / `token_revoked` make the
  account terminal-unhealthy (`503` with the category; no refresh loops);
  the other accounts are unaffected.
- A token-refresh transport timeout or disconnect has an unknown rotation
  outcome. The broker keeps the preflight fingerprint marker, enters
  `refresh_outcome_unknown`, and requires a new `auth login` rather than
  risking reuse of a single-use refresh token. A definitive retryable token
  endpoint response clears the marker and remains retryable.
- Broker down ⇒ requests fail closed; launchd restarts it; profiles never
  fall back to the archived legacy OAuth entries.
