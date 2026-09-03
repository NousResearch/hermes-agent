# Orca–Hermes Account Bridge Design

## Objective

Make the Codex account selected in Orca's status-bar account switcher the
effective first-choice Codex account for every local Hermes agent. When Hermes
rotates to another Codex credential after a confirmed authentication or quota
failure, update Orca through its authenticated local runtime RPC so the status
bar and usage meters show that newly effective account.

The existing fallback chain remains:

1. the Codex account currently selected in Orca;
2. the remaining usable Hermes Codex credentials in stable order;
3. OpenRouter `qwen/qwen3-coder-next`.

## Scope

This is a Windows-host, user-local integration for the installed Orca and
Hermes instances. It applies globally to all local Hermes agents. It does not
add account pinning per pane, copy OAuth credentials between products, or
change the behavior of native Codex sessions.

## Safety invariants

- Never print, log, transmit, or copy access tokens, refresh tokens, or the
  contents of either product's credential files.
- Match Orca and Hermes accounts only by the stable ChatGPT provider account
  identifier already present in each local OAuth JWT.
- Change only Hermes pool ordering and exhaustion metadata. Each application
  remains responsible for refreshing the OAuth session it owns.
- Never reorder or reload an in-memory pool while that Hermes process has an
  active credential lease. A running request finishes with the credential it
  started with; the new order applies to the next request.
- Use Hermes' auth-store lock and atomic persistence helpers for every pool
  mutation.
- Use Orca's authenticated local runtime RPC for Orca state changes. Do not
  edit `orca-data.json` while Orca is running.
- Preserve the existing `fill_first` strategy and OpenRouter fallback.
- Rate-limit notifications and state writes so polling cannot create a log,
  toast, filesystem, or RPC storm.

## Components

### 1. Generic live credential-pool reload

Extend `agent/credential_pool.py` with a small cross-process reconciliation
step. Before a new selection or lease, and only while the local pool has no
active leases, compare a cached auth-store fingerprint with the current
persisted provider pool while holding the existing auth-store lock. The
fingerprint may use file metadata plus a digest of the relevant provider pool;
it does not require an auth-file schema change. When the relevant persisted
state changed, reload entries, preserve valid local lease accounting, clear a
stale current cursor, and sort by the new priorities.

This layer is Orca-agnostic. It solves the general case where another trusted
Hermes process changes credential order or refreshes a pooled credential.
Existing auth-store locks remain the serialization boundary. Selection,
rotation, and persistence continue to use existing `PooledCredential` and
`write_credential_pool` behavior.

### 2. Local Orca–Hermes bridge daemon

Add a user-local bridge module launched with the Hermes virtual environment.
Only one daemon instance may run, enforced by a process-held lock file.

The daemon observes:

- Orca's active host Codex account and managed-account metadata;
- Hermes' `openai-codex` credential pool priorities and exhaustion markers;
- Orca runtime metadata required for authenticated local RPC.

It maintains a small non-secret sidecar containing only account identifiers,
timestamps, and the last bridge-originated Orca selection. This state prevents
feedback loops and survives a bridge restart. No token material is written to
the sidecar.

### 3. Wrapper lifecycle integration

Update `hermes-orca-resume.py` so an Orca-launched Hermes process idempotently
ensures that the singleton bridge daemon is running before it starts or
resumes Hermes. Start the daemon detached with a hidden Windows window. The
wrapper continues immediately; a bridge startup failure is logged but does
not block Hermes.

Start the daemon once during installation so currently open agents receive
future account changes without waiting for another restart.

### 4. Orca runtime client

The bridge uses Orca's runtime metadata and local authenticated transport to
call the existing account RPC surface:

- `accounts.list` to resolve the active host selection and provider account
  identifiers;
- `accounts.selectCodex` or `accounts.selectCodexForTarget` to change the host
  selection after Hermes rotates.

The client validates that it is talking to the currently running local Orca
runtime and treats connection loss as transient. It never falls back to
editing Orca persistence directly.

## State transitions

### Manual Orca selection

1. The bridge observes an Orca active-account change that was not its own
   acknowledged RPC echo.
2. It maps the selected provider account identifier to an existing Hermes
   credential.
3. If no matching Hermes credential exists, it leaves the current pool
   unchanged and records a warning instructing the user to authorize that
   account with `hermes auth add openai-codex`.
4. If a match exists, it moves that entry to priority `0` and keeps all other
   Codex entries in stable relative order.
5. An explicit manual selection clears only that entry's stale exhaustion
   marker, allowing the next request to probe it. It does not change tokens.
6. Idle Hermes pools see the new persisted order before their next lease.

### Hermes Codex failover

1. A Hermes process receives a confirmed 401/402/429 and persists the normal
   credential status through the existing pool logic.
2. The bridge computes the next usable entry using the same priority and
   cooldown rules.
3. If another Codex account is available, the bridge calls Orca's account
   selection RPC for that account.
4. Orca updates the active account, status-bar identity, and rate-limit meter.
5. The bridge records the RPC-originated selection before applying the same
   stable ordering in Hermes, preventing an Orca-to-Hermes echo loop.

### All Codex credentials unavailable

When no Codex credential is usable, Hermes follows its existing OpenRouter
fallback. The bridge leaves Orca on the last effective Codex account because
the Codex account switcher cannot represent OpenRouter. On the transition into
this state, it displays one Windows notification stating that Hermes switched
to OpenRouter/Qwen. A persisted state flag suppresses duplicate notifications;
the flag resets after any Codex account becomes effective again.

### Quota restoration

The bridge does not automatically jump back to an earlier account merely
because its cooldown expired. The currently displayed and effective account
remains stable. The user can select the restored account in Orca; that manual
selection makes it priority `0` and triggers a fresh probe on the next Hermes
request.

## Failure handling

- Malformed or partially written JSON: retain the last known-good state and
  retry after the next file change/poll interval.
- Orca unavailable: continue managing Hermes ordering, queue no unbounded RPC
  work, and retry with capped backoff.
- Hermes auth store locked: retry later; never bypass the lock.
- Unknown selected account: do not mutate pool ordering or credentials.
- Duplicate Hermes entries for one provider account: fail closed for that
  identity and log labels/IDs only, never token previews.
- Notification failure: log once and continue; inference routing must not
  depend on desktop notifications.
- Bridge crash: Hermes retains the last valid pool order and its normal
  fallback behavior.

## Testing strategy

All production behavior is implemented through RED–GREEN tests.

- Pure mapping tests: system default, managed account, missing match, duplicate
  match, and stable ordering of remaining credentials.
- State-machine tests: manual selection, bridge-originated RPC echo, 429
  rotation, all-Codex exhaustion, recovery, and notification de-duplication.
- Credential-pool tests: persisted priority changes are ignored during an
  active lease and adopted before the following lease.
- Persistence tests use temporary Hermes and Orca homes and real JSON parsing;
  no test accesses the user's actual credential files.
- RPC tests exercise request serialization and authentication boundaries with
  a local fake transport; an integration probe against the live Orca runtime
  verifies `accounts.list` before enabling write-back.
- End-to-end validation uses synthetic credentials in temporary stores for
  ordering, then performs a controlled live switch between the two already
  authorized account identifiers without exposing tokens.

## Installation and rollback

Before installation, create timestamped backups of every modified Hermes file
and the wrapper. Do not modify the existing unrelated dirty files in the
Hermes checkout.

Rollback consists of stopping the bridge, restoring the wrapper and core file
backups, and removing only the bridge-created sidecar/lock files. Hermes'
credential tokens remain untouched throughout installation and rollback.

## Acceptance criteria

- Selecting either configured Codex account in Orca makes it priority `0` for
  all Hermes agents before their next request.
- An in-flight request is never interrupted or switched mid-request.
- A confirmed Hermes rotation to the other Codex account updates Orca's active
  account and its usage display.
- Exhausting all Codex accounts produces one Qwen fallback notification and no
  repeated toast storm.
- Switching back to a restored account is possible directly from Orca.
- Tokens remain byte-for-byte unchanged during a selection-only transition.
- Existing Codex fallback and Qwen `xhigh` behavior remain functional.
- Unit, integration, configuration, and controlled live verification pass.
