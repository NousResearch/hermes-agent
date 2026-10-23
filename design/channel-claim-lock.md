# Design: Channel Claim-Lock (Record, Lease, and Admission Semantics)

Task: t_691a6334 · Builds on RCA t_de412919 · Author: ponytailkimi

## 1. Problem

A thread reply in a DM currently receives a session key that includes the thread id, while the parent DM session uses a key without it. `SessionStore.get_or_create_session` only checks the exact session key string. A thread reply therefore mints a fresh session even when the DM session for that same channel is live. This produces two answers to one user.

The fix is a channel-scoped claim lock: per (platform, scope_id, chat_id, profile) only one session can be active at a time. Thread replies route into that session (thread match) or queue behind it.

## 2. Scope

| Item | In scope | Out of scope |
|---|---|---|
| Intra-host SQLite claim table | Yes | — |
| Cross-instance relay claim surface | Yes (best-effort, documented dependencies) | — |
| Acquire / renew / release / expire state machine | Yes | — |
| Admission-time decision table | Yes | — |
| Event drain on session end (via existing admissions queue) | Yes | — |
| Crash-safe lease expiry | Yes | — |
| Second-gateway-on-same-token arbitration | Yes | — |
| Changing session key format | No — keep existing keys | |
| Message-level ordering guarantees | No — queue drain is best-effort | |
| New relay connector frame type for cross-instance forwarding | No — named as dependency | |

## 3. Claim Record Schema

Both backends share this record shape. The primary key is the **channel scope:** `(platform, scope_id, chat_id, profile)`. Thread roots are tracked but do **not** widen the key — one claim covers the channel and every thread inside it.

```
class ChannelClaim:
    platform: str               -- Platform enum value (e.g. "mattermost")
    scope_id: str               -- Slack workspace; empty string for platforms that do not scope
    chat_id: str                -- channel / chat identifier from SessionSource.chat_id
    profile: str                -- normalized profile name (see §3.3)
    gateway_instance_id: str    -- intra-host: SessionAuthority.instance_id; cross-host: relay_instance_id()
    session_id: str             -- the session that currently holds the claim
    thread_root_id: str         -- the thread root active in this session (empty string if none)
    lease_expire: float         -- Unix epoch seconds; default time.time() + 300; renewed on activity
    created_at: float           -- Unix epoch seconds, set once at acquire
    updated_at: float           -- Unix epoch seconds, touched on every renew
```

`thread_root_id` is a **tag**, not a key component. It records which thread the session last spoke in so outbound replies target the right thread. It does **not** allow multiple simultaneous claims for different threads in the same channel.

### 3.1 SQLite table

Hermes `state.db` uses `REAL` for timestamps (Unix epoch float), matching `gateway_routing` and `session_admissions`.

```sql
CREATE TABLE IF NOT EXISTS channel_claims (
    platform     TEXT NOT NULL,
    scope_id     TEXT NOT NULL DEFAULT '',
    chat_id      TEXT NOT NULL,
    profile      TEXT NOT NULL,
    instance_id  TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    thread_root_id TEXT NOT NULL DEFAULT '',
    lease_expire REAL NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (platform, scope_id, chat_id, profile)
);
CREATE INDEX IF NOT EXISTS idx_channel_claims_lease
    ON channel_claims(lease_expire);
CREATE INDEX IF NOT EXISTS idx_channel_claims_session
    ON channel_claims(session_id);
```

Notes:
- `profile` stores the normalized profile name — `"main"` for default/None, `"main~"` for a profile literally named `main`, or the explicit profile id. A shared normalization helper (see §3.3) must be used by every caller.
- `instance_id` is `SessionAuthority.instance_id` on the intra-host path and `relay_instance_id()` on the cross-host path.
- `scope_id` prevents Slack workspace collisions: two workspaces with the same `chat_id` must not share a claim.
- `thread_root_id` defaults to `''` (empty string) for channel-level conversations.

### 3.2 Cross-instance relay claim surface

Cross-instance claims are stored in the existing relay file tree:
- `bot_relay/channel_claims/<platform>/<scope_id>/<chat_id>/<profile>.json` — one file per claim.
- `scope_id` path segment is omitted when it is the empty string, so the path depth stays minimal for non-Slack platforms.
- File content is the JSON serialization of the `ChannelClaim` record above.
- Atomic acquire is a rename from `outbox/` to `channel_claims/` (same pattern as `tools/bot_relay.py` atomic envelope claim).
- Conflict detection: if the file already exists, read it; if `lease_expire` is in the past, overwrite (steal); else the acquire fails.
- The existing `_sweep_stale` in `tools/bot_relay.py` uses `glob("*.json")` (not `rglob`). Nested claim files are therefore immune to envelope sweep. This is documented as intentional.

### 3.3 Profile normalization helper

All claim callers use the same normalization as `_session_key_namespace` (minus the `agent:` prefix):

```python
def _normalize_profile_for_claim(profile: Optional[str]) -> str:
    if not profile or profile == "default":
        return "main"
    return "main~" if profile == "main" else profile
```

This helper lives in `gateway/session.py` next to `_session_key_namespace` and is imported by `hermes_state_runtime.py` and `tools/bot_relay.py`.

## 4. State Machine

```
 States:
   ABSENT   -- no claim record for this channel
   HELD     -- a gateway holds the claim; lease_expire is in the future
   EXPIRED  -- lease_expire is in the past; the record is stale

 Transitions:
   ABSENT --(acquire)--> HELD
   HELD   --(renew)----> HELD   (lease_expire extended)
   HELD   --(transfer)--> HELD   (session_id changes during reset; lease extended)
   HELD   --(release)--> ABSENT  (row deleted, or for relay, file unlinked)
   HELD   --(expire)---> EXPIRED (implicit; detect on read)
   EXPIRED--(steal)----> HELD   (new instance overwrites stale record)
```

Renewal policy:
- The session turn loop extends the lease at most once per turn. The hook is in `run_turn_runner.py` after a successful model response is delivered and the session state is committed.
- Lease extension sets `lease_expire = now + LEASE_SECONDS`.
- `LEASE_SECONDS = 300` (5 minutes). This is the same order of magnitude as existing session idle timeouts.
- A session that ends normally releases the claim immediately, so the channel opens for the next event.
- A crash leaves an EXPIRED record. The next event from any gateway steals it after `lease_expire`.
- An open-but-idle session (>300 s since last turn) has an EXPIRED claim. New events for that channel steal the claim and create a fresh session. The idle session row remains in `sessions` (normal Hermes behavior) but no longer blocks the channel.

## 5. Acquire Semantics

### 5.1 Intra-host SQLite acquire

```python
def acquire_channel_claim(
    db, platform: str, scope_id: str, chat_id: str, profile: str,
    instance_id: str, session_id: str,
    lease_seconds: int = 300,
) -> tuple[bool, Optional[ChannelClaim]]:
    """
    Atomic acquire.
    Returns (acquired, current_claim).
    If acquired is True, current_claim is the newly written row.
    If acquired is False, current_claim is the existing row (may be expired).
    """
```

Implementation sketch (uses `SessionDB._execute_write`, the existing atomic-write API):

```python
now = time.time()  # same epoch float as gateway_routing / session_admissions
lease_expire = now + lease_seconds
profile = _normalize_profile_for_claim(profile)

def _atomic_insert():
    db._execute_write(lambda conn: conn.execute("""
        INSERT INTO channel_claims
            (platform, scope_id, chat_id, profile, instance_id, session_id,
             thread_root_id, lease_expire, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (platform, scope_id, chat_id, profile, instance_id, session_id,
          '', lease_expire, now, now)))

try:
    _atomic_insert()
    return True, ChannelClaim(...)
except IntegrityError:
    # Already held. Read the current row to let caller decide (steal if expired, or queue/wait).
    row = db._read_one("""
        SELECT * FROM channel_claims
        WHERE platform=? AND scope_id=? AND chat_id=? AND profile=?
    """, (platform, scope_id, chat_id, profile))
    return False, ChannelClaim._from_row(row)
```

**Steal on expiry:** before returning `(False, claim)`, check `if claim.lease_expire < now`. If expired, overwrite inside a single `_execute_write` callback:

```python
def _atomic_steal(conn):
    # CAS read inside the write transaction
    old = conn.execute("""
        SELECT lease_expire FROM channel_claims
        WHERE platform=? AND scope_id=? AND chat_id=? AND profile=?
    """, (platform, scope_id, chat_id, profile)).fetchone()
    if old is None:
        # Row disappeared between our failed INSERT and now; INSERT again
        conn.execute("""
            INSERT INTO channel_claims (...)
            VALUES (...)
        """)
        return True
    if old[0] >= now:
        return False  # someone renewed or stole first
    conn.execute("""
        UPDATE channel_claims
        SET instance_id=?, session_id=?, thread_root_id=?,
            lease_expire=?, updated_at=?
        WHERE platform=? AND scope_id=? AND chat_id=? AND profile=?
          AND lease_expire=?
    """, (instance_id, session_id, '', lease_expire, now,
          platform, scope_id, chat_id, profile, old[0]))
    if conn.total_changes == 0:
        return False  # lost the race
    return True

stolen = db._execute_write(_atomic_steal)
```

### 5.2 Cross-instance relay acquire

```python
def acquire_relay_claim(
    relay_root: Path, platform: str, scope_id: str, chat_id: str, profile: str,
    instance_id: str, session_id: str,
    lease_seconds: int = 300,
) -> tuple[bool, Optional[ChannelClaim]]:
```

Atomic acquire uses the same rename pattern as `tools/bot_relay.py::claim_pending_envelopes`:

1. Build the claim path: `<relay_root>/bot_relay/channel_claims/<platform>/<scope_id>/<chat_id>/<profile>.json`. Omit empty `scope_id` segments.
2. Write a temporary file in the parent directory with suffix `.tmp`.
3. `rename(tmp, final_path)`.
4. If the destination exists:
   - Read the existing file.
   - If expired, overwrite in place (last writer wins; the relay file system is best-effort).
   - If not expired, return `(False, existing)`.

The fresh create uses `O_CREAT | O_EXCL`: two gateways that try to admit the same first message cannot both win. The peer that created the claim file keeps the claim; the loser re-reads the file and stands down. If the loser cannot parse the winner's file, the name still counts as held while the file is younger than the lease (fail closed: an unreadable claim can never quietly turn the cross-instance gate off). Past the lease window such a file heals: the next gateway removes it and retries the exclusive create under the same CAS rule. The expired-claim steal stays a bounded last-writer-wins window inside `os.replace`; the lease limits how long a stolen claim stays authoritative. A tree shared across hosts through a sync layer that does not preserve exclusive-create semantics needs a distributed guard. That is a known limitation: the lease bounds the damage, and the sanctioned topology is one bot token on one gateway.

### 5.3 Transfer (session reset)

When a routing transition replaces one session with another for the same channel (`/new`, auto-reset), the claim must move atomically from old session to new session. A naive release + re-acquire allows a concurrent inbound event to steal the channel in between.

```python
def transfer_channel_claim(
    db, platform, scope_id, chat_id, profile,
    old_session_id, new_session_id, new_instance_id=None,
    lease_seconds=300,
) -> bool:
    now = time.time()
    lease_expire = now + lease_seconds
    profile = _normalize_profile_for_claim(profile)
    def _tx(conn):
        conn.execute("""
            UPDATE channel_claims
            SET session_id=?, instance_id=COALESCE(?, instance_id),
                lease_expire=?, updated_at=?, thread_root_id=''
            WHERE platform=? AND scope_id=? AND chat_id=? AND profile=?
              AND session_id=?
        """, (new_session_id, new_instance_id, lease_expire, now,
              platform, scope_id, chat_id, profile, old_session_id))
        return conn.total_changes > 0
    return db._execute_write(_tx)
```

If `total_changes == 0`, the claim was stolen or released concurrently; the caller must fall back to a normal acquire.

Hook point: inside `_finish_route_transition` in `gateway/session_recovery.py`, when both `end_session_id` and `create_kwargs` are present, call `transfer_channel_claim` instead of `release_channel_claim`. If transfer returns `False`, the old claim was already gone; proceed with a fresh acquire.

## 6. Admission-Time Decision Table

Event arrives from platform adapter with `SessionSource` (platform, scope_id, chat_id, thread_id, profile).

The admission decision happens **before** `get_or_create_session`. The gate is inserted into `SessionStore._get_or_create_session_impl` at the start of Phase 2 (the lock-held phase), after `_ensure_loaded_locked()` and before the existing route checks run.

Insertion sketch:

```python
channel_key = (source.platform.value,
               source.scope_id or '',
               source.chat_id,
               _normalize_profile_for_claim(source.profile))

# Phase 0: claim gate (runs under _lock alongside route checks)
claim_acquired, claim = self._claim_gate_locked(source, channel_key, now)
if not claim_acquired and claim:
    # Decision table cases B-F
    return self._handle_blocked_claim(source, channel_key, claim)
```

`_claim_gate_locked` performs the atomic acquire (or returns the existing claim). `_handle_blocked_claim` implements the decision table below.

| Case | Condition | Action |
|---|---|---|
| **A** Acquire succeeds (ABSENT or EXPIRED stolen) | `acquired == True` | Continue to existing session logic. The new session will own the claim. |
| **B** Claim held by SELF, same session | `not acquired AND claim.instance_id == self_instance_id AND claim.session_id == session_being_created` | Re-entrant admission. Update `claim.thread_root_id` if `source.thread_id` changed, then continue. |
| **C** Claim held by SELF, different session, alive | `not acquired AND claim.instance_id == self_instance_id AND claim.session_id != new_session_id AND _is_session_alive(claim.session_id)` | A concurrent session for this channel exists. Inject the event into that session via `admit_session_input` (existing durable queue). Return the existing `SessionEntry`. |
| **D** Claim held by SELF, different session, dead | `not acquired AND claim.instance_id == self_instance_id AND claim.session_id != new_session_id AND NOT _is_session_alive(claim.session_id)` | The old session died without releasing. Steal the claim (atomic CAS) and continue to create the new session. |
| **E** Claim held by OTHER instance, lease valid | `not acquired AND claim.instance_id != self_instance_id AND claim.lease_expire >= now` | Another gateway holds the channel. For same-machine (shared `state.db`): inject via `admit_session_input` into the session if its `session_id` is known locally; otherwise queue. For cross-instance: this design does **not** implement remote passthrough forwarding — see §8.2. |
| **F** Claim held by OTHER instance, lease expired | `not acquired AND claim.instance_id != self_instance_id AND claim.lease_expire < now` | The other gateway crashed. Steal the claim (CAS) and continue to session creation. |

**Important invariant:** The exact `session_key` check for existing sessions stays in place. The claim lock is a coarse gate *above* that. A thread reply that matches an existing thread-keyed session still routes to that session (thread match via `_entries.get(session_key)` happening before the claim gate). The claim gate only prevents a *new* session from being created for the same channel when another claims it.

### 6.1 Thread-reply routing detail

The fold acts in enforce mode only. In monitor mode it logs the would-be fold and admits with pre-claim keying: monitor changes no routing (release §13).

When a thread reply arrives and the channel is claimed by session S:
- If the thread reply's `session_key` (which includes `thread_id`) already exists in `_entries`, route to that session (existing thread session). This preserves the existing behavior for already-opened thread sessions.
- If the thread reply's `session_key` does NOT exist, but the channel-level `session_key` (without `thread_id`) does exist and is alive:
  - **Decision for this design:** Route the thread reply into the channel-level session. The session's `origin` is updated with the new `thread_id` for outbound targeting. Do NOT mint a new thread-keyed session while the channel claim is held. The `session_key` of the routed session stays channel-scoped.
  - **Rationale:** This is the minimal change that fixes the double-session bug. The thread reply joins the same conversation as the parent DM. The adapter already targets outbound messages from `source.thread_id`.
- If neither the thread-keyed nor the channel-keyed session exists, the claim gate runs. On successful acquire, a new session is created with the **channel-level key** (without `thread_id`). The thread reply is the first message in that channel session. The `thread_id` is preserved in `source` for downstream rendering.

**`session_key` generation is unchanged.** `_generate_session_key` keeps its current logic. The claim gate in `_get_or_create_session_impl` intercepts thread replies before the normal key-lookup path and routes them to the channel-level session when it exists.

**Mutating `origin.thread_id` on a live session changes outbound targeting.** This is an acknowledged side effect: a session that starts in a DM and later receives a thread reply will send its next response to the thread. If the user then sends a top-level DM message, the session may reply in the thread instead. This is a minimal-fix tradeoff; exact per-message thread targeting is future adapter work.

## 7. Renew and Release

### 7.1 Renew

```python
def renew_channel_claim(db, platform, scope_id, chat_id, profile, session_id,
                        thread_root_id='', lease_seconds=300):
    now = time.time()
    lease_expire = now + lease_seconds
    profile = _normalize_profile_for_claim(profile)
    def _tx(conn):
        conn.execute("""
            UPDATE channel_claims
            SET lease_expire=?, updated_at=?, thread_root_id=?
            WHERE platform=? AND scope_id=? AND chat_id=?
              AND profile=? AND session_id=?
        """, (lease_expire, now, thread_root_id,
              platform, scope_id, chat_id, profile, session_id))
        return conn.total_changes > 0
    return db._execute_write(_tx)
```

Hook point: `gateway/run_turn_runner.py`, in `TurnRunner._complete_turn` or the adapter post-delivery callback after a successful model response is committed. The call uses the `SessionEntry.origin` fields for `(platform, scope_id, chat_id, profile)` and the `session_id` from the turn context.

### 7.2 Release

```python
def release_channel_claim(db, platform, scope_id, chat_id, profile, session_id):
    profile = _normalize_profile_for_claim(profile)
    def _tx(conn):
        conn.execute("""
            DELETE FROM channel_claims
            WHERE platform=? AND scope_id=? AND chat_id=?
              AND profile=? AND session_id=?
        """, (platform, scope_id, chat_id, profile, session_id))
    db._execute_write(_tx)
```

Hook points:
1. Normal session end (`_finish_route_transition` with `end_session_id` set and `create_kwargs` absent).
2. Session eviction (`evict_session` / `remove_by_session_id`).
3. Session delete (`delete` operation in `mutate_session`).

For each hook, the release is best-effort: if the row is already gone (stolen, expired), the delete is a no-op.

For the relay root, release unlinks `bot_relay/channel_claims/.../<profile>.json`.

## 8. Event Drain on Blocked Admission

When an event arrives and the channel claim is held by another session, the event must not be dropped. This design reuses the existing Hermes admission queue (`session_admissions`) instead of introducing a new in-memory queue.

### 8.1 Same-instance drain

If the claim holder is a session on the same `SessionAuthority`, the blocked event is injected into that live session using `admit_session_input` (the same durable queue that handles cron and queued controls). This is case C in the decision table.

If the claim holder's `session_id` is not in `authority.sessions` (a stale claim for a dead session), the caller steals the claim and creates or resumes a session normally (case D/F).

### 8.2 Cross-instance drain

Cross-instance event forwarding — sending an inbound platform event from gateway B to gateway A because A holds the claim — is **not implemented in this design**.

The relay connector contract (`website/docs/developer-guide/relay-connector-contract.md`) has no frame type for forwarding an arbitrary inbound event to another gateway instance. Adding it requires:
- A new relay frame type (e.g., `forward_inbound`) including `target_instance_id` and the normalized `MessageEvent` payload.
- Connector-side routing logic to deliver the frame to the gateway that currently holds the WebSocket for that instance.
- Gateway-side handling to treat the forwarded event as a normal inbound admission.

Until that frame type exists, cross-instance claim conflicts are handled by lease expiry: if gateway A crashes, its claim expires after 300 s, and gateway B steals it on the next event. Live cross-instance handoff is best-effort and may drop events during the overlap.

**Named dependency:** `bot_relay/forward_inbound` frame type (connector repo, not hermes-agent).

## 9. Crash Safety and Liveness

### 9.1 Lease expiry covers the crash case

If a gateway process crashes while holding a claim:
- The SQLite row stays in `channel_claims`.
- `lease_expire` is in the future immediately after the crash.
- No live process renews it.
- After `LEASE_SECONDS` (default 300 s), `lease_expire` is in the past.
- The next inbound event for that channel sees the expired claim, steals it (case F), and creates a new session.
- The old session's SQLite row in `sessions` and `session_admissions` remains (normal Hermes behavior; recovery or cleanup handles it separately).

### 9.2 Open-but-idle sessions do not hold a live claim

A session that exists in `state.db` and `_entries` but has had no turn for >300 s has an expired claim. A new inbound event for that channel will steal the claim and mint a fresh session. The old session row stays in `sessions` until the normal pruning interval.

This is intentional: the claim protects against concurrent active turns, not against creating a new session for an idle channel. If a long-running DM has been idle for days, a new thread reply should not be forced into a stale session; it gets a fresh session with no context. The user can `/resume` the old session explicitly if needed.

### 9.3 Same-machine, second-gateway-on-same-token

Two gateway processes on the same machine, sharing the same SQLite `state.db` and relay root, using the same `GATEWAY_RELAY_ID` / token:
- Both processes read and write the same `channel_claims` table. SQLite locking serializes their `_execute_write` operations. The compare-and-swap steal ensures only one wins.
- The `instance_id` field distinguishes which process acquired the claim. If process A crashes, process B sees an expired claim from A's `instance_id` and steals it.
- If both are alive, the first to `INSERT` wins; the second sees the existing claim and injects via `admit_session_input` (case C) or waits for expiry.

### 9.4 Different-machine, same token

Two gateways on different machines, connected to the same relay connector with the same bot token:
- Each gateway has its own SQLite `state.db`.
- Cross-instance claims live in the relay file tree (`bot_relay/channel_claims/`) only if the gateways share a filesystem (rare). For truly remote gateways, the relay connector must expose a claim query API.
- **This design does not implement a remote claim API.** The fallback is lease expiry: a gateway that crashes leaves an expired claim (if relay file tree is shared) or no claim at all (if not shared), and the surviving gateway acquires afresh.
- Live handoff between two healthy remote gateways on the same token is unsupported without the `forward_inbound` frame type.

### 9.5 Duplicate-credential profiles (multiplex blocker)

Hermes allows multiple profiles to use the same platform token (e.g., `bureaucratic-vp` and `default` both using `MATTERMOST_TOKEN`). Each profile has its own `SessionAuthority`, `state.db`, and `channel_claims` table. The claim lock is scoped to `(platform, scope_id, chat_id, profile)`.

**Result:** Two profiles sharing one token do **not** share a claim. Both can simultaneously create sessions for the same Mattermost channel, causing the same double-answer bug this design is meant to fix.

Fixing this requires either:
1. A credential-scoped claim layer above the profile layer, or
2. Migrating to one profile per platform token (multiplex consolidation).

**This design documents the residual risk and names the multiplex-migration blocker.** The claim lock eliminates double sessions within a single profile but does not protect against cross-profile duplication on shared tokens. Rollout environments with duplicate-token profiles must accept this limitation or schedule the multiplex migration first.

## 10. Implementation Plan

### 10.1 Files to touch

| File | Change |
|---|---|
| `hermes_state_schema.py` | Add `channel_claims` DDL next to existing `gateway_routing` and `session_admissions` DDL. |
| `hermes_state_runtime.py` | Add `acquire_channel_claim`, `transfer_channel_claim`, `renew_channel_claim`, `release_channel_claim`, `steal_channel_claim`. Use `SessionDB._execute_write` / `_read_one` patterns. |
| `gateway/session.py` | Add `_normalize_profile_for_claim`. In `_get_or_create_session_impl`, insert the claim gate in Phase 2 (after `_ensure_loaded_locked`, before route checks). Add `_handle_blocked_claim` decision table. |
| `gateway/session_recovery.py` | In `_finish_route_transition`: call `transfer_channel_claim` when both `end_session_id` and `create_kwargs` are present; call `release_channel_claim` when only `end_session_id` is present. |
| `gateway/session_mutations.py` | In delete/evict paths, call `release_channel_claim`. |
| `gateway/run_turn_runner.py` | Hook `renew_channel_claim` after successful turn delivery. |
| `tools/bot_relay.py` | Add `acquire_relay_claim`, `renew_relay_claim`, `release_relay_claim`, `read_relay_claim` using `bot_relay/channel_claims/` directory and JSON helpers. |

### 10.2 Minimal first slice

1. Add `channel_claims` table in `hermes_state_schema.py`.
2. Implement `acquire_channel_claim` with CAS steal in `hermes_state_runtime.py`.
3. Hook claim gate into `SessionStore._get_or_create_session_impl`:
   - Build `channel_key = (platform, scope_id, chat_id, profile)`.
   - `_claim_gate_locked` → `_handle_blocked_claim` per decision table.
   - Thread replies route into channel session when it exists.
4. Hook `transfer_channel_claim` and `release_channel_claim` into `_finish_route_transition`.
5. Hook `renew_channel_claim` into `run_turn_runner.py` post-delivery.
6. Write unit tests in `tests/gateway/test_session_claims.py`:
   - Acquire succeeds on empty channel.
   - Acquire fails when another session holds it.
   - Steal succeeds on expired claim.
   - CAS steal race: two concurrent steals, one wins.
   - Release deletes the row; transfer updates session_id.
   - Thread reply routes into channel session when claim is held.
   - Idle session (>300s) claim is stolen by new event.

### 10.3 Second slice (cross-instance)

1. Implement relay claim file ops in `tools/bot_relay.py` under `bot_relay/channel_claims/`.
2. On session reset, transfer the relay claim file atomically (rename with updated session_id).
3. Document `forward_inbound` frame type dependency for cross-instance live handoff.

## 11. Testing

| Test | What it proves |
|---|---|
| `test_acquire_empty_channel` | Happy path: new session gets the claim. |
| `test_acquire_blocked_live` | Second session for same channel is refused while claim is live. |
| `test_steal_expired` | A crashed gateway's expired claim is stolen. |
| `test_cas_race` | Two simultaneous steals on the same expired row: one wins, one retries. |
| `test_release_on_end` | Session end deletes the claim. |
| `test_transfer_on_reset` | Session reset moves claim atomically. |
| `test_renew_extends_lease` | A live session keeps its claim. |
| `test_idle_claim_expires` | An idle session's claim expires; new event creates fresh session. |
| `test_thread_reply_routes_to_channel` | A DM thread reply joins the existing DM session, does not split. |
| `test_two_gateways_same_db` | Two gateway processes racing on the same SQLite DB: serializes correctly. |

## 12. Open questions resolved

1. **Should the claim be per-profile or global?** Per-profile. A profile-scoped agent is the unit that processes a channel. Two profiles on the same channel are independent.
2. **What if a thread reply arrives for an existing thread-keyed session that predates the claim-lock?** The exact `session_key` match still wins. The claim lock only blocks *new* session creation. The session migration plan should eventually collapse thread-keyed sessions into channel-keyed ones.
3. **What about group channels with threads?** Same behavior: one session per channel. Thread replies route into that session. The agent sees `source.thread_id` and can switch context internally.
4. **Backward compatibility?** The change is additive: a new table, a gate before session creation. Existing sessions without a claim row continue to work (they will acquire the claim on their next turn or inbound event). No migration of existing `state.db` rows is needed.
5. **What about open-but-idle sessions?** They are not protected by an active claim. After 300 s of inactivity, the claim expires. New events steal and start fresh sessions. Old sessions remain in `sessions` for explicit `/resume`.
6. **What about duplicate-credential profiles?** This design does not prevent cross-profile double sessions on shared tokens. The multiplex-migration blocker is named in §9.5.

## 13. Release configuration (`gateway.claim_lock`)

The claim gate rolls out behind one config block:

```yaml
gateway:
  claim_lock:
    enabled: true
    mode: monitor   # monitor | enforce
```

- `enabled: false` is the kill switch: no claim acquire, no thread fold, no renewal, no stand-down. This is the pre-claim-lock behavior exactly.
- `mode: monitor` (the default) keeps both claim stores maintained (intra-host SQLite, the cross-instance relay surface) and logs each would-have-blocked admission. It refuses no one and it changes no routing: the thread fold, the route-into-holder move and the queue/stand-down acts run in enforce only. A mode value that is not one of `monitor`, `enforce` fails open to monitor with a warning, so a config typo cannot arm enforcement.
- `mode: enforce` acts on live claim conflicts as sections 5 and 6 describe: route into the holder, stand down to the owning gateway, refuse the double-admit.

`scope` and `lease_s` stay with the cross-instance claim ops (`tools/bot_relay.py`) and are not read from this block yet.
