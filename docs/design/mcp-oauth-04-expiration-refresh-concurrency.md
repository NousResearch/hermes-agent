# MCP OAuth Chunk 4 — Expiration and Refresh Concurrency

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: transactional reauthorization (Chunk 3)
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §4.2, §4.3, §5.1, §7.1, §8.3, §8.4, §12.1, §15
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-5 wall-clock guard, F-8 revision envelope; F-1, F-2, F-7)

## Purpose

Make access-token expiration and refresh deterministic across restarts and safe across concurrent Hermes processes. A stale refresher must not overwrite credentials produced by a newer refresh or reauthorization.

## Token time model

Extend the token record with:

```python
@dataclass(frozen=True)
class OAuthTokenRecord:
    access_token: str
    refresh_token: str | None
    token_type: str
    scopes: tuple[str, ...]
    accepted_at_utc: datetime
    expires_at: datetime | None
    original_expires_in: int | None
```

When a response supplies `expires_in`:

```text
expires_at = accepted_at_utc + expires_in
```

Three values are persisted: `accepted_at_utc`, the absolute `expires_at`, and
`original_expires_in` — the relative lifetime exactly as the provider returned it, in seconds.
`original_expires_in` is **never recomputed on load**; it is authoritative for the wall-clock
plausibility guard below. A refresh response that supplies a new `expires_in` replaces it; one that
omits it leaves the stored value (parallel to the refresh-token merge rule).

UTC wall time determines validity across restarts. Monotonic time measures in-process waits,
request deadlines, and retry delays.

## Expiration classification

```python
class TokenExpirationState(Enum):
    VALID = "valid"
    REFRESH_DUE = "refresh_due"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
```

**Wall-clock plausibility guard (architecture §4.3).** Run before classifying from `expires_at`,
because `expires_at` is wall-clock and a backward step (NTP correction, manual change, VM snapshot
restore) would make a token look valid longer than it is:

```text
if original_expires_in is None:                              -> unknown
if now < accepted_at_utc:                                    -> unknown
elapsed = now - accepted_at_utc
if elapsed < 0 or elapsed > original_expires_in * CLOCK_SLACK:  -> unknown
otherwise classify normally from expires_at
```

`CLOCK_SLACK = 2.0` — wide enough to absorb legitimate NTP drift, narrow enough to trip on a gross
step. The guard only *demotes* to `unknown`; it never shortens `expires_at`, so it cannot cause a
forced-refresh storm. Residual risk: a small backward step within the slack band still passes;
the one-bounded-recovery-on-rejection path is the backstop.

For known expiration (guard passed):

```text
token_lifetime = expires_at - accepted_at_utc
refresh_window = min(60 seconds, token_lifetime × 10%)
refresh_due_at = expires_at - refresh_window
```

- Before `refresh_due_at`: `valid`.
- At/after `refresh_due_at` but before `expires_at`: `refresh_due`.
- At/after `expires_at`: `expired`.
- No trustworthy expiry, or guard demoted: `unknown`.

Invalid or non-positive lifetimes are immediately expired; their safety window is zero.

## Transitional revision envelope

Chunk 5 introduces one physical bundle, but Chunk 4 needs logical revisions first. The revision is
embedded **inside the token record's own JSON envelope** (`<safe-server>.json`) — not in a separate
manifest. It is written by the same atomic `os.replace` that writes the token record, so no crash
window can pair a revision with token state it does not describe (architecture §8.4). The client
and metadata files are unchanged; the `revision` key is additive and ignored by an older reader.
This is the shape Chunk 5's bundle envelope continues (`revision` at the top level).

```python
@dataclass(frozen=True)
class StoredState:
    state: LegacyOAuthState
    revision: str            # sourced from and written to <safe-server>.json
```

`StoredState` is the Chunk 2–4 transitional form of architecture §4.2's `StoredBundle`; Chunk 5 replaces it once the bundle is unified.

Every successful credential mutation creates a random 128-bit revision, encoded as lowercase hex.
The revision contains no token-derived material.

## Compare-and-swap API

Add to the store facade:

```python
def compare_and_swap_tokens(
    identity: OAuthIdentity,
    *,
    expected_revision: str,
    tokens: OAuthTokenRecord,
) -> StoredState: ...
```

Under a short cross-process mutation lock, the backend reads the current revision from
`<safe-server>.json`, rejects a mismatch, then writes the merged token record and its new revision
as a single atomic `os.replace` of that file, and releases the lock. CAS in Chunk 4 is token-record
scoped; Chunk 5's versioned bundle backend generalizes it to whole-bundle CAS as architecture
§5.1's `compare_and_swap`.

## Refresh coordination

Within one process, the provider manager continues to deduplicate refresh/401 work per server. Cross-process correctness comes from CAS, not the in-process future map.

The stored revision is consulted only at decision points — before a refresh, on CAS, during 401 recovery, and on rebuild/status — never on the per-request read path (architecture §8.3, §12.1). Chunk 4 still watches token-file mtime for cache invalidation; replacing that with revision watching and adding the 10 s in-memory revision TTL (with backoff and last-known-good fallback) is Chunk 5.

Refresh algorithm:

```text
1. Load state + revision.
2. Classify expiration.
3. If refresh is required, call the token endpoint.
4. Merge response with loaded token record.
5. Preserve old refresh_token if response omitted it.
6. CAS using loaded revision.
7. On conflict, reload and evaluate newer state.
```

Conflict handling:

- Different valid access token: retry the resource request once with it.
- Newer refreshable but expired state: one bounded refresh retry.
- Identity/issuer change: return typed mismatch/reauthorization result.
- Missing credential: return `credential_not_found`; do not recreate silently.

## Request behavior

- `valid`: send request.
- `refresh_due`: refresh before request when refreshable.
- `expired`: refresh before request or return `reauthorization_required`.
- `unknown`: send request; recover from rejection once.

An authentication rejection causes one coordinated reload. If it finds a newer valid token, retry once. Otherwise refresh once when possible. There is no recursive or unbounded 401 loop.

A credential committed with `probe=deferred` (Chunk 3) reaches the runtime unverified by design. Its first authentication rejection is expected: it routes straight into this reload/refresh recovery and surfaces `reauthorization_required` if that fails, without the elevated logging reserved for a fresh-token rejection at commit (architecture §7.1). The `HTTP 5xx/timeout` vs `invalid_grant` vs `401` branches in "Provider response rules" are the refresh-side application of the same definitive/indeterminate kind classification Chunk 2 introduced (`classify_outcome`).

## Authorization interaction

Explicit successful reauthorization holds the administrative lock and intentionally replaces active credentials. A refresh started from the previous revision loses CAS after reauthorization commits.

A failed staged reauthorization never changes the revision, so concurrent refresh proceeds normally.

## Provider response rules

- Omitted `refresh_token`: retain loaded refresh token.
- Returned rotated refresh token: replace it atomically with access token.
- Supplied `expires_in`: recompute `expires_at` and replace `original_expires_in`. Omitted `expires_in`: retain the stored `original_expires_in` and the derived `expires_at`.
- `invalid_grant`: return `reauthorization_required`, preserve stored state for diagnostics.
- HTTP 5xx/timeout: preserve state and return retryable failure.
- Malformed response: preserve state and return typed invalid-response error.

## Tests

Use an injectable UTC clock and real temporary backend:

- `expires_in` converts to exact `expires_at`.
- `original_expires_in` is persisted and is not recomputed on load; a refresh with a new `expires_in` replaces it, one without leaves it.
- Long-lived token uses 60-second window.
- Short-lived token uses ten-percent window.
- Boundary instants classify correctly.
- Wall-clock guard: `now` before `accepted_at_utc` → `unknown`; elapsed beyond `original_expires_in × CLOCK_SLACK` → `unknown`; NTP-scale drift within the slack band classifies normally (no false demotion).
- Unknown lifetime remains usable until rejection.
- Omitted refresh token is preserved.
- Rotated refresh token replaces old token.
- Two processes refresh one revision; one CAS wins.
- Crash between the token-record temp write and `os.replace` leaves the old complete record (old revision + old state); crash after `os.replace` leaves the new complete record. No revision/state mismatch either way.
- No emitted field — log line, `mcp_oauth.refresh_conflict` event, or `OAuthCredentialStatus.revision_prefix` — contains more than 8 hex characters of a revision.
- Stale refresh loses to successful reauthorization.
- Refresh during failed reauthorization succeeds.
- 401 recovery retries at most once.
- A `probe=deferred` credential with an invalid token surfaces `reauthorization_required` on first use without fresh-token-rejection logging.
- Wall-clock adjustment does not change monotonic timeout behavior.

## Demonstration

Start two worker processes with the same expired credential and revision. Release both token-endpoint responses together. Show one commit succeeds, one receives `revision_conflict`, and both subsequent requests use the winning token.

Separately: with a `valid` token, step the injectable clock back by more than `original_expires_in`. Show the next classification returns `unknown` (not `valid`), the request is still sent, and a rejection triggers one bounded recovery.

## Non-goals

- Do not yet migrate to a single physical bundle. Adding the additive `revision` key to `<safe-server>.json` is not that migration — the client and metadata files are untouched.
- Do not replace token-file mtime watching with revision watching, or add the in-memory revision TTL (Chunk 5).
- Do not add provider-specific refresh-window configuration without a demonstrated provider requirement.
- Do not re-anchor `expires_at` on load; the guard only demotes to `unknown` (never shortens).
- Do not delete credentials on `invalid_grant`.
- Do not add Keychain.

## Completion criteria

- Expiration behavior survives restart.
- A gross backward wall-clock step demotes classification to `unknown` rather than extending apparent validity.
- Proactive refresh follows the documented safety window.
- Stale writes cannot replace newer credentials.
- A crash during a revisioned token-record write leaves a revision/state-consistent record.
- No logged or diagnostic revision prefix exceeds 8 hex characters.
- Refresh-token rotation and omission are correct; `original_expires_in` tracks the current grant.
- Authentication rejection recovery is bounded.
