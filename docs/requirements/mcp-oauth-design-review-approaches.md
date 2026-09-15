# MCP OAuth Credential Store - Approach Analysis for Design-Review Findings

Status: Approach decisions closed (F-0..F-8); ready to fold into architecture + design docs
Audience: Hermes maintainers and contributors
Scope: Candidate approaches for each finding in `mcp-oauth-design-review-findings.md`
Inputs: architecture `../architecture/mcp-oauth-credential-store-architecture.md`; designs `../design/mcp-oauth-0[1-7]-*.md`; plan `../plans/2026-09-01-mcp-oauth-chunk-1-implementation-plan.md`
Method: superpowers brainstorming (architectural path, "exploring approaches" stage). No code written. This document proposes; it does not amend the architecture or design.

---

## How to read this

Each finding gets 2-3 approaches, trade-offs, and a recommendation. Recommendations are opinionated;
the "My bias" section at the end states the priors those opinions rest on so you can discount them.

A cross-cutting inconsistency surfaced during research and is recorded as F-0.

---

## F-0. Architecture/plan identity-type mismatch (new, blocks F-3)

`architecture §4.1` declares `OAuthIdentity.profile_id: str` (a digest). The Chunk 1 plan and
`design/mcp-oauth-01` declare `OAuthIdentity.profile_home: Path` and compute the SHA-256 digest
later, at backend-key construction. These are not reconcilable as written.

- Disposition: Resolve before F-3. The digest-derivation rule cannot be specified until the type
  that feeds it is fixed.

### Decision (from the existing codebase): use `profile_home: Path`

The current code already has a settled answer, and it is the plan/design-01 shape, not the
architecture's.

Evidence:

- `hermes_constants.get_hermes_home() -> Path` is "the single source of truth" for profile home;
  every caller imports it and works with a `Path`.
- `hermes_constants.hermes_home_key(path) -> str` is the established canonical-identity helper:
  `os.path.normcase(str(candidate.expanduser().resolve(strict=False)))`. It exists specifically so
  "runtime registries" can isolate profile-scoped entries by a stable key. It is a resolved string,
  not a hash.
- `MCPOAuthManager._key()` (`tools/mcp_oauth_manager.py:681`) already keys the provider cache on
  `(str(home.expanduser().resolve(strict=False)), server_name)` - the same pattern, minus
  `normcase`.
- `HermesTokenStorage` (`tools/mcp_oauth.py:468`) keys purely on the `hermes_home` path plus
  `server_name`; there is no digest anywhere in the current OAuth storage path. The digest is a
  new concept the architecture introduces only to make filesystem/Keychain identifiers safe.

Conclusion:

1. The domain object keeps `profile_home: Path` (plan Step 3, design-01). It is already under
   test (`test_identity_canonicalizes_profile_and_server_url`) and matches `_key()`.
2. `architecture §4.1` is rewritten so `profile_id` is not a field but a *derived backend key*:
   `SHA-256(canonical_identity)` where `canonical_identity` is built from
   `profile_home` (canonicalized) + normalized `server_url` + `server_name`. This is what
   `§4.1`'s own second paragraph already describes ("The backend key uses a SHA-256 digest of the
   canonical identity"); only the first paragraph's `profile_id: str` field contradicts it.
3. Canonicalization reuses `hermes_constants.hermes_home_key()` rather than a private rule, which
   also pre-answers F-3. Align `MCPOAuthManager._key()` to call the same helper (it currently
   omits `normcase` - a pre-existing inconsistency worth closing in the same change).

Path B (plan/design-01) selected. No maintainer decision required; the codebase already committed
to it.

---

## F-1. Keychain revision-probe cost on the hot path

Question the reviewer raised: how often is the revision probed, and is there an in-memory
TTL/backoff, so an auth-sensitive request does not cost a `security` subprocess spawn?

Current text: `§12.1` says the provider cache "asks the lifecycle service whether the revision
changed" before an auth flow; `§8.3` lists Load as a "brief read", no revision check. The cadence
of the probe is unspecified, which is the gap.

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Specify probe points explicitly: only on provider-rebuild decisions (pre-authorization-flow, post-401 recovery, explicit refresh/status). Never per resource request. State "no revision probe on the runtime read path" as an invariant. | Removes the per-request cost by definition. Documentation-only. Matches what `§12.1` already implies. | Between rebuild points, a process can hold a stale bundle. Acceptable: CAS still catches stale *writes*; a stale *read* only means a slightly late refresh, recovered by 401 handling. |
| B | Add a short in-memory probe TTL (default ~10s, not configurable) plus exponential backoff on probe failure with last-known-good fallback. | Bounds staleness to a known number. Survives a briefly unavailable backend without failing requests. | One more cache with its own invalidation. TTL is a guess until measured. |
| C | Cheap revision hint per backend: file backend uses `stat` mtime/inode (no read); Keychain has no cheap equivalent, so pair with kqueue/FSEvents on the lock directory for push invalidation. | Near-zero cost on the file backend. Push invalidation is more timely than any TTL. | Keychain still needs a real probe. Event watchers are platform code and a known source of flakiness. High complexity for a marginal gain. |
| D | Replace the `security` subprocess with Security.framework bindings for load/probe. | No fork/exec; probe becomes an in-process XPC round trip to `securityd`. | Native binding dependency and build/packaging work. Still not free. Larger blast radius. |

Recommendation: **A + B together.** Define the probe points (A) and add a 10s in-memory TTL with
backoff and last-known-good fallback (B). Note D as the escalation path if gateway profiling shows
subprocess cost dominating a hot path; do not build D speculatively. Reject C - the file-backend
win is real but the Keychain half stays unsolved and the watchers cost more than they save at this
stage.

Target sections: `§8.3` (add probe-cadence row), `§12.1` (state the invariant and TTL),
Chunk 1 plan provider-cache step.

---

## F-2. Probe failure conflates auth rejection with transient transport failure

Current: `§6.3` and `design/mcp-oauth-03` step 7 require "the configured authentication probe" to
succeed before commit. A network timeout forces a full browser redo of a valid token.

### Findings from the existing codebase

1. **"The configured authentication probe" today is `_probe_single_server()`**
   (`hermes_cli/mcp_config.py:278`): a full MCP connect -> `initialize` -> `tools/list` ->
   `shutdown`, wrapped in `asyncio.wait_for(connect_timeout)`, followed by an
   `_oauth_tokens_present()` disk check (`mcp_config.py:404`). `hermes mcp login` and
   `hermes mcp reauth` both go through `_reauth_oauth_server()` (`mcp_config.py:810`); the
   dashboard re-auth path in `web_server.py` is deliberately kept identical (comment at
   `mcp_config.py:843`). It is a real network round-trip that can fail on transport, not a local
   check.

2. **Current behavior already is the frustrating case the reviewer describes.** In
   `_reauth_oauth_server`, *any* probe exception falls into `except Exception` ->
   `_error("Authentication failed")` -> returns `False` (`mcp_config.py:897-907`), with no
   distinction between 401 and a timeout. Under today's legacy storage the SDK has usually already
   written the token to disk mid-flow, but the next `login`/`reauth` run calls
   `get_manager().remove(name)` first (`mcp_config.py:831`), wiping it - so the user redoes the
   browser flow. Chunk 3 makes this stricter (in-memory staging, commit gated on the probe), so
   without a taxonomy the transient-failure redo goes from "usually" to "always" - a regression.

3. **Status-class discrimination is already an established pattern here.**
   `_maybe_flag_poisoned_client` acts only on `status in (400, 401)` (`mcp_oauth_manager.py:458`);
   the refresh-failure branch keys on `400/401` (`:438`). `_unwrap_exception_group`
   (`mcp_config.py:420`) exists specifically to surface `"401 Unauthorized"` out of anyio
   `BaseExceptionGroup` wrappers so callers can tell auth rejection from transport failure.
   Classifying a probe outcome by status / exception type reuses this, it is not a new mechanism.

4. **The runtime 401-recovery path is mature and its contract is exactly "discover a bad token on
   first use".** `MCPOAuthManager.handle_401()` (`mcp_oauth_manager.py:862`): thundering-herd
   dedup via `pending_401` futures, checks whether disk changed (external refresh), checks
   `can_refresh_token()`, returns `True` -> caller reconnects and retries, `False` -> surface
   `needs_reauth` to the model. Backed by the `async_auth_flow` override (`:513`), `_refresh_token`
   (`:196`), and `_maybe_flag_poisoned_client`. "Commit and let the runtime discover" routes into
   this existing, tested path.

5. **No transient-vs-permanent classifier exists in `tools/mcp_oauth.py` today.**
   `humanize_oauth_registration_error` (`:1830`) is display-only. Approaches A and B both need a
   small new classifier helper - but it sits on top of `_unwrap_exception_group` plus httpx
   exception-type / status-code inspection, all already in use.

6. **Config-surface resistance is codified locally, not just in `§11`.** `_reauth_oauth_server`
   hardcodes a 315s connect-timeout floor rather than exposing a knob (`mcp_config.py:857`).
   Approach C cuts against a consistent local convention.

### Options

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Probe-outcome classifier with three results. `authenticated` -> commit. `rejected` (HTTP 401/403; bodies with `invalid_token` / `invalid_grant`) -> abort, return `reauthorization_required`. `indeterminate` (httpx `ConnectError` / `ConnectTimeout` / `ReadTimeout` / `PoolTimeout`, HTTP 5xx, DNS failure, `asyncio.TimeoutError` from the `wait_for`) -> one retry after ~2s, then commit the staged bundle and emit `mcp_oauth.reauth_committed` with `probe=deferred`. The mature `handle_401` path covers a token that turns out bad on first real use. | Never discards a good token for a transport blip. Routes into a recovery path that already exists and is tested (finding 4). Keeps a hard, fast abort for real rejection. Classifier reuses `_unwrap_exception_group` + the 400/401 precedent (findings 3, 5). No new config (finding 6). | Commits a token never positively confirmed against the server. Mitigated: it is the output of a completed authorization-code exchange against the discovered/configured issuer, and `handle_401` will catch a genuinely bad one. |
| B | Same classifier, but `indeterminate` -> bounded retry loop (2-3 attempts, backoff kept inside the admin-lock timeout budget), then abort with a typed transient error if still failing. | Preserves the "positively verified before commit" invariant. | Holds the browser flow and the admin lock longer. Still forces a full redo when the server outage outlasts the retry budget - the reviewer's case is reduced, not removed. |
| C | `mcp.oauth.probe_failure_policy: strict\|lenient` config key. | Operators choose per environment. | New config surface `§11` and local convention (finding 6) both resist. A knob whose wrong setting is either needless redos or silent bad-token commits. |

### Determination

**Approach A.** The codebase evidence is decisive rather than a matter of taste:

- Approach A's "let the runtime discover it" is not hand-waving - `handle_401` is a mature,
  deduplicated recovery path whose documented contract is exactly this (finding 4).
- The classifier it needs is small and follows an established status-class pattern (findings 3, 5).
- Current behavior is already the frustrating case (finding 2), and Chunk 3 would harden it into a
  guaranteed redo; A is the fix that keeps Chunk 3's staging benefit without that regression.
- C is ruled out by both `§11` and the local 315s-floor convention (finding 6).

Keep the single ~2s retry so an instantly-failing probe (misconfigured endpoint, immediate
connection refused) still aborts fast rather than committing.

Approach A is approved. Approach B remains a documented fallback if the "positively verified before
commit" invariant is later judged worth a guaranteed browser redo during any upstream outage - a
policy preference, not a technical blocker.

### Why A balances the competing biases

Three biases act on this decision and they do not pull the same way:

- **Reviewer bias (webtecnica):** availability-leaning, from their own experience with
  credential-pool rotation and WAF/403 races. Pulls *lenient* - do not make the user redo work for
  a transient blip.
- **Original-developer / experiential bias (this codebase):** an accretion of "the happy path
  lied to us" fixes - `_oauth_tokens_present()` exists because a clean probe was a false positive
  (Google Drive serves `tools/list` unauthenticated); the 315s floor, poisoned-client detection,
  CIMD fallback, `update_token_expiry` are each a real bug. Pulls *strict; verify concretely; fail
  visibly, not silently*.
- **Good-pattern bias (circuit breaker):** when a downstream dependency is unhealthy, do not
  hammer it, preserve state, fail fast, recover on a schedule.

Approach A removes the strict-vs-lenient axis and replaces it with an observable signal
(`response.status_code` / exception type), which is present at every failure point today and
discarded - `_handle_token_response` raises the same `OAuthTokenError` for 400 and 500 alike
(`tools/mcp_oauth.py:1238`).

- Honors the reviewer: the one case where a cryptographically-good token is discarded - dependency
  *unavailable* after a completed code exchange - is preserved. Leniency is applied exactly where
  it is valid.
- Honors the original developer: `authenticated` still requires the staged token to exist
  ("clean probe != proof" survives); `rejected` aborts loud; `probe=deferred` is written to
  diagnostics so a hopeful commit is visible, not hidden; `handle_401` is reused rather than
  duplicated.
- Honors the pattern: `indeterminate` is the circuit breaker's half-open decision at commit time.
  The retry budget is a single ~2s step, never a loop, so the per-identity admin lock is not held
  across backoff.

### The flow is five network hops, not one - two-axis classification

`authorize()` touches five endpoints, only the last of which holds a staged token:

| Step | Endpoint | Staged token yet? |
|------|----------|-------------------|
| 1. PRM/ASM discovery | MCP server + AS `.well-known` | no |
| 2. Dynamic client registration | AS registration endpoint | no |
| 3. Authorization (browser) | AS authorization endpoint | no |
| 4. Token exchange | AS token endpoint | no |
| 5. Probe | MCP server (`initialize`/`tools/list`) | yes |

Classify by *position* (pre-token vs post-token) x *kind* (definitive rejection vs indeterminate):

**Pre-token (steps 1-4) - nothing to commit; "commit-deferred" does not apply.**

- Definitive (400 `invalid_grant`, `invalid_client`, unsupported registration): abort ->
  permanent typed error ("fix config" / "re-authorize").
- Indeterminate (500/502/503/504, timeout, connection reset at an AS endpoint): one immediate
  retry of that sub-step, then abort -> *transient* typed error ("authorization server
  unavailable, retry shortly", not "authentication failed"). The step-3 authorization code is
  single-use and expires in under 60s (RFC 6749 4.1.2), so a sub-second retry may catch it but a
  browser redo after that is protocol-mandated, not strictness - the reviewer's leniency
  principle does not reach here because there is no artifact to preserve.
- 429 at the token endpoint: respect `Retry-After`; if it exceeds a few seconds the code is dead
  -> abort transient and surface `Retry-After`. Do not trigger a fresh authorization against a
  rate-limited AS.

**Post-token (step 5) - staged token exists, valid for its full lifetime.**

- `authenticated` (staged token present AND probe succeeded): commit.
- `rejected` (401/403 from the MCP server, `invalid_token` body): one ~2s retry first - a fresh
  token 401ing is often AS-to-resource-server replication lag or small clock skew, which 2s
  clears. Still 401 -> abort loud (`reauthorization_required` + "check clock sync / token
  audience" hint); another browser round will not fix a genuinely refused fresh token.
- `indeterminate` (5xx / timeout / 429 from the MCP server): one ~2s retry -> still failing ->
  commit with `probe=deferred`. `handle_401` covers a token that turns out bad on first real use.

### Stakeholder accounting

| Stakeholder | What the algorithm guarantees |
|-------------|-------------------------------|
| User (CLI, dashboard, TUI - one path; `web_server.py` kept identical) | A truthful per-class outcome ("retry shortly" / "fix config" / "re-authorize"), not today's uniform "Authentication failed". Browser redo only when the protocol requires it. |
| MCP server | Its 500/429 is never read as "credential is bad" and never triggers a re-auth storm. One bounded retry, then defer. |
| AS / token endpoint | A 500 there does not immediately trigger a new authorization (new code request = more load). 429 respected. |
| Concurrent Hermes processes | The per-identity admin lock is held across a single ~2s retry step, never a loop - siblings are not starved. |
| Runtime (post-commit) | Must treat a `probe=deferred` token's first 401 as expected - straight to refresh / `needs_reauth`, no "anomalous fresh-token rejection" logging. |
| Operator / debugger | `probe=deferred` in `hermes mcp credentials status` and `mcp_oauth.reauth_committed` distinguishes "committed verified" from "committed hopeful". No silent state. |

### Target sections / code

- `architecture §6.2` - "Failure behavior" gains the pre-token position/kind split: definitive vs
  indeterminate, and indeterminate returns a *transient* typed error.
- `architecture §6.3` - replace "The MCP server completed the configured authentication probe" with
  the post-token three-outcome rule (`authenticated` / `rejected` / `indeterminate`), each with
  its one ~2s retry.
- `architecture §14` - add `authorization_endpoint_unavailable` (transient, pre-token) as a
  surfaced code distinct from `authorization_timeout`. `reauthorization_required` already covers
  the definitive cases. `probe_indeterminate` is an internal classifier result, not a surfaced
  error.
- `architecture §15` - add `probe` field (`authenticated` / `deferred`) to
  `mcp_oauth.reauth_committed`; add `mcp_oauth.reauth_aborted` reason values
  (`rejected` / `endpoint_unavailable`).
- `architecture §7` / `design/mcp-oauth-04` - `handle_401` must treat a `probe=deferred` token's
  first rejection as expected (no anomaly logging).
- `design/mcp-oauth-03` - "Flow" steps 5-8 and "Commit validation": classify by position and kind,
  not pass/fail.
- `architecture §17.2` / `design/mcp-oauth-03` tests - add: token-endpoint 500 -> one retry ->
  transient error, active bundle untouched; token-endpoint 400 `invalid_grant` -> permanent error;
  probe 401 -> one ~2s retry -> abort loud; probe 503 -> one ~2s retry -> staged bundle commits
  with `probe=deferred`; committed-deferred token that is actually invalid -> `handle_401`
  surfaces `needs_reauth` on first use without anomaly logging; 429 with `Retry-After` at the
  token endpoint -> transient error surfaces the interval.
- New helper (likely `tools/mcp_oauth_store/lifecycle.py` or alongside the staged adapter):
  `classify_outcome(stage, exc_or_response) -> Outcome` where `stage` is `pre_token` | `probe`,
  built on `_unwrap_exception_group` + httpx exception-type / status-code inspection. Single seam;
  all three surfaces inherit it.

---

## F-3. `profile_id` canonicalization under symlinks

Risk: CLI and gateway reach the same profile home through different path spellings (`~` vs
absolute, trailing slash, `..`, a symlinked parent, macOS `/var` -> `/private/var`). If the
canonical form differs, the identity digest differs and credentials orphan silently.

### Findings from the existing codebase

1. **`get_hermes_home()` returns the raw, unnormalized env value.** `_hermes_home_from_env()`
   (`hermes_constants.py:62`) is `Path(os.environ["HERMES_HOME"])` with no `expanduser` and no
   `resolve`. Launchers disagree on spelling: the systemd/launchd unit bakes in
   `str(get_hermes_home())` captured at install time (`gateway.py:4075`), the CLI may inherit a
   shell-expanded `~/.hermes` or nothing (platform default), Docker sets `HERMES_HOME=/opt/data`
   (`gateway.py:2751`). So two live Hermes processes genuinely hold different spellings of the
   same home.

2. **Today it "works" only because the filesystem resolves the paths, not Hermes.**
   `HermesTokenStorage` writes to `Path(get_hermes_home()) / "mcp-tokens"` with the raw spelling
   (`tools/mcp_oauth.py:200`). `~/.hermes` and `/Users/x/.hermes` are the same inode, so the OS
   lands both at the same directory. The moment a digest is taken of the path *string* (the new
   architecture), that equivalence is gone unless Hermes canonicalizes first. This is why F-3 is a
   real regression risk, not a theoretical one.

3. **The codebase already has a settled canonicalization helper, and it is battle-tested.**
   `hermes_constants.hermes_home_key()` (`:142`) =
   `os.path.normcase(str(candidate.expanduser().resolve(strict=False)))`. It is *the* profile
   scope key for plugins (`plugins.py:3745`, `:6237`), tool config (`tools_config.py:3130`),
   the registry (`registry.py:486`), and the browser tool (`browser_tool.py:807`).
   `profile_matches_home()` (`profiles.py:402`) independently does the same
   `expanduser().resolve(strict=False)` equality check - added for security defect #91583 (a
   gateway serving another profile's config). The pattern is proven; it is just copy-pasted rather
   than centralized, and inconsistent on `normcase` (`_key()` and `profile_matches_home` omit it).

4. **The codebase also has a competing identity pattern - inode tuples - used deliberately for the
   opposite purpose.** `checkpoint_manager.py` records `(st_dev, st_ino)` of the workdir parent
   and treats a change as "the directory was swapped under us" (`:547`, `:1739`);
   `session_recovery.py:157` and `web_server.py:651` do the same. Inode identity exists precisely
   to *detect* when a path now points at a different directory.

5. **Named profiles carry a path-independent name identity already** (`profiles/<name>/`,
   `normalize_profile_name` "the canonical profile id used on disk", `profiles.py:321`), but it is
   not threaded down to the OAuth storage layer, and the default profile's identity is still just
   its path.

### What the digest actually feeds

The identity digest becomes durable on-disk filenames (`mcp-credentials/v1/<digest>.json`),
Keychain account strings, and lock filenames (`runtime/mcp-oauth-locks/<digest>.*`). The
properties that matter: stable across process restarts; computable before the credential file
exists; identical across two processes on one machine reaching the same directory; **survives the
profile directory being deleted and recreated** (backup restore, `hermes profile` reset) so a
restored profile keeps its credentials.

### Options

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Canonicalize the path via a **centralized** helper with `hermes_home_key()` semantics (`normcase(str(expanduser().resolve(strict=False)))`); digest = SHA-256 over that key + normalized server URL + server name. Migrate the copy-paste sites (`MCPOAuthManager._key()`, `profile_matches_home`) to the shared helper, resolving the `normcase` inconsistency. Parametrized contract test: `{tilde, trailing slash, embedded .., symlinked parent, /var vs /private/var}` -> identical digest across two independent calls. | Reuses the codebase's proven profile-scope pattern (finding 3). `resolve(strict=False)` collapses symlinks and `..` when the dir exists - which it does by OAuth time. **Follows the path**: a deleted-and-recreated profile keeps its credentials. Centralizing removes three copies and one real bug (the `normcase` drift). | Case-insensitive-FS spellings (`~/.hermes` vs `~/.Hermes`) still hash differently on macOS/Linux (`normcase` is a no-op off Windows). Pathological; matches how `_key()` and `profile_matches_home` already behave. Cross-namespace (container vs host, same volume, different mount path) is not solved. |
| C | Inode-tuple identity: digest over `f"{st_dev}:{st_ino}"` of the profile home + server URL + name, matching the `checkpoint_manager` precedent (finding 4). No new file. | Immune to every path-spelling, symlink, and case edge. No bootstrap file to lose. | **Orphans every credential when the profile dir is deleted and recreated** - `st_ino` changes (finding: this is exactly the failure F-3 fears, re-triggered). Unavailable on some Windows filesystems (checkpoint_manager carries a fallback for this). Still does not solve cross-namespace (different `st_dev` per namespace). |
| D (deferred) | Explicit operator-set `mcp.oauth.credential_identity: <stable-string>` in profile config, used verbatim (hashed) when present, auto-derivation (A) otherwise. | The only thing that actually works for cross-namespace / multi-host shared-volume topologies. Explicit, no guessing. | Config surface `§11` resists; only justified once that topology is supported. Not needed now. |

### Determination

**Approach A, centralized.** The open decision resolves firmly, not tentatively:

- The two identity patterns in the codebase exist for *opposite* needs (finding 3 vs finding 4).
  Credentials must **follow the path**, not detect directory swaps - so a restored/recreated
  profile keeps working. That rules C out on the merits, not just on cost. C would re-introduce
  the silent-orphan failure through a different door.
- A is not a compromise; it is the pattern the codebase already uses everywhere else for exactly
  "which profile is this". Centralizing it (removing the `_key()` / `profile_matches_home`
  copy-paste and the `normcase` drift) is the "evolve toward good pattern use in the context of
  the existing code" move.
- The residual gap A does not close - cross-namespace shared volumes - is not a supported MCP
  OAuth topology today (Keychain is single-machine by construction; the gateway runs as a local
  process). If it becomes supported, the answer is **D** (an explicit operator-set identity), not
  a cleverer auto-derivation. Record D as the named future path.

No maintainer decision required beyond confirming the centralization is in scope for Chunk 1
(it touches `_key()` and `profile_matches_home`, both small).

### Target sections / code

- `architecture §4.1` - state the canonicalization rule explicitly (delegates to
  `hermes_home_key()` semantics); note the case-insensitive-FS and cross-namespace limitations and
  that D is the path for the latter.
- `hermes_constants.py` - if `hermes_home_key()` is reused as-is, fine; otherwise add a thin
  `profile_identity_key()` alias so the OAuth digest and the plugin scope key cannot drift.
- `MCPOAuthManager._key()` (`tools/mcp_oauth_manager.py:681`), `profile_matches_home()`
  (`hermes_cli/profiles.py:402`) - migrate to the shared helper; add `normcase`.
- Chunk 1 plan Step 3 and its tests - the parametrized cross-spelling digest-equality test;
  `test_identity_canonicalizes_profile_and_server_url` already exists and covers the `..` case,
  extend it.
- `architecture §14` - no new error code (A does not fail closed; it always produces a digest).

---

## F-4. Legacy reader-incoherence window between Chunk 3 and Chunk 5

The Chunk 3 compatibility backend still writes 3-4 separate files. Phase 2's exit criterion
("failed authorization cannot modify active credentials") does not imply coherent concurrent reads.

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Add one sentence to `§18` Phase 2 and to `design/mcp-oauth-03`: Phase 2 guarantees no destructive failure only. Reader coherence across the token/client/metadata triple is not guaranteed until Phase 3. | Honest. Zero code. Sets reviewer and operator expectations correctly. | Does nothing about the window itself. Fine - it is transitional. |
| B | A, plus make Chunk 3's commit order explicit: metadata -> client -> token last, each via atomic `os.replace`. Document precisely the one remaining benign window (reader between client-write and token-write sees old token + new client; the old token still validates against an unchanged dynamic registration). | Nearly free - Chunk 3 already says "commit orders ... to minimize inconsistency". Shrinks the window to a single benign interleaving and names it. | Still a window. Only benign as long as the client registration did not change in a way that invalidates the old token; note that case as the exception. |
| C | Add a per-identity "generation" marker file the reader checks to detect an in-progress commit and retry the read once. | Closes the window. | Real complexity added to a backend that Chunk 5 deletes. Poor ROI. |

Recommendation: **A + B.** State the limitation plainly and make the commit order a written
guarantee with the residual window named. Reject C - do not invest in a throwaway backend.

Target sections: `§18` Phase 2, `design/mcp-oauth-03` "Flow" step 9.

---

## F-5. Wall-clock step vs expiry classification

Current `§7.1`: state is recalculated from absolute `expires_at` on every load; monotonic time is
for in-process waits only. A backward wall-clock step (NTP, manual change) makes a token look valid
longer than it is. The reviewer asks whether `expires_in` should be re-anchored on reload.

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Document the residual risk only: `expires_at` is authoritative; a large backward clock step can extend apparent validity; the mitigation is the unknown-lifetime / one-bounded-refresh-on-rejection path. No behavior change. | Zero code. The 401 recovery path genuinely does catch the case on first use. | A token that looks valid but is not still gets sent once before recovery kicks in. |
| B | Persist the raw `expires_in` seconds alongside `accepted_at_utc`. On load, if `now < accepted_at_utc`, or `now - accepted_at_utc` is negative or wildly larger than `expires_in`, classify as `unknown` (a state already handled well) instead of trusting `valid`. | Cheap. Catches the gross clock-break cases and routes them into an existing safe state. Does not shorten healthy tokens. | Adds one persisted field. Does not catch a *small* backward step (minutes), only clearly broken clocks. |
| C | Long-running processes keep an in-memory `(monotonic_at_load, walltime_at_load)` pair and re-classify conservatively when wall time drifts from the monotonic prediction beyond a threshold. | Catches gradual/small drift for the gateway. | Nothing for CLI (short-lived). Adds process state and a threshold to tune. |
| D | Re-anchor on every load: `effective_expires_at = min(persisted_expires_at, now + original_expires_in)`. Directly implements "re-anchor when remaining lifetime is unknown". | Bounds apparent lifetime to the original grant on every load. | After a *forward* clock correction this repeatedly shortens a legitimately long token and can cause premature-refresh storms against a rate-limited endpoint. |

### Findings from the existing codebase

1. **The token file already carries both `expires_in` and `expires_at` today.** `set_tokens`
   (`tools/mcp_oauth.py:527`) dumps the SDK payload (which includes `expires_in`) and then adds
   `payload["expires_at"] = time.time() + int(expires_in)`. `_write_json` persists the whole dict.
   B does not add a *file* field so much as stop discarding one.

2. **`get_tokens` currently clobbers the raw grant on read.** `tools/mcp_oauth.py:507-509`:
   `data["expires_in"] = int(max(absolute_expiry - time.time(), 0))` - the on-disk `expires_in` is
   overwritten with *remaining* seconds before `model_validate`. B's change is precisely: keep the
   original grant duration as its own field and never recompute it on load.

3. **A plausibility-clamp against a second time reference already exists.** The legacy fallback
   (`:510-518`) takes `file_mtime + expires_in` as a proxy acceptance time and clamps `expires_in`
   to zero when that implied expiry is already past. B generalizes this same idea (sanity-check a
   persisted expiry against a second reference) to the `accepted_at_utc` + `original_expires_in`
   pair.

4. **The codebase consistently uses `time.monotonic()` for in-process deadlines** (approval,
   browser, code_kernel, clarify_gateway, bot_relay) and `time.time()` for persisted timestamps.
   No "clock went backwards" guard for persisted values exists yet - B introduces one, but the
   split it relies on (monotonic for waits, wall for persistence) is already the house style and
   matches `design/mcp-oauth-04`.

5. **The current architecture text says the opposite of B.** `§4.2`: "Relative `expires_in` is
   accepted from protocol responses but converted to an absolute timestamp before persistence."
   `design/mcp-oauth-04` `OAuthTokenRecord` has `accepted_at_utc` and `expires_at` but no raw
   `expires_in`. B requires amending both.

### Decision: Approach B (approved)

`OAuthTokenRecord` gains `original_expires_in: int | None` - the grant duration exactly as the
provider returned it, persisted, **never recomputed on load**. `expires_at` and `accepted_at_utc`
stay as design-04 already has them.

Load-time classification guard (in the lifecycle service's expiration evaluation, `§4.3` / `§7.1`):

```
given now (wall-clock UTC), accepted_at_utc, expires_at, original_expires_in:

  if original_expires_in is None:        -> unknown        (already the rule)
  if now < accepted_at_utc:              -> unknown        (clock behind acceptance: impossible)
  elapsed = now - accepted_at_utc
  if elapsed < 0 or elapsed > original_expires_in * SLACK:  -> unknown
        (SLACK ~ 2.0: absorbs legitimate NTP drift and refresh-token re-anchoring,
         trips only on a gross wall-clock step)
  otherwise classify normally from expires_at
```

`unknown` is an already-tested, already-safe state: the token is used until the provider rejects
it, and F-2's bounded rejection recovery handles the rejection. B only *demotes* to it; it never
shortens `expires_at`, so there is no forced-refresh storm (that was D's failure mode).

Also record A's residual risk explicitly: a *small* backward step (seconds to minutes, under the
SLACK band) still slips through and can make a token look valid slightly longer than it is; the
F-2 recovery path is the backstop.

### Target sections / code

- `architecture §4.2` - replace "converted to an absolute timestamp before persistence" with:
  both the absolute `expires_at` and the original `original_expires_in` are persisted; the latter
  is authoritative for the plausibility guard and is never recomputed.
- `architecture §4.3` / `§7.1` - add the load-time guard above; add the residual-small-step note.
- `design/mcp-oauth-04` - add `original_expires_in` to `OAuthTokenRecord`; the "Token time model"
  and "Expiration classification" sections gain the guard.
- `tools/mcp_oauth.py` - `set_tokens` already writes `expires_in`; `get_tokens` must stop
  overwriting it (keep the remaining-seconds value in a transient field the SDK sees, keep the
  original in the persisted record). The Chunk-4 backend owns this once the legacy adapter is
  gone.
- `design/mcp-oauth-04` tests - add: `now` before `accepted_at_utc` -> `unknown`; `elapsed`
  far beyond `original_expires_in * SLACK` -> `unknown`; normal drift within SLACK -> classified
  normally (no false demotion); a demoted token that the provider then accepts is used, and a
  rejection routes into F-2 recovery.

---

## F-6. Keychain duplicate-item ambiguity missing from the test matrix

`§10.3` already maps duplicate items to a typed error; `§17.1` does not test it.

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Add a Keychain-only contract case: pre-create two generic-password items matching service+account, then assert `load`, `compare_and_swap`, `replace_authorized`, and `delete` each return a typed error and never operate on an arbitrary match. Add a dedicated code `credential_ambiguous` to `§14` (mapping to `credential_corrupt` is misleading - the payload is fine). | Closes the coverage gap. The dedicated code makes diagnostics and operator guidance accurate. | One new error code to plumb. |
| B | A, plus a remediation path: the error names the exact `security delete-generic-password` command, or a `hermes mcp credentials repair` subcommand resolves it. | Operator can self-serve. | Repair tooling is Chunk 7 scope; adds surface now. |

Recommendation: **A.** Add `credential_ambiguous` and the four-operation test. Defer B's repair
command to Chunk 7 diagnostics unless the error string alone is judged insufficient.

Target sections: `§14`, `§17.1`, `design/mcp-oauth-06` "Backend contract" tests.

---

## F-7. Bound the revision prefix allowed in diagnostics

`§15` prohibits full revisions in logs. The reviewer wants a testable length bound (4 hex fine, 32
hex not).

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | `§15`: logged/diagnostic revision prefix MUST be <= 8 hex chars (32 bits). Add a test asserting no log field and no `OAuthCredentialStatus.revision_prefix` exceeds it. | One-line rule, one test. 8 hex is enough to correlate two log lines by eye; 96 bits stay unguessable. | Picks a number somewhat arbitrarily. The actual attack value of a partial revision is near zero (CAS needs the full 128-bit value *and* backend write access, which is already game over), so this is hygiene, not a real vulnerability fix. |
| B | Never log any part of the CAS revision. Introduce a separate per-mutation correlation ID (random, logged in full) for tracing. | Clean separation: the revision's only job is CAS; correlation is a different concern with its own identifier. | A second identifier to generate and plumb through events. |

Recommendation: **A now** (8-hex bound + test), with B noted as the tidy refactor if a correlation
ID gets introduced for other observability reasons. Do not build B just for this.

Target sections: `§15`, `§17` (add the assertion), `design/mcp-oauth-07` diagnostic projection.

---

## F-8. Chunk 4 transitional revision envelope - crash between state write and manifest write

Chunk 4 stores "a non-secret revision alongside legacy state or in a compatibility manifest". If
the revision and the state it describes are written by two separate atomic operations, a crash
between them leaves a torn pair: new state + old revision (a stale writer's CAS falsely succeeds)
or old state + new revision (CAS falsely fails - benign).

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Embed `"revision"` inside the token record's own JSON envelope (`<server>.json`). The bundle revision is the revision on the token file - the CAS-relevant record. The same `os.replace` that writes the record writes the revision. No separate manifest, no cross-file atomicity problem. | The torn window disappears - one atomic write. Forward-compatible: this is exactly the Chunk 5 bundle envelope shape (`§9.2` already has top-level `"revision"`). Old Hermes ignores the unknown key. | Mutates the legacy token-file format one release early. Downgrade risk is low (unknown key ignored). Client/metadata files carry no independent revision - acceptable, CAS is defined on the token record. |
| B | Keep a separate manifest but make it self-checking: manifest stores `revision` + a hash of the state it describes; write state, then manifest; on load, if `hash(state) != manifest.state_hash` the manifest is stale -> treat revision as `unknown` and force a reload/CAS-miss. | Detects the torn window rather than preventing it. | Window between the two `os.replace` calls still exists; load now hashes state every time; more moving parts than A. |
| C | Accept the risk transitionally: document Chunk 4 CAS as best-effort across the torn window; rely on Chunk 3's admin lock + pre-commit recheck for explicit flows; note that only concurrent *refresh* races are exposed and a falsely-successful stale CAS self-corrects at the next combined read. | No new format work. | Reintroduces, even briefly, the exact reader/writer incoherence this project exists to eliminate. |

Recommendation: **A.** Making the record write and the revision write the same operation removes
the failure mode instead of detecting it. It also means Chunk 4 -> Chunk 5 is a format continuation,
not a rewrite. Add the crash tests (crash between temp-write and `os.replace`; crash after
`os.replace`) and the two-writer one-winner test. Reject B (window remains) and C (reintroduces the
core bug).

Target sections: `design/mcp-oauth-04` "Transitional revision envelope" and "Compare-and-swap API",
`design/mcp-oauth-05` (note the continuity), `§8.4` crash guarantees, `§9.2`.

---

## Summary of recommendations

| ID | Recommended approach | Type | Maintainer decision needed? |
|----|----------------------|------|-----------------------------|
| F-0 | Keep `profile_home: Path`; rewrite `§4.1` so `profile_id` is a derived key; reuse `hermes_home_key()` | Consistency | No - codebase already decided (see F-0) |
| F-1 | A+B: fixed probe points + 10s in-memory TTL with backoff; D as escalation | Tuning | No |
| F-2 | A (approved): two-axis (position x kind) probe-outcome classifier; pre-token indeterminate -> transient error; post-token indeterminate -> commit `probe=deferred`; `handle_401` catches a bad token on first use | Policy | No - approved; B noted as fallback if "verified before commit" is later judged worth a guaranteed redo during upstream outages |
| F-3 | A, centralized: `hermes_home_key()` semantics for the digest; migrate `_key()` + `profile_matches_home` to the shared helper; D (explicit operator-set identity) is the deferred path for cross-namespace | Correctness | No - C ruled out on merits (orphans on profile recreation); only confirm centralization is in Chunk 1 scope |
| F-4 | A+B: state the limitation + explicit commit order with named residual window | Docs | No |
| F-5 | B (approved): add `original_expires_in` to the record, never recomputed; load-time guard demotes impossible/gross-skew deltas to `unknown`; small-step residual noted | Policy | No - approved |
| F-6 | A: add `credential_ambiguous` code + four-operation Keychain test | Test coverage | No |
| F-7 | A: 8-hex logged-prefix bound + test | Minor | No |
| F-8 | A: embed revision in the record envelope; add crash + two-writer tests | Correctness | No |

All F-2 through F-8 approach decisions are now closed. F-2 and F-5 carry a residual policy note
(F-2: B remains available if "verified before commit" is judged worth a guaranteed redo during
upstream outages; F-5: a small sub-SLACK backward clock step still slips through by design).

Settled by existing code or approved:
- F-0: `profile_home: Path` in the domain object; `profile_id` becomes a derived SHA-256 backend
  key; canonicalization reuses `hermes_constants.hermes_home_key()`.
- F-2: Approach A. `handle_401` (`mcp_oauth_manager.py:862`) is a mature recovery path whose
  contract is "discover a bad token on first use"; status-class discrimination is already an
  established pattern (`:458`, `:438`); config knobs are resisted locally (`mcp_config.py:857`).
  Approved. Two-axis (position x kind) taxonomy; five-endpoint stakeholder model.
- F-3: Approach A, centralized. The codebase has two identity patterns for opposite purposes -
  canonical-path (`hermes_home_key`, plugins/registry/config) which follows the path, and
  inode-tuple (`checkpoint_manager`) which detects directory swaps. Credentials need the
  path-following one so a restored/recreated profile keeps its credentials; that rules out the
  inode/UUID-file approach on the merits. Cross-namespace shared volumes -> deferred to an
  explicit operator-set identity (D), not a cleverer auto-derivation.

---

## My bias

These recommendations lean a consistent way. The priors:

1. **Remove the failure mode over detecting it.** F-8 A (one atomic write) over B (torn-window
   detector). Note F-3 landed the other way once researched: the "purer" inode/UUID identity was
   ruled out because it *creates* a failure mode (orphan-on-recreation) that the canonical-path
   approach does not - structural-purity instinct lost to a concrete regression, correctly.

2. **Resist new configuration surface.** Against F-2 C outright. This matches the architecture's own
   stated position (`§11`), but it is also a personal prior: every knob is a support burden and a
   path to being misconfigured into insecurity.

3. **Smallest defensible change for a chunk that is mid-flight.** Chunk 1 stopped at Task 6.
   Documentation-and-test fixes (F-4, F-6, F-7) are recommended as-is; deeper reworks (F-3 C,
   F-7 B) are deferred with an explicit trigger rather than done now. Someone prioritizing
   long-term cleanliness over delivery would pull more of that work forward.

4. **Trust protocol evidence over transport-liveness evidence.** F-2 A commits a token that
   completed an authorization-code exchange against a verified issuer even when a connectivity
   probe fails. A more conservative reviewer would keep the probe as a hard gate (F-2 B).

5. **Conservative about anything that can shorten a token's life.** F-5 B over D specifically
   because D's refresh-storm failure mode under forward clock corrections is, to me, worse than the
   original problem. Someone weighting "never serve a stale token" above "never hammer the refresh
   endpoint" would pick D.

6. **Accept the reviewer's collaboration offers.** Cross-process CAS matrices and headless Keychain
   testing are exactly where an author's mental model has blind spots and a fresh adversarial eye
   pays off.

7. **Known blind spot: I have run none of this.** The F-1 subprocess-cost analysis is estimation,
   not measurement. My default is "measure before optimizing", which risks under-serving a real
   gateway latency problem. If profiling later shows subprocess cost is material, F-1 D and F-3 C
   both deserve to be pulled forward.
