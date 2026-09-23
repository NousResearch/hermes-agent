# MCP OAuth Chunk 2 — Unified Lifecycle Service

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: shared store facade (Chunk 1)
Delivery plan: [`../plans/2026-09-01-mcp-oauth-credential-store-delivery-plan.md`](../plans/2026-09-01-mcp-oauth-credential-store-delivery-plan.md)
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §6.2, §6.3, §14
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-2 shape + classification; forward notes F-1, F-5, F-7)

## Purpose

Replace surface-owned authorization policy with one `OAuthLifecycleService`. CLI, dashboard, Desktop/TUI RPC, runtime refresh, reconnect, status, and explicit deletion use the same service and typed outcomes.

This chunk centralizes control flow but deliberately preserves legacy reauthorization persistence until Chunk 3 introduces staging.

## New modules

```text
tools/mcp_oauth_store/
├── lifecycle.py
├── interaction.py
├── diagnostics.py
└── sdk_adapter.py
```

## Lifecycle interface

```python
class OAuthLifecycleService:
    def load_for_runtime(self, identity, expected_issuer=None) -> StoredState | None: ...
    async def authorize(self, identity, oauth_config, interaction) -> AuthorizationResult: ...
    async def persist_refresh(self, identity, loaded, response) -> StoredState: ...
    def delete(self, identity, *, revoke_remote=False) -> DeletionResult: ...
    def status(self, identity) -> OAuthCredentialStatus: ...
```

`StoredState` is the transitional name for the loaded-credential handle in Chunks 2–4: it wraps `LegacyOAuthState` here, gains a revision in Chunk 4, and is replaced by architecture §4.2's `StoredBundle` in Chunk 5. The lifecycle signatures above match architecture §5.2 with `StoredState` substituted for `StoredBundle`.

`load_for_runtime` classifies token expiration from the absolute `expires_at` (architecture §4.3). The wall-clock plausibility guard — demote to `unknown` when the elapsed time since `accepted_at_utc` is negative or exceeds `original_expires_in × CLOCK_SLACK` — is added in Chunk 4, when `original_expires_in` is persisted. It is not a legacy-format field, so it cannot be applied here.

## Interaction abstraction

Authorization UI and callback transport implement:

```python
class OAuthInteraction(Protocol):
    async def publish_authorization_url(self, url: str) -> None: ...
    async def wait_for_callback(self, *, timeout: float) -> AuthorizationCodeResult: ...
    def report_progress(self, event: OAuthProgressEvent) -> None: ...
```

Concrete adapters:

- `LoopbackCLIInteraction`: browser open plus local callback waiter.
- `DashboardOAuthInteraction`: dashboard URL publication and HTTP callback delivery.
- `RPCOAuthInteraction`: Desktop local callback relay through TUI gateway RPC.

The interaction never receives stored tokens or client secrets.

## Authorization result

```python
@dataclass(frozen=True)
class AuthorizationResult:
    status: Literal["authorized", "cancelled", "failed"]
    probe: Literal["authenticated", "deferred", "not_run"]
    tools: tuple[OAuthToolSummary, ...]
    credential_status: OAuthCredentialStatus
    error: MCPOAuthCredentialError | None
```

No caller infers success merely because MCP initialization or `tools/list` returned successfully. `authorized` requires a persisted token.

`probe` reports the post-exchange authentication probe outcome (architecture §6.3):

- `authenticated` — the probe confirmed the token.
- `deferred` — the probe was indeterminate (MCP server 5xx / timeout / 429); `status` is still `authorized` because the token came from a completed code exchange, and the runtime's 401-recovery path is the backstop.
- `not_run` — the flow failed before the probe.

This is the final shape of the cross-surface contract. Chunk 2 fills `probe` from whatever the legacy flow already persisted. The *behavior* that makes `deferred` and the probe-`rejected` case safe — the one ~2 s retry, and committing a validated staged bundle rather than a directly-written token — is Chunk 3.

## Surface changes

### CLI

`_reauth_oauth_server` becomes presentation around `authorize`. It no longer owns provider construction, timeout policy, token verification, or error humanization.

### Dashboard

Dashboard flow storage retains only session/progress state. Its worker invokes `authorize` with `DashboardOAuthInteraction`.

### Desktop/TUI RPC

RPC start, callback, and poll methods manage transport session IDs and invoke the same service with `RPCOAuthInteraction`.

### Runtime and reconnect

Runtime loading and refresh call the service. Transport parking calls `MCPOAuthManager.evict`, which is defined as memory-only.

### Deletion

Only `OAuthLifecycleService.delete` invokes durable store deletion. Server configuration removal and remote token revocation are separate reported results.

## Provider manager boundary

`MCPOAuthManager` owns cached provider entries, in-flight 401 coordination, and provider reconstruction. It no longer exposes a method whose name `remove` ambiguously means both memory eviction and credential deletion.

`load_for_runtime` is a rebuild-decision-point call (before a flow, during 401 recovery, on explicit refresh or status), not a per-request call. Chunk 2 keeps legacy disk-mtime watching. Revisions and CAS arrive in Chunk 4; the switch from mtime watching to revision watching, and the in-memory revision TTL that architecture §8.3 / §12.1 formalize, arrive in Chunk 5.

Use explicit methods:

```python
manager.evict(identity)             # memory only
lifecycle.delete(identity)          # durable credentials
```

## Error translation

Protocol and backend errors become stable lifecycle codes before reaching surfaces. Presentation layers may add instructions but cannot choose destructive recovery.

The mapping runs through one classifier in `lifecycle.py` — `classify_outcome(stage, exc_or_response) -> Outcome` where `stage` is `pre_token` or `probe` (architecture §6.2, §6.3), built on `_unwrap_exception_group` plus httpx exception-type / status-code inspection:

- `pre_token` + definitive (HTTP 400, `invalid_grant`, `invalid_client`, unsupported registration) → a permanent code (`reauthorization_required` or a configuration error).
- `pre_token` + indeterminate (HTTP 5xx / connection error / timeout / 429 at discovery, registration, or token exchange) → `authorization_endpoint_unavailable` — a new lifecycle code, distinct from `authorization_timeout` (which means the user did not complete the browser step). A 429 `Retry-After` interval is carried through.
- `probe` → `authenticated` / `rejected` (HTTP 401/403) / `indeterminate` (5xx / timeout / 429).

Chunk 2 introduces the classifier and the `authorization_endpoint_unavailable` code and maps outcomes to `AuthorizationResult`. The one ~2 s retry per stage and the staged-commit mechanics are Chunk 3.

Background contexts map interaction-required states to `reauthorization_required`; they never start a browser.

## Tests

- One table-driven authorization outcome suite runs through each interaction adapter.
- Success, cancellation, invalid state, registration error, and public unauthenticated MCP response yield identical lifecycle semantics.
- Token-exchange failure is two cases: HTTP 400 `invalid_grant` → permanent code; HTTP 500 → `authorization_endpoint_unavailable`.
- Timeout is two cases: browser step not completed → `authorization_timeout`; transport timeout at discovery / registration / token exchange → `authorization_endpoint_unavailable`.
- An indeterminate probe outcome (MCP server 5xx) sets `AuthorizationResult.probe == "deferred"` while `status` stays `authorized`.
- CLI, dashboard, and RPC surface tests verify presentation and transport only.
- Transient reconnect proves durable state remains present after `evict`.
- Explicit delete proves memory cache and durable state are handled separately.

## Demonstration

Run the same fake OAuth peer through CLI, dashboard, and RPC adapters. Show identical lifecycle event order and final credential status:

```text
authorization_url → callback → token_obtained → probe → authorized
```

For a browser-completion timeout, show all three return `authorization_timeout` and the same safe guidance. For a transport failure at an OAuth endpoint (5xx / timeout), show all three return `authorization_endpoint_unavailable` with "retry shortly" guidance.

## Non-goals

- Do not yet eliminate the current snapshot/delete/restore internals.
- Do not add staged storage, the per-stage ~2 s retry, or committing a validated staged bundle past a deferred probe (F-2 behavior, Chunk 3).
- Do not change the legacy file format, add `original_expires_in`, or apply the wall-clock plausibility guard (F-5, Chunk 4).
- Do not add revision/CAS behavior or the 8-hex logged-revision-prefix bound (F-7, Chunk 4).
- Do not add Keychain or the `credential_ambiguous` code (Chunk 6).

Chunk 2 does own: the `classify_outcome` classifier, the `authorization_endpoint_unavailable` code, the final `AuthorizationResult` shape, and the `mcp_oauth.reauth_committed` `probe=…` / `mcp_oauth.reauth_aborted` `reason=…` event fields (these describe the authorize outcome, not a revision).

## Completion criteria

- Every surface calls the lifecycle service.
- Interaction adapters own no durable credential logic.
- Provider eviction and credential deletion are unambiguous and separate.
- Success always requires token persistence.
- A transient OAuth-endpoint failure yields `authorization_endpoint_unavailable`, not `authorization_timeout` or a generic failure, identically across all three surfaces.
- `AuthorizationResult` carries its final shape, including `probe`, so Chunk 3 changes behavior without changing the type.
- Chunk 0 failure remains reproducible through the unified service, setting up Chunk 3.
