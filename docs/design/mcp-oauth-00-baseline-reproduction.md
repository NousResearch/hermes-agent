# MCP OAuth Chunk 0 — Baseline Reproduction

Status: design proposal (harness partially built; F-2 kind-aware injection is a 2026-09-03 design addition not yet in `tests/fakes/mcp_oauth_peer.py`)
Delivery plan: [`../plans/2026-09-01-mcp-oauth-credential-store-delivery-plan.md`](../plans/2026-09-01-mcp-oauth-credential-store-delivery-plan.md)
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md)
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-2, F-3, F-5)

## Purpose

Establish deterministic, executable evidence for the MCP OAuth credential-loss bug on current `NousResearch/main`. This chunk changes no production behavior. Its output is a behavioral harness reused by later chunks.

The existing proposed fix branch is substantially behind current `main`; reproductions must be rebuilt against the current code rather than copied as source-shape tests.

## Behaviors to reproduce

The harness must prove these current behaviors independently:

1. Dashboard reauthorization snapshots live OAuth files, deletes them, writes partial client or metadata state, then skips rollback and loses the previous token.
2. Desktop/TUI RPC reauthorization follows the same destructive partial-state pattern.
3. CLI `hermes mcp login`/`reauth` deletes durable state before authorization and does not restore it after failure.
4. Runtime reconnect or parking must not delete a token after a transient transport failure.

The first three are expected failures on the baseline. The fourth is a preservation invariant and guards the useful soft-eviction portion of the earlier proposal. It is the same invariant the fix relies on: runtime reconnect calls only `MCPOAuthManager.evict` (architecture §12.4), and "a dependency being unreachable is not evidence against the credential" (architecture §6.3, §7.1) — so the preservation test cross-references those sections.

## Test harness

Add a reusable fake OAuth/MCP peer under the test tree. It must execute enough of the real flow to control failure points without external network access.

```python
class FakeOAuthMCPPeer:
    def register_client(self) -> ClientRecord: ...
    def publish_metadata(self) -> OAuthMetadata: ...
    def authorize(self) -> AuthorizationCode: ...
    def exchange_token(self) -> OAuthToken: ...
    def initialize_mcp(self) -> list[Tool]: ...
```

The peer supports failure injection after:

- Protected-resource discovery.
- Authorization-server discovery.
- Dynamic client registration.
- Authorization URL publication.
- Callback receipt.
- Token exchange.
- Token persistence.
- MCP initialization/probe.

Tests use an isolated temporary `HERMES_HOME`, real `HermesTokenStorage`, and real surface/lifecycle entry points where possible. Patch the network transport or provider boundary, not the persistence methods under test. The fixture seeds `HERMES_HOME` from `Path(tmp_path).resolve()` so a canonical spelling is the baseline for the identity-digest tests added in Chunk 1.

### Failure kind

Each pre-token network-boundary injection point (discovery, registration, token exchange) also accepts an optional *kind*, so the same peer serves the F-2 probe/commit taxonomy that later chunks assert (architecture §6.2, §6.3):

- `definitive` — HTTP 400, `invalid_grant`, `invalid_client`, or unsupported registration. The default, and the only kind the Chunk 0 baseline tests use: the current bug loses the token regardless of kind.
- `indeterminate` — HTTP 5xx, connection error, timeout, or HTTP 429 with an optional `Retry-After`.

The MCP probe point yields a classifiable outcome rather than a single raise: `authenticated`, `rejected` (HTTP 401/403), or `indeterminate` (5xx / timeout / 429). Browser-authorization failures (cancellation, callback timeout, invalid state) are exercised through the existing "Authorization URL publication" / "Callback receipt" points and are not kind-classified.

Chunk 0 only exposes this capability. The behavior built on it lands later: Chunk 2 introduces the failure classifier, the `authorization_endpoint_unavailable` code, and the `AuthorizationResult` shape; Chunk 3 adds the per-stage retry and the staged commit (including the `probe=deferred` path).

## Scenario fixture

Each destructive-flow scenario begins with:

```text
server.json        = OLD access + refresh token
server.client.json = OLD client registration
server.meta.json   = OLD issuer/token endpoint metadata
```

The injected flow writes a distinguishable `PARTIAL` client or metadata record and then fails. Assertions record the resulting artifact set and demonstrate that the active token disappears.

No real token values are used. Test sentinel strings must be unmistakably fake.

The OLD/NEW token fixtures use the legacy on-disk shape in Chunk 0 (`access_token`, `refresh_token`, `token_type`, `expires_in`). Chunk 4 extends them with `accepted_at_utc`, `expires_at`, and `original_expires_in`, and adds an injectable UTC clock to the peer for the wall-clock plausibility guard — reusing the same peer, not replacing it.

## Proposed test locations

- `tests/tools/test_mcp_oauth_reauth_regression.py`: shared persistence and failure-point matrix.
- `tests/hermes_cli/test_mcp_reauth_lifecycle.py`: CLI behavior.
- `tests/hermes_cli/test_mcp_dashboard_reauth.py`: dashboard behavior.
- `tests/tui_gateway/test_mcp_oauth_reauth.py`: Desktop/TUI RPC behavior.
- Extend the existing parked/reconnect tests for durable-token retention.

If current test organization already has a closer behavioral home, extend it rather than creating duplicate suites.

## Expected baseline evidence

The harness should emit a compact state transition in assertion failures or test diagnostics:

```text
before: token=OLD client=OLD metadata=OLD
flow:   client=PARTIAL; failure=authorization_timeout
after:  token=MISSING client=PARTIAL metadata=MISSING
```

Later chunks invert the expected result without rewriting the harness:

```text
after:  token=OLD client=OLD metadata=OLD
```

## Demonstration

Run the prescribed repository test wrapper against the focused files. The demonstration is complete when the test reliably reproduces credential loss on the baseline commit and passes against the transactional implementation in Chunk 3.

## Non-goals

- Do not introduce the new store protocol.
- Do not change OAuth production code.
- Do not read source files or assert function-call text.
- Do not depend on Todoist, Hugging Face, or another live provider.
- Do not freeze implementation-specific line numbers or file counts.
- Do not add the F-2 taxonomy assertions. Chunk 0 exposes kind-aware and classifiable-probe injection; Chunk 2 asserts classification and typed codes, Chunk 3 asserts the retry / abort / `probe=deferred` behavior.
- Do not add the Chunk 4 token time-model fields or clock injection yet.

## Merge strategy

A test that intentionally fails cannot merge alone. Use one of these approaches:

1. Land the harness with the first production change and show its baseline failure in the PR description.
2. Mark only the destructive expectations with a narrowly documented `xfail(strict=True)` tied to the bug, then remove the marks in Chunk 3.

The preferred approach is to include the harness in Chunk 1 while preserving a recorded baseline run in the PR description.

## Completion criteria

- Every surface-specific failure is reproducible without network access.
- The tests execute production imports and real temporary credential state.
- Failure injection is reusable by transactional reauthorization tests, including failure kind (definitive rejection vs indeterminate transport) and classifiable probe outcome — not only failure point.
- Transient reconnect retention has a positive behavioral test.
- No secrets or real user paths appear in output.
