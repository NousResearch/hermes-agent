# Webhook Testing Fundamentals

## Principle

Webhook delivery is eventually consistent — your application fires HTTP callbacks asynchronously after events occur. Tests must poll until the expected webhook arrives or time out. The `@seontechnologies/playwright-utils` webhook module provides deterministic polling, typed matchers, rich timeout diagnostics, and cleanup strategies safe under `fullyParallel: true`.

## Rationale

Webhook tests fail for four structural reasons:

- **Eventually consistent**: Webhook delivery happens asynchronously — you cannot assert immediately after triggering an event
- **Parallel journal pollution**: When multiple workers share the same mock server, a fast worker's teardown can delete records a slow worker is still polling
- **Opaque timeouts**: A bare timeout tells you only that the webhook didn't arrive — it shows you nothing about what did arrive
- **Cleanup drift**: Resetting the full journal in `afterEach` creates a race condition under `fullyParallel: true`

The playwright-utils approach:

- **Polling via `recurse`**: Uses Playwright's `expect.poll` under the hood — retries with configurable timeout and interval until a match is found
- **Typed matchers**: `matchField`, `matchPartial`, `matchPredicate` — all must pass (AND semantics); matchers never throw on missing paths
- **Rich timeout errors**: `WebhookTimeoutError` carries `totalReceived`, `receivedWebhooks`, and `matcherDetails` so you can see what arrived vs. what was expected
- **Isolation via `startedAt`**: Each `WebhookRegistry` instance records its creation timestamp; polling only fetches webhooks received after that point, preventing leakage from prior tests
- **Two cleanup strategies**: `full-reset` (resets entire journal) and `matched-only` (deletes only matched webhooks — parallel-safe when the provider supports delete-by-ID, e.g. WireMock)

## When to Use Webhook Tests

| Scenario                                                          | Use webhook tests         |
| ----------------------------------------------------------------- | ------------------------- |
| Application publishes events to external subscribers              | ✅ Required               |
| Event-driven architecture with Kafka/event bus → webhook delivery | ✅ Required               |
| Payment, order, or notification side effects via webhooks         | ✅ Required               |
| Testing that a webhook was NOT delivered                          | ✅ Verify via timeout     |
| Polling a status endpoint for eventual consistency                | ❌ Use `recurse` directly |
| Frontend receiving push notifications (WebSocket)                 | ❌ Different mechanism    |

## Related Fragments

- `webhook-module-setup.md` — Fixture wiring and cleanup strategies
- `webhook-template-matchers.md` — matchField, matchPartial, matchPredicate
- `webhook-waiting-querying.md` — waitFor, waitForCount, getReceived, drain pattern
- `webhook-timeout-error.md` — WebhookTimeoutError debugging
- `webhook-providers.md` — WireMock, MockServer, Mockoon, custom provider
- `webhook-risk-guidance.md` — Risk-based guidance for TA and TD capabilities
