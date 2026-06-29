# Webhook Module Setup

## Principle

Wire the provider once in a central fixtures file using the `webhookProviderFixture + webhookFixture + mergeTests` pattern. Tests that request `webhookRegistry` get automatic setup and teardown; tests that don't pay nothing (Playwright lazy fixture evaluation).

## Fixture Wiring Pattern

### WireMock Provider (recommended for most setups)

The WireMock provider works with any backend that implements the `/__admin/requests` API format — not just actual WireMock. The playwright-utils sample app's Express backend uses this exact format.

```typescript
// playwright/support/merged-fixtures.ts
import { test as base, mergeTests } from '@playwright/test';
import { test as webhookFixture } from '@seontechnologies/playwright-utils/webhook/fixtures';
import { WireMockWebhookProvider } from '@seontechnologies/playwright-utils/webhook';
import { API_URL } from '../config/local.config';

// Lazy-initialized by Playwright — no cost for tests that don't request webhookRegistry.
const webhookProviderFixture = base.extend<{
  webhookProvider: WireMockWebhookProvider;
}>({
  webhookProvider: async ({ request }, use) => {
    const provider = new WireMockWebhookProvider(API_URL, request);
    await use(provider);
  },
});

const test = mergeTests(
  base,
  // ...your other fixtures...
  webhookFixture,
  webhookProviderFixture,
);

// Use matched-only cleanup project-wide: each test only deletes the webhooks it
// matched, so a parallel worker's teardown cannot wipe the shared journal while
// another test is still mid-flight (fullyParallel: true race condition).
test.use({ webhookConfig: { cleanupStrategy: 'matched-only' } });

export { test };
```

This is the exact pattern used in the playwright-utils E2E suite (`playwright/support/merged-fixtures.ts`).

### MockServer Provider

```typescript
import { MockServerWebhookProvider } from '@seontechnologies/playwright-utils/webhook';

const webhookProviderFixture = base.extend<{
  webhookProvider: MockServerWebhookProvider;
}>({
  webhookProvider: async ({ request }, use) => {
    await use(new MockServerWebhookProvider(API_URL, request));
  },
});

const test = mergeTests(base, /* ...other fixtures... */ webhookFixture, webhookProviderFixture);

// MockServer has no delete-by-ID on log entries — use full-reset for explicit cleanup
test.use({ webhookConfig: { cleanupStrategy: 'full-reset' } });
```

### Mockoon Provider

```typescript
import { MockoonWebhookProvider } from '@seontechnologies/playwright-utils/webhook';

const webhookProviderFixture = base.extend<{
  webhookProvider: MockoonWebhookProvider;
}>({
  webhookProvider: async ({ request }, use) => {
    await use(new MockoonWebhookProvider(API_URL, request));
  },
});

const test = mergeTests(base, /* ...other fixtures... */ webhookFixture, webhookProviderFixture);

// Mockoon has no delete-by-ID on log entries — use full-reset for explicit cleanup
test.use({ webhookConfig: { cleanupStrategy: 'full-reset' } });
```

## Cleanup Strategy Decision

| Strategy                 | Behaviour                                                                            | When to choose                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `'full-reset'` (default) | Calls `provider.resetJournal()` — wipes the entire mock server journal               | Safe only for serial execution or when each worker has an isolated provider instance                                 |
| `'matched-only'`         | Calls `provider.deleteById(id)` for each webhook matched by `waitFor`/`waitForCount` | Required for `fullyParallel: true` with a shared journal **when the provider supports `deleteById`** (e.g. WireMock) |

**The race condition under `fullyParallel: true`**: Worker A finishes and calls `resetJournal()`. Worker B is mid-poll waiting for its webhook. Worker A's reset just deleted Worker B's webhook — the poll times out with `WebhookTimeoutError`. Use `matched-only` to avoid this — but only when the provider supports `deleteById`.

**MockServer and Mockoon limitation**: Neither supports `deleteById` — their implementations are no-ops. The `startedAt` timestamp filter isolates _reads_ inside `waitFor`/`waitForCount`, but `cleanup()` with `full-reset` still calls `resetJournal()`, which wipes the entire journal. This means the teardown race exists for these providers too under `fullyParallel: true`. For parallel suites with MockServer or Mockoon, either run serially (`workers: 1`) or provision an isolated mock server instance per worker.

## Fixture Lifecycle

The fixture calls these in order:

1. `provider.setup?.()` — optional health check or stub registration
2. Tests run with `webhookRegistry` available
3. `registry.cleanup()` — deletes matched webhooks (`matched-only`) or resets journal (`full-reset`)
4. `provider.teardown?.()` — optional resource cleanup

Both cleanup and teardown failures are caught and logged as warnings — they don't mask actual test failures.

## WebhookRegistryConfig Options

```typescript
type WebhookRegistryConfig = {
  defaultTimeout?: number; // default: 30000 ms
  defaultInterval?: number; // default: 1000 ms
  cleanupStrategy?: 'matched-only' | 'full-reset'; // default: 'full-reset'
};
```

## Related Fragments

- `webhook-testing-fundamentals.md` — Why webhook tests are hard
- `webhook-template-matchers.md` — Template building and matcher patterns
- `webhook-providers.md` — WireMock, MockServer, Mockoon, custom provider details
- `fixtures-composition.md` — mergeTests pattern
