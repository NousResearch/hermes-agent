# Webhook Provider Patterns

## Principle

Three built-in providers ship with playwright-utils. Each wraps a different mock server API. For any backend not covered, implement the `WebhookProvider` interface. The registry only cares about the contract — not the backend technology.

## WireMockWebhookProvider

Uses `GET /__admin/requests` to fetch the webhook log and `DELETE /__admin/requests` to reset. Supports `deleteById` for `matched-only` cleanup.

**Works with any backend implementing the `/__admin/requests` format** — not just actual WireMock. The playwright-utils sample app's Express backend uses this exact format.

```typescript
import { WireMockWebhookProvider } from '@seontechnologies/playwright-utils/webhook';
import { API_URL } from '../config/local.config';

const webhookProviderFixture = base.extend<{
  webhookProvider: WireMockWebhookProvider;
}>({
  webhookProvider: async ({ request }, use) => {
    const provider = new WireMockWebhookProvider(API_URL, request);
    await use(provider);
  },
});
```

Supports both cleanup strategies. Use `matched-only` when running `fullyParallel: true`.

## MockServerWebhookProvider

Uses `PUT /mockserver/retrieve` to fetch logs with client-side `since` filtering.

**Limitation**: `deleteById` is a no-op — MockServer does not support deleting individual log entries by ID. The `startedAt` timestamp filter handles per-test isolation. Use `full-reset` for explicit journal cleanup.

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

// MockServer has no delete-by-ID on log entries — use full-reset
test.use({ webhookConfig: { cleanupStrategy: 'full-reset' } });
```

## MockoonWebhookProvider

Uses `GET /mockoon-admin/logs` to fetch logs. The admin API is enabled by default in `@mockoon/cli`. Default log limit is 100 entries — increase with `--max-transaction-logs` if your suite generates more.

**Limitation**: `deleteById` is a no-op for the same reason as MockServer. Use `full-reset`.

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

// Mockoon has no delete-by-ID on log entries — use full-reset
test.use({ webhookConfig: { cleanupStrategy: 'full-reset' } });
```

Start Mockoon with an increased log limit if needed:

```bash
mockoon-cli start --data ./mockoon-config.json --max-transaction-logs 500
```

## Custom Provider

Implement `WebhookProvider` for any backend that exposes a queryable request log:

```typescript
// support/providers/custom-webhook-provider.ts
import type { WebhookProvider, ReceivedWebhook, WebhookQueryFilter } from '@seontechnologies/playwright-utils/webhook';
import type { APIRequestContext } from '@playwright/test';

export class CustomWebhookProvider implements WebhookProvider {
  constructor(
    private readonly baseUrl: string,
    private readonly request: APIRequestContext,
  ) {}

  async getReceivedWebhooks(filter?: WebhookQueryFilter): Promise<ReceivedWebhook[]> {
    const params = new URLSearchParams();
    if (filter?.since) params.set('since', filter.since.toISOString());
    if (filter?.method) params.set('method', filter.method);

    const response = await this.request.get(`${this.baseUrl}/webhooks/received?${params}`);
    const { webhooks } = await response.json();
    return webhooks.map((w: Record<string, unknown>) => ({
      id: String(w.id),
      url: String(w.url),
      method: String(w.method),
      headers: (w.headers as Record<string, string>) ?? {},
      body: w.body,
      receivedAt: new Date(String(w.receivedAt)),
    }));
  }

  async resetJournal(): Promise<void> {
    await this.request.delete(`${this.baseUrl}/webhooks/received`);
  }

  async deleteById(id: string): Promise<void> {
    await this.request.delete(`${this.baseUrl}/webhooks/received/${id}`);
  }

  async getCount(): Promise<number> {
    const response = await this.request.get(`${this.baseUrl}/webhooks/count`);
    const { count } = await response.json();
    return count as number;
  }
}
```

## WebhookProvider Interface

```typescript
interface WebhookProvider {
  getReceivedWebhooks(filter?: WebhookQueryFilter): Promise<ReceivedWebhook[]>;
  resetJournal(): Promise<void>;
  deleteById(id: string): Promise<void>;
  getCount(criteria?: Record<string, unknown>): Promise<number>;
  removeByCriteria?(criteria: Record<string, unknown>): Promise<void>;
  setup?(): Promise<void>; // optional — called before test
  teardown?(): Promise<void>; // optional — called after test
}
```

## Provider Comparison

| Provider                  | deleteById | resetJournal | Parallel-safe (shared journal)      | Recommended strategy                                  | API endpoint           |
| ------------------------- | ---------- | ------------ | ----------------------------------- | ----------------------------------------------------- | ---------------------- |
| WireMockWebhookProvider   | ✅ Yes     | ✅ Yes       | ✅ Yes (`matched-only`)             | `matched-only`                                        | `/__admin/requests`    |
| MockServerWebhookProvider | ❌ No-op   | ✅ Yes       | ⚠️ No — serial or isolated instance | `full-reset` (serial or isolated provider per worker) | `/mockserver/retrieve` |
| MockoonWebhookProvider    | ❌ No-op   | ✅ Yes       | ⚠️ No — serial or isolated instance | `full-reset` (serial or isolated provider per worker) | `/mockoon-admin/logs`  |
| Custom                    | Depends    | Depends      | Depends on implementation           | Depends                                               | Your API               |

## Related Fragments

- `webhook-module-setup.md` — Full fixture wiring for each provider
- `webhook-testing-fundamentals.md` — Cleanup strategy rationale
