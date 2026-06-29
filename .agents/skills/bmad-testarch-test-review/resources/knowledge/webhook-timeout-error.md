# WebhookTimeoutError and Debugging

## Principle

`WebhookTimeoutError` is thrown when `waitFor` or `waitForCount` does not find a matching webhook within the configured timeout. It carries a snapshot of received webhooks from the last polling cycle — truncated to the last 10 entries — so you can inspect what arrived vs. what was expected. The full count of all received webhooks is available in `totalReceived`.

## Error Properties

```typescript
class WebhookTimeoutError extends Error {
  readonly name = 'WebhookTimeoutError';
  readonly templateName: string; // from webhookTemplate('...')
  readonly timeoutMs: number; // the timeout that was exceeded
  readonly totalReceived: number; // total webhooks seen in polling window
  readonly receivedWebhooks: ReceivedWebhook[]; // last ≤10 received webhooks
  readonly matcherDetails: string[]; // human-readable matcher summary

  toJSON(): Record<string, unknown>; // serialize all fields for CI logs
}
```

`receivedWebhooks` is capped at the last 10 entries. If more than 10 webhooks arrived, `totalReceived` shows the full count but `receivedWebhooks` contains only the most recent 10.

## Reading the Error

The error message format:

```
Webhook "movie.deleted" not received within 15000ms.
3 webhook(s) were received but none matched.
Matchers: field(event="movie.deleted"), field(data.id=42).
```

Use `matcherDetails` to confirm the matchers were configured correctly. Use `receivedWebhooks` to inspect actual payloads — compare field paths and values against what the matchers expect.

## Validating the Error Shape in Tests

```typescript
import { WebhookTimeoutError, webhookTemplate } from '@seontechnologies/playwright-utils/webhook';

const neverArrivingTemplate = webhookTemplate('never.arrives')
  .matchField('event', 'event.that.never.happens')
  .withTimeout(500)
  .withInterval(100)
  .build();

const [waitResult] = await Promise.allSettled([webhookRegistry.waitFor(neverArrivingTemplate)]);

expect(waitResult.status).toBe('rejected');
if (waitResult.status !== 'rejected') {
  throw new Error('Expected webhook wait to reject with WebhookTimeoutError');
}

const error = waitResult.reason as WebhookTimeoutError;
expect(error).toBeInstanceOf(WebhookTimeoutError);
expect(error.templateName).toBe('never.arrives');
expect(error.timeoutMs).toBe(500);
expect(error.toJSON()).toMatchObject({
  name: 'WebhookTimeoutError',
  templateName: 'never.arrives',
  timeoutMs: 500,
  totalReceived: expect.any(Number),
  matcherDetails: ['field(event="event.that.never.happens")'],
});
```

## Inspecting receivedWebhooks

When a webhook arrives but doesn't match, `receivedWebhooks` shows you what actually came in:

```typescript
// Wait for create webhook first — puts it in the journal
await webhookRegistry.waitFor(movieCreated(movieId));

// Wait for delete webhook that will never arrive — no delete was called
const undeliveredDelete = webhookTemplate<{
  event: string;
  data: { id: number };
}>('movie.deleted.not.delivered')
  .matchField('event', 'movie.deleted')
  .matchField('data.id', movieId)
  .withTimeout(2_000)
  .withInterval(200)
  .build();

const [waitResult] = await Promise.allSettled([webhookRegistry.waitFor(undeliveredDelete)]);

expect(waitResult.status).toBe('rejected');
if (waitResult.status !== 'rejected') {
  throw new Error('Expected webhook wait to reject with WebhookTimeoutError');
}

const error = waitResult.reason as WebhookTimeoutError;
expect(error).toBeInstanceOf(WebhookTimeoutError);
expect(error.totalReceived).toBeGreaterThanOrEqual(1);

// The movie.created webhook that did arrive is visible in the error
const createdWebhook = error.receivedWebhooks.find((w) => (w.body as { data: { id: number } }).data.id === movieId);
expect(createdWebhook).toBeDefined();
expect((createdWebhook!.body as { event: string }).event).toBe('movie.created');
```

## Common Failure Patterns

| What you see                           | Likely cause                                         | Fix                                                               |
| -------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| `totalReceived: 0`                     | Webhook not delivered; wrong URL or event not firing | Check application event publishing and webhook routing            |
| `totalReceived > 0`, none match        | Webhooks arriving but matchers not matching          | Inspect `receivedWebhooks[0].body` — check field paths and values |
| `matcherDetails` shows wrong path      | Template factory misconfigured                       | Print `error.toJSON()` and compare paths against actual payload   |
| `totalReceived: 0` with `matched-only` | Another worker claimed and deleted the webhook first | Ensure template is scoped by entity ID                            |
| Parse error in body                    | Webhook body is not valid JSON                       | Check `receivedWebhooks[n].parseError` and `rawBody`              |

## matcherDetails Format per Matcher Type

| Matcher                         | matcherDetails string |
| ------------------------------- | --------------------- |
| `matchField('event', 'x')`      | `field(event="x")`    |
| `matchPartial({ a: 1 })`        | `partial({"a":1})`    |
| `matchPredicate('my desc', fn)` | `predicate(my desc)`  |

## Import

```typescript
import { WebhookTimeoutError } from '@seontechnologies/playwright-utils/webhook';
```

## Related Fragments

- `webhook-template-matchers.md` — matcherDetails string format per matcher type
- `webhook-waiting-querying.md` — waitFor and waitForCount throw this error on timeout
