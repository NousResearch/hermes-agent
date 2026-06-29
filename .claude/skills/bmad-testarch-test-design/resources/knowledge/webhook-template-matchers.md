# Webhook Template Matchers

## Principle

Build typed templates with `webhookTemplate()` and compose matchers using `matchField`, `matchPartial`, and `matchPredicate`. All matchers on a template use AND semantics — every matcher must pass for a webhook to be considered a match. Templates are immutable value objects produced by a fluent builder.

## Template Factory Pattern

Define template factories as pure functions that accept a test-scoped ID. This is the key pattern for parallel isolation — each factory call produces a template bound to a specific entity:

```typescript
import { webhookTemplate } from '@seontechnologies/playwright-utils/webhook';

// Template factories for movie webhooks
// 15s timeout: the Kafka → HTTP webhook delivery pipeline can back up under
// high CI concurrency (burn-in with many parallel workers). 10s was occasionally
// not enough; 15s gives the pipeline headroom without slowing normal runs.
const movieCreated = (movieId: number) =>
  webhookTemplate<{ event: string; data: { id: number } }>('movie.created')
    .matchField('event', 'movie.created')
    .matchField('data.id', movieId)
    .withTimeout(15_000)
    .withInterval(500)
    .build();

const movieDeleted = (movieId: number) =>
  webhookTemplate<{ event: string; data: { id: number } }>('movie.deleted')
    .matchField('event', 'movie.deleted')
    .matchField('data.id', movieId)
    .withTimeout(15_000)
    .withInterval(500)
    .build();
```

The ID parameter scopes each template to a specific entity, preventing parallel workers from matching each other's webhooks.

## Matcher Reference

### matchField — dot-path exact match

Traverses dot-notation paths into the payload. Never throws if the path is missing — a missing path evaluates as non-matching.

```typescript
webhookTemplate('order.created')
  .matchField('event', 'order.created') // top-level field
  .matchField('data.id', orderId) // nested path
  .matchField('data.status', 'pending') // nested string value
  .build();
```

Matcher detail output: `field(data.id=42)`

### matchPartial — deep subset check

Checks that the expected object is a subset of the received payload. Extra fields in the payload are ignored. Arrays use strict length matching.

```typescript
const partialTemplate = webhookTemplate<{
  event: string;
  data: { id: number; name: string };
}>('movie.created.partial')
  .matchPartial({ event: 'movie.created', data: { id: movieId } })
  .withTimeout(10_000)
  .withInterval(500)
  .build();
```

Matcher detail output: `partial({"event":"movie.created","data":{"id":42}})`

### matchPredicate — arbitrary function

Accepts any `(payload: T) => boolean` function. Always requires a human-readable description string — this appears in `WebhookTimeoutError.matcherDetails` for debugging.

**ID-scoped parallel isolation** (prevents cross-worker contamination in `waitForCount`):

```typescript
const batchTemplate = webhookTemplate<{
  event: string;
  data: { id: number };
}>('movie.created.batch')
  .matchField('event', 'movie.created')
  .matchPredicate(`data.id is ${id1} or ${id2}`, (p) => p.data.id === id1 || p.data.id === id2)
  .withTimeout(15_000)
  .withInterval(500)
  .build();
```

**Business data filtering**:

```typescript
const highRatingTemplate = webhookTemplate<{
  event: string;
  data: { id: number; rating: number };
}>('movie.created.high-rating')
  .matchField('event', 'movie.created')
  .matchPredicate(`data.id is ${movieId} and data.rating >= 9`, (p) => p.data.id === movieId && p.data.rating >= 9)
  .withTimeout(10_000)
  .withInterval(500)
  .build();
```

Matcher detail output: `predicate(data.id is 42 and data.rating >= 9)`

## Combining Matchers

All matchers use AND semantics — all must pass for the webhook to match:

```typescript
// Combined field + partial: both matchers must pass
const updateTemplate = webhookTemplate<{
  event: string;
  data: { id: number; name: string };
}>('movie.updated')
  .matchField('event', 'movie.updated')
  .matchPartial({ data: { id: movieId, name: nameUpdate.name } })
  .withTimeout(10_000)
  .withInterval(500)
  .build();
```

## Per-Template Timeout and Interval

Override the registry defaults on a per-template basis:

```typescript
webhookTemplate('slow.pipeline.event')
  .matchField('event', 'slow.pipeline.event')
  .withTimeout(60_000) // 60s for slow delivery pipelines
  .withInterval(2_000) // poll every 2s
  .build();
```

## clone() for Base Template Variations

> **Note**: `clone()` is available on the builder but is not used in the playwright-utils E2E suite. Use it when multiple tests share the same base template with slight field variations.

```typescript
const base = webhookTemplate<OrderPayload>('order').matchField('event', 'order.completed');

const forOrderA = base.clone().matchField('data.orderId', 'A').build();
const forOrderB = base.clone().matchField('data.orderId', 'B').build();
```

## Builder API Summary

| Method                      | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `webhookTemplate<T>(name)`  | Create a new builder with the given template name      |
| `.matchField(path, value)`  | Add dot-path exact-match matcher                       |
| `.matchPartial(expected)`   | Add deep-subset matcher                                |
| `.matchPredicate(desc, fn)` | Add arbitrary predicate matcher (description required) |
| `.withTimeout(ms)`          | Override registry default timeout                      |
| `.withInterval(ms)`         | Override registry default poll interval                |
| `.clone()`                  | Copy current builder state for variation               |
| `.build()`                  | Produce the immutable `WebhookTemplate<T>` object      |

## Related Fragments

- `webhook-waiting-querying.md` — waitFor, waitForCount, drain pattern
- `webhook-timeout-error.md` — Reading matcherDetails in error output
