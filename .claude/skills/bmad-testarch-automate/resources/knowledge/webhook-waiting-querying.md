# Webhook Waiting and Querying Patterns

## Principle

`waitFor` and `waitForCount` poll until matching webhooks arrive; `getReceived` queries without waiting. Always drain preceding events before asserting on subsequent ones. Scope templates by entity ID to prevent parallel worker cross-contamination.

## Pattern Examples

### Example 1: waitFor — single webhook

Poll until the first webhook matching the template arrives. Returns the typed `ReceivedWebhook<T>`.

```typescript
const webhook = await webhookRegistry.waitFor(movieCreated(movieId));

expect(webhook.body).toMatchObject({
  event: 'movie.created',
  timestamp: expect.any(String),
  data: {
    id: movieId,
    name: movie.name,
    year: movie.year,
    rating: movie.rating,
  },
});
```

### Example 2: The drain pattern — sequential events

When testing a downstream event (e.g. deletion), always `waitFor` the preceding event first. Without the drain, the create webhook may remain in the journal and interfere with cleanup or subsequent polling.

```typescript
test('movie deletion triggers a webhook with correct payload', async ({ authToken, addMovie, deleteMovie, webhookRegistry }) => {
  const movie = generateMovieWithoutId();
  const { body: createResponse } = await addMovie(authToken, movie);
  const movieId = createResponse.data.id;

  await log.step('Drain the create webhook before testing the delete path');
  await webhookRegistry.waitFor(movieCreated(movieId)); // drain — consume the create event

  await deleteMovie(authToken, movieId);

  await log.step('Wait for the delete webhook');
  const webhook = await webhookRegistry.waitFor(movieDeleted(movieId));

  expect(webhook.body).toMatchObject({
    event: 'movie.deleted',
    data: { id: movieId, name: movie.name },
  });
});
```

**Why drain?** If you skip the drain and go directly to `waitFor(movieDeleted)`, the create webhook is already in the journal. The delete webhook may arrive and be cleaned up by another test before your poll reaches it. Draining makes the event order explicit and removes the ambiguity.

### Example 3: waitForCount — collect N webhooks concurrently

Collect exactly N matching webhooks. Use `matchPredicate` with all IDs to prevent cross-worker contamination when running `fullyParallel: true`:

```typescript
await log.step('Create two movies concurrently');
const [{ body: res1 }, { body: res2 }] = await Promise.all([
  addMovie(authToken, generateMovieWithoutId()),
  addMovie(authToken, generateMovieWithoutId()),
]);

const [id1, id2] = [res1.data.id, res2.data.id];

const batchTemplate = webhookTemplate<{
  event: string;
  data: { id: number };
}>('movie.created.batch')
  .matchField('event', 'movie.created')
  .matchPredicate(`data.id is ${id1} or ${id2}`, (p) => p.data.id === id1 || p.data.id === id2)
  .withTimeout(15_000)
  .withInterval(500)
  .build();

const webhooks = await webhookRegistry.waitForCount(batchTemplate, 2);

expect(webhooks).toHaveLength(2);
const receivedIds = webhooks.map((w) => w.body.data.id);
expect(receivedIds).toContain(id1);
expect(receivedIds).toContain(id2);
expect(new Set(receivedIds).size).toBe(2); // guard against the same ID delivered twice
```

### Example 4: getReceived — query without waiting

Query the journal without polling. Useful for asserting presence of webhooks after a `waitFor`, or for method/URL filtering.

```typescript
await webhookRegistry.waitFor(movieCreated(movieId)); // wait first

const all = await webhookRegistry.getReceived();
expect(all.length).toBeGreaterThanOrEqual(1);

// Method filter — all sample-app webhooks are delivered via POST
const postOnly = await webhookRegistry.getReceived({ method: 'POST' });
expect(postOnly.every((w) => w.method === 'POST')).toBe(true);

// URL pattern filter — match the webhooks endpoint path
const byUrl = await webhookRegistry.getReceived({ urlPattern: '/webhooks' });
expect(byUrl.every((w) => w.url.includes('/webhooks'))).toBe(true);
```

`getReceived` accepts `WebhookQueryFilter`:

```typescript
type WebhookQueryFilter = {
  urlPattern?: string; // glob or regex string
  method?: string; // HTTP method filter
  since?: Date; // only return webhooks after this timestamp
};
```

Note: `getReceived` is a direct passthrough to the provider — it does **not** automatically apply the `startedAt` filter. Only `waitFor` and `waitForCount` apply the since-filter internally during polling. If you need to scope a manual `getReceived` call to this test's time window, record your own timestamp before the action under test and pass `{ since: myTimestamp }` explicitly.

## Parallel Worker Safety

Always scope template factories to the entity's ID:

```typescript
// ✅ Scoped — only matches webhooks for this specific movie
const movieCreated = (movieId: number) =>
  webhookTemplate('movie.created')
    .matchField('event', 'movie.created')
    .matchField('data.id', movieId) // scoped by ID
    .build();

// ❌ Unscoped — will match any movie.created from any parallel worker
const movieCreatedUnscoped = webhookTemplate('movie.created').matchField('event', 'movie.created').build();
```

## Method Summary

| Method                      | Returns                         | Description                                                                                       |
| --------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `waitFor(template)`         | `Promise<ReceivedWebhook<T>>`   | Poll until first match; throws `WebhookTimeoutError` on timeout                                   |
| `waitForCount(template, n)` | `Promise<ReceivedWebhook<T>[]>` | Poll until N matches; throws `WebhookTimeoutError` on timeout                                     |
| `getReceived(filter?)`      | `Promise<ReceivedWebhook[]>`    | Direct passthrough to provider — no automatic since-filter; pass `{ since }` explicitly if needed |
| `resetJournal()`            | `Promise<void>`                 | Wipe the entire journal and clear matchedIds                                                      |
| `cleanup()`                 | `Promise<void>`                 | Delete matched webhooks (`matched-only`) or reset journal (`full-reset`)                          |

## Anti-Patterns

**DON'T skip the drain for sequential events:**

```typescript
// Bad: direct jump to delete webhook — create webhook pollutes the journal
await addMovie(authToken, movie);
const webhook = await webhookRegistry.waitFor(movieDeleted(movieId));
```

**DO drain preceding events:**

```typescript
// Good: drain create first, then wait for delete
await webhookRegistry.waitFor(movieCreated(movieId)); // drain
await deleteMovie(authToken, movieId);
const webhook = await webhookRegistry.waitFor(movieDeleted(movieId));
```

## Related Fragments

- `webhook-template-matchers.md` — How to build templates
- `webhook-timeout-error.md` — What to do when waitFor times out
- `recurse.md` — The polling primitive used internally by the registry
