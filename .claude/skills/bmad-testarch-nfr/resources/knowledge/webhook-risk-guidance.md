# Webhook Testing Risk Guidance

## Principle

Webhook integration points are high-risk boundaries — they represent asynchronous side effects that cross service boundaries. A missing or malformed webhook means a downstream system never received its trigger. Default risk level: **P2 × I3** (medium probability, high impact = Risk Score 6) → must be covered by integration tests.

## When Webhook Tests Are Required

Webhook tests are **required** (not optional) when:

| Condition                                                          | Rationale                                                              |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Application publishes events to external subscribers               | External consumers depend on correct payload shape and delivery timing |
| Event-driven architecture (Kafka/SQS/event bus → webhook delivery) | The delivery pipeline is a risk boundary; delivery failures are silent |
| Payment, order, or notification side effects                       | Business-critical; missed webhooks = missed transactions               |
| Integration with third-party services via webhooks                 | Breaking payload changes won't surface in unit or component tests      |
| Any async side effect that a consumer polls-on or reacts-to        | Polling tests (`recurse`) can mask webhook delivery failures entirely  |

## Risk Scoring

```
Risk = Probability × Impact

Probability factors (P1–P3):
  P1 (low):    Webhook system is mature, well-tested, no history of failures
  P2 (medium): Kafka pipeline, multiple consumers, new integrations
  P3 (high):   New delivery mechanism, external third-party webhooks, no retry logic

Impact factors (I1–I3):
  I1 (low):    Non-critical notifications (e.g. audit logs)
  I2 (medium): Feature-level side effects (e.g. search index updates)
  I3 (high):   Business-critical events (payments, orders, compliance)
```

Default webhook integrations: **P2 × I3 = 6** → High → must be tested.

## What a Complete Webhook Test Looks Like

A complete webhook test covers:

1. **Happy path**: Action fires → webhook arrives with correct payload
2. **Sequential events (drain pattern)**: Preceding event drained before asserting on next
3. **Parallel isolation**: Template scoped by entity ID — workers don't cross-contaminate
4. **Timeout/error shape**: `WebhookTimeoutError` tested for negative path coverage
5. **Cleanup verification**: Fixture auto-cleans; no leaked webhooks after test

**Minimal complete example** (from playwright-utils E2E suite):

```typescript
// Template factories scoped by ID — parallel safety
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

test('movie deletion triggers a webhook with correct payload', async ({ authToken, addMovie, deleteMovie, webhookRegistry }) => {
  const movie = generateMovieWithoutId();
  const { body: createResponse } = await addMovie(authToken, movie);
  const movieId = createResponse.data.id;

  // Drain: consume the create webhook before testing the delete path
  await webhookRegistry.waitFor(movieCreated(movieId));

  await deleteMovie(authToken, movieId);
  const webhook = await webhookRegistry.waitFor(movieDeleted(movieId));

  expect(webhook.body).toMatchObject({
    event: 'movie.deleted',
    data: { id: movieId, name: movie.name },
  });
});
```

## Common Failure Patterns

| Failure pattern                        | Root cause                                             | How the module addresses it                                                  |
| -------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Test passes but webhook never verified | Test asserted on status endpoint, not delivery         | `waitFor` forces assertion on actual webhook arrival                         |
| Flaky under `fullyParallel: true`      | `full-reset` cleanup deletes another worker's webhooks | `matched-only` strategy — only matched webhooks are deleted                  |
| Timeout gives no useful information    | No payload inspection on failure                       | `WebhookTimeoutError.receivedWebhooks` snapshot                              |
| Template matches wrong test's webhook  | Template not scoped by entity ID                       | Template factories accept ID parameter; `matchPredicate` for complex scoping |
| Test hangs at 30s default timeout      | Webhook not arriving; pipeline is slow                 | Use `withTimeout()` and `withInterval(500)` per template                     |
| Journal grows unbounded                | No cleanup strategy configured                         | Configure `cleanupStrategy` in `webhookConfig`; fixture auto-cleans          |

## Risk Mitigation Checklist (for TA assessment)

When a system uses webhooks, verify the test suite covers:

- [ ] Happy path for each event type that has an external subscriber
- [ ] Template factories scoped by entity ID (parallel-safe)
- [ ] Drain pattern applied to all sequential event assertions
- [ ] Cleanup strategy matches provider capability: `matched-only` for providers that support `deleteById` (e.g. WireMock); `full-reset` with serial execution or an isolated provider instance per worker for MockServer/Mockoon
- [ ] Timeout values appropriate for the delivery pipeline latency (Kafka pipelines need 15s+)
- [ ] `WebhookTimeoutError` imported and tested in negative path coverage
- [ ] Mock server (WireMock/MockServer/Mockoon) in Docker Compose / test infra

## Related Fragments

- `webhook-testing-fundamentals.md` — Why webhook tests are hard
- `webhook-module-setup.md` — Fixture wiring for each provider
- `webhook-template-matchers.md` — Template and matcher patterns
- `risk-governance.md` — Risk scoring framework
- `probability-impact.md` — P×I scale definitions
