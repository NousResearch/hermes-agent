# Condition-Based Waiting

Adapted from `obra/superpowers` (skills/systematic-debugging). Used by `diagnose` when a flaky test or a race condition is the bug — or when a guessed `sleep` inside a test is itself the flake.

## Core principle

Tests and scripts that guess timing (`sleep`, `setTimeout`, `Thread.sleep`) pass on fast machines and fail under load or in CI. **Wait for the actual condition you care about, not a guess about how long it takes.**

## When to use

- Test has arbitrary delays and is flaky (pass/fail varies by machine or load)
- Tests time out when run in parallel
- Waiting for an async operation to complete

Do NOT use when the test is verifying *timing behaviour itself* (debounce windows, throttle intervals) — an arbitrary delay there is the subject under test; document WHY the chosen duration is correct.

## Pattern

```typescript
// ❌ Guessing at timing — passes locally, flakes in CI
await new Promise(r => setTimeout(r, 50));
expect(getResult()).toBeDefined();

// ✅ Waiting for the condition — fast when fast, patient when slow
await waitFor(() => getResult() !== undefined, 'result');
expect(getResult()).toBeDefined();
```

Generic polling helper:

```typescript
async function waitFor<T>(
  condition: () => T | undefined | null | false,
  description: string,
  timeoutMs = 5000,
): Promise<T> {
  const start = Date.now();
  for (;;) {
    const result = condition();
    if (result) return result;
    if (Date.now() - start > timeoutMs) {
      throw new Error(`Timeout waiting for ${description} after ${timeoutMs}ms`);
    }
    await new Promise(r => setTimeout(r, 10));
  }
}
```

Common wait patterns:

| Scenario | Condition |
|---|---|
| Event emitted | `events.find(e => e.type === 'DONE')` |
| State reached | `machine.state === 'ready'` |
| Count reached | `items.length >= 5` |
| File exists | `fs.existsSync(path)` |

Equivalents exist in most stacks — .NET: `SpinWait`/`Task.Run` poll loops or `WaitUntil`; Python: `tenacity`/`for` loop with `time.sleep(0.01)`; Playwright: `expect(...).toHaveText()` / `waitForSelector` (already condition-based); bash: `until ...; do sleep 0.1; done` with a deadline.

## Common mistakes

- **Polling too fast** (`sleep 1ms`) — wastes CPU; 10ms is a good default.
- **No timeout** — always bound the wait and name the condition in the error.
- **Stale data** — call the getter inside the loop; don't cache state before it.
- **Timeout sized to the dev machine** — the timeout is a failure bound, not the expected duration; give CI headroom.

## When an arbitrary delay IS correct

Sequence matters: first `waitFor` the triggering condition, then — only if the system ticks on a known clock — wait a *documented* duration (e.g. "200ms = 2 ticks at 100ms"). Comment the reasoning; undocumented sleeps are future flakes.
