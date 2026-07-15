---
name: test-api
description: test/it function for defining tests with modifiers
---

# Test API

## Basic Test

```ts
import { expect, test } from 'vitest'

test('adds numbers', () => {
  expect(1 + 1).toBe(2)
})

// Alias: it
import { it } from 'vitest'

it('works the same', () => {
  expect(true).toBe(true)
})
```

## Async Tests

```ts
test('async test', async () => {
  const result = await fetchData()
  expect(result).toBeDefined()
})

// Promises are automatically awaited
test('returns promise', () => {
  return fetchData().then(result => {
    expect(result).toBeDefined()
  })
})
```

## Test Options

```ts
// Timeout (default: 5000ms)
test('slow test', async () => {
  // ...
}, 10_000)

// Or with options object
test('with options', { timeout: 10_000, retry: 2 }, async () => {
  // ...
})
```

## Test Modifiers

### Skip Tests

```ts
test.skip('skipped test', () => {
  // Won't run
})

// Conditional skip
test.skipIf(process.env.CI)('not in CI', () => {})
test.runIf(process.env.CI)('only in CI', () => {})

// Dynamic skip via context
test('dynamic skip', ({ skip }) => {
  skip(someCondition, 'reason')
  // ...
})
```

### Focus Tests

```ts
test.only('only this runs', () => {
  // Other tests in file are skipped
})
```

### Todo Tests

```ts
test.todo('implement later')

test.todo('with body', () => {
  // Not run, shows in report
})
```

### Failing Tests

```ts
test.fails('expected to fail', () => {
  expect(1).toBe(2) // Test passes because assertion fails
})
```

### Concurrent Tests

```ts
// Run tests in parallel
test.concurrent('test 1', async ({ expect }) => {
  // Use context.expect for concurrent tests
  expect(await fetch1()).toBe('result')
})

test.concurrent('test 2', async ({ expect }) => {
  expect(await fetch2()).toBe('result')
})
```

### Opt Out of Concurrency

`test.sequential` was **removed in v5**. Use `concurrent: false` to opt a test out of inherited or globally configured concurrency:

```ts
test('must run alone', { concurrent: false }, async () => {})
```

## Parameterized Tests

### test.each

```ts
test.each([
  [1, 1, 2],
  [1, 2, 3],
  [2, 1, 3],
])('add(%i, %i) = %i', (a, b, expected) => {
  expect(a + b).toBe(expected)
})

// With objects
test.each([
  { a: 1, b: 1, expected: 2 },
  { a: 1, b: 2, expected: 3 },
])('add($a, $b) = $expected', ({ a, b, expected }) => {
  expect(a + b).toBe(expected)
})

// Template literal
test.each`
  a    | b    | expected
  ${1} | ${1} | ${2}
  ${1} | ${2} | ${3}
`('add($a, $b) = $expected', ({ a, b, expected }) => {
  expect(a + b).toBe(expected)
})
```

### test.for

Preferred over `.each` - doesn't spread arrays:

```ts
test.for([
  [1, 1, 2],
  [1, 2, 3],
])('add(%i, %i) = %i', ([a, b, expected], { expect }) => {
  // Second arg is TestContext
  expect(a + b).toBe(expected)
})
```

## Test Context

First argument provides context utilities:

```ts
test('with context', ({ expect, skip, task, signal, annotate }) => {
  console.log(task.name)        // Test metadata
  skip(someCondition, 'reason') // Skip dynamically
  expect(1).toBe(1)             // Context-bound expect
})

// signal (3.2+): AbortSignal aborted on timeout/cancel/bail
test('aborts on timeout', async ({ signal }) => {
  await fetch('/resource', { signal })
}, 2000)

// annotate (3.2+): attach notes shown by the reporter
test('annotated', async ({ annotate }) => {
  await annotate('see issue #123', 'issues')
})
```

## Custom Test with Fixtures

Prefer the **builder pattern** (4.1+) for automatic type inference:

```ts
import { test as base } from 'vitest'

const test = base
  .extend('db', async ({}, { onCleanup }) => {
    const db = await createDb()
    onCleanup(() => db.close()) // runs after the test/scope
    return db
  })

test('query', async ({ db }) => {
  const users = await db.query('SELECT * FROM users')
  expect(users).toBeDefined()
})
```

See [features-context](features-context.md) for fixture scopes, `test.override`, and the Playwright-compatible object syntax.

## Retry Configuration

```ts
test('flaky test', { retry: 3 }, async () => {
  // Retries up to 3 times on failure
})

// Advanced retry options
test('with delay', {
  retry: {
    count: 3,
    delay: 1000,
    condition: /timeout/i, // Only retry on timeout errors
  },
}, async () => {})
```

## Tags

Tags must be declared in config first, then applied to tests (4.1+):

```ts
test('database test', { tags: ['db', 'slow'] }, async () => {})

// Run with a tag expression:
// vitest --tagsFilter "db && !flaky"
```

See [features-test-tags](features-test-tags.md) for defining tags and filter syntax.

## Benchmarks (v5)

`bench` is no longer a top-level import — it is a [test-context fixture](features-benchmarking.md) used inside `test()`:

```ts
// file must match benchmark.include (e.g. *.bench.ts)
test('sort', async ({ bench }) => {
  await bench('Array.sort', () => [3, 1, 2].sort()).run()
})
```

## Key Points

- Pass options as the **second argument**; the 3rd-arg options object was removed in v4 (a trailing timeout number is still allowed)
- Tests with no body are marked as `todo`
- `test.only` throws in CI unless `allowOnly: true`
- Use context's `expect` for concurrent tests and snapshots
- Function name is used as test name if passed as first arg
- `test.sequential` was removed in v5 — use `{ concurrent: false }`

<!-- 
Source references:
- https://vitest.dev/api/test.html
-->
