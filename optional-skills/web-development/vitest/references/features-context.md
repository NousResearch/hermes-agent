---
name: test-context-fixtures
description: Test context, custom fixtures with test.extend (builder pattern), scopes, and test.override
---

# Test Context & Fixtures

## Built-in Context

Every test receives context as its first argument:

```ts
test('context', ({ task, expect, skip, signal, annotate }) => {
  console.log(task.name)        // Test metadata (readonly)
  expect(1).toBe(1)             // Expect bound to this test
  skip(condition, 'reason')     // Skip dynamically
})
```

Properties:
- `task` — test metadata (name, file, etc.)
- `expect` — expect bound to this test (required for concurrent snapshot tests)
- `skip(condition?, message?)` — skip the test
- `signal` (3.2+) — `AbortSignal` aborted on timeout/cancel/bail
- `annotate(message, type?, attachment?)` (3.2+) — attach reporter annotations
- `onTestFinished(fn)` / `onTestFailed(fn)` — per-test cleanup/handlers
- `bench` (v5) — benchmark fixture (only in `*.bench.ts` files)

## Custom Fixtures — Builder Pattern (4.1+, recommended)

`.extend(name, options?, fixture)` infers types automatically. Use `onCleanup` for teardown:

```ts
import { test as baseTest } from 'vitest'

export const test = baseTest
  // Plain value — type inferred as { port: number; host: string }
  .extend('config', { port: 3000, host: 'localhost' })
  // Function fixture — can read previously defined fixtures
  .extend('server', async ({ config }, { onCleanup }) => {
    const server = await startServer(config)
    onCleanup(() => server.close()) // runs after test/scope ends
    return server
  })

test('uses server', ({ config, server }) => {
  expect(server.url).toContain(String(config.port))
})
```

> `onCleanup` can be called **once per fixture**. For multiple resources, split into separate fixtures.

### Fixture Options

```ts
const test = baseTest
  .extend('metrics', { auto: true }, () => new Metrics())       // runs for every test
  .extend('config', { scope: 'worker' }, () => loadConfig())    // once per worker
  .extend('db', { scope: 'file' }, async ({ config }, { onCleanup }) => {
    const db = await createDatabase(config)
    onCleanup(() => db.close())
    return db
  })
  .extend('baseUrl', { injected: true }, () => 'http://localhost:3000') // overridable via config
```

## Object Syntax (Playwright-compatible)

Uses the `use()` callback; types must be declared manually:

```ts
const test = baseTest.extend<{ page: Page; baseUrl: string }>({
  page: async ({}, use) => {
    const page = await browser.newPage()
    await use(page)        // test runs here
    await page.close()     // cleanup after
  },
  baseUrl: 'http://localhost:3000',
})
```

Tuple form sets options: `fixture: [async ({}, use) => {…}, { scope: 'file' }]`.

## Fixture Scopes (3.2+)

| Scope | Lifetime | Can access |
|-------|----------|------------|
| `test` (default) | each test | worker + file + test fixtures + built-in context |
| `file` | once per file | worker + file fixtures |
| `worker` | once per worker process | only worker fixtures |

Only `test`-scoped fixtures can access the built-in context (`task`, `expect`, …). In file/worker fixtures use `expect.getState().testPath` for the file path. By default every file is its own worker, so `file` and `worker` behave the same unless [isolation is disabled](features-concurrency.md).

## Injected Fixtures (per-project values)

```ts
// fixtures.ts
const test = baseTest.extend('url', { injected: true }, '/default')

// vitest.config.ts — provide per project
defineConfig({
  test: {
    projects: [
      { test: { name: 'prod', provide: { url: 'https://prod' } } },
    ],
  },
})
```

Read raw provided values without fixtures via `import { inject } from 'vitest'`.

## Overriding Fixtures — test.override (4.1+)

`test.override` replaces fixture values for a suite and its children (replaces the deprecated `test.scoped`):

```ts
describe('production', () => {
  test
    .override('config', { port: 8080, host: 'api.example.com' })
    .override('debug', false)        // chainable

  test('uses prod config', ({ server }) => {
    expect(server.url).toBe('http://api.example.com:8080')
  })
})

// Function override (reads other fixtures) with cleanup
test.override('db', async ({ config }, { onCleanup }) => {
  const db = await createTestDatabase(config)
  onCleanup(() => db.drop())
  return db
})
```

You cannot introduce new fixtures or change `scope`/`auto` via `override`; use `test.extend` for new fixtures.

## Composing & Hooks

Extend an already-extended test, and use type-aware hooks on the extended `test`:

```ts
import { test as dbTest } from './db-test'

export const test = dbTest.extend('user', ({ db }) => db.createUser())

test.beforeEach(({ db }) => db.seed())            // sees fixtures
test.beforeAll(({ db }) => db.migrate())          // file/worker fixtures only (4.1+)
test.aroundAll(async (run, { db }) => db.tx(run))
```

## Key Points

- Prefer the **builder pattern** — types are inferred, cleanup via `onCleanup`
- Fixtures are lazy — only initialized when destructured
- Always destructure `{ db }` (not `context.db`)
- Use `{ scope: 'file' | 'worker' }` for expensive shared resources
- Use `test.override` (not `test.scoped`) to vary fixture values per suite
- Use `{ injected: true }` + project `provide` for per-project values

<!-- 
Source references:
- https://vitest.dev/guide/test-context.html
-->
