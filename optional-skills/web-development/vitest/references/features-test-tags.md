---
name: test-tags
description: Label tests with tags to filter runs and apply shared runner options
---

# Test Tags (4.1+)

Tags label tests so you can filter what runs and apply shared options (timeout, retry) to a *category* of tests that span many files. Reach for tags over projects when the category needs different timeouts/retries (not different pools/environments).

## Defining Tags

Tags **must be declared in config** — using an undefined tag throws unless `strictTags: false`. Each tag can carry options applied to every test marked with it:

```ts
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    tags: [
      { name: 'frontend', description: 'Frontend tests.' },
      { name: 'db', description: 'Database queries.', timeout: 60_000 },
      {
        name: 'flaky',
        retry: process.env.CI ? 3 : 0,
        timeout: 30_000,
        priority: 1, // lower priority number wins on conflicts
      },
    ],
    strictTags: true, // default: error on unknown tags
  },
})
```

Type-safe tag names (augment `TestTags`, include the file in your tsconfig):

```ts
import 'vitest'

declare module 'vitest' {
  interface TestTags {
    tags: 'frontend' | 'backend' | 'db' | 'flaky'
  }
}
```

## Applying Tags

```ts
import { describe, test } from 'vitest'

test('renders homepage', { tags: ['frontend'] }, () => {})

// Tags inherit from the parent suite
describe('API endpoints', { tags: ['backend'] }, () => {
  test('validates input', { tags: ['validation'] }, () => {
    // has both "backend" (inherited) and "validation"
  })
})
```

Tag every test in a file with a JSDoc `@module-tag` at the top of the file (applies to **all** tests in the file, not just the next one):

```ts
/**
 * @module-tag admin/pages/dashboard
 */
test('dashboard renders', () => {})
```

### Option conflict resolution

When several tags set the same option on a test, `priority` wins first (lower number), then array order. Options on the test itself always win:

```ts
test('flaky db test', { tags: ['flaky', 'db'] }) // timeout 30_000 (flaky priority 1), retry 3
test('override', { tags: ['flaky', 'db'], timeout: 120_000 }) // timeout 120_000, retry 3
```

## Filtering by Tag

Use `--tagsFilter` with an expression:

```bash
vitest --tagsFilter "frontend"
vitest --tagsFilter "db && !flaky"
vitest --tagsFilter "(unit || e2e) && !slow"
vitest --tagsFilter "api/*"            # wildcard
vitest --list-tags                     # list defined tags (=json for JSON)
```

Operators: `and`/`&&`, `or`/`||`, `not`/`!`, `*` wildcard, `()` grouping. Precedence: `not` > `and` > `or`. Multiple `--tagsFilter` flags combine with AND. Tag names can't be `and`/`or`/`not` or contain special chars/spaces.

Programmatic: pass `tagsFilter: ['frontend and backend']` to `startVitest`/`createVitest`.

## Checking the Filter at Runtime

Skip expensive setup when no matching tests are scheduled:

```ts
import { beforeAll, TestRunner } from 'vitest'

beforeAll(async () => {
  if (TestRunner.matchesTags(['db'])) {
    await seedDatabase()
  }
})
```

Returns `true` when the active `--tagsFilter` would include a test with those tags (or when no filter is active).

## Key Points

- Tags must be declared in config; the CLI filter is `--tagsFilter` (not `--tags`)
- Tags inherit from parent suites and `@module-tag` JSDoc comments
- Use tags for cross-cutting categories with shared timeout/retry; use [projects](advanced-projects.md) for different pools/environments
- `TestRunner.matchesTags` gates expensive `globalSetup`/`beforeAll` work

<!--
Source references:
- https://vitest.dev/guide/test-tags
- https://vitest.dev/config/tags
-->
