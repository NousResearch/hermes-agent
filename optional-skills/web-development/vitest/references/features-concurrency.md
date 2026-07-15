---
name: concurrency-parallelism
description: Concurrent tests, parallel execution, and sharding
---

# Concurrency & Parallelism

## File Parallelism

By default, Vitest runs test files in parallel across workers:

```ts
defineConfig({
  test: {
    // Run files in parallel (default: true)
    fileParallelism: true,
    
    // Max concurrent workers (v4: replaces maxThreads/maxForks; minWorkers removed)
    maxWorkers: 4,
    
    // Pool type: 'forks' (default), 'threads', 'vmForks', 'vmThreads'
    pool: 'forks',
  },
})
```

> **v4 pool rework:** `poolOptions` was removed — all pool settings are now top-level. `singleThread`/`singleFork` become `maxWorkers: 1, isolate: false`. VM `memoryLimit` is `vmMemoryLimit`. These can now be set **per project**.

## Concurrent Tests

Run tests within a file in parallel:

```ts
// Individual concurrent tests
test.concurrent('test 1', async ({ expect }) => {
  expect(await fetch1()).toBe('result')
})

test.concurrent('test 2', async ({ expect }) => {
  expect(await fetch2()).toBe('result')
})

// All tests in suite concurrent
describe.concurrent('parallel suite', () => {
  test('test 1', async ({ expect }) => {})
  test('test 2', async ({ expect }) => {})
})
```

**Important:** Use `{ expect }` from context for concurrent tests.

## Opting Out of Concurrency

`test.sequential`/`describe.sequential` were **removed in v5**. Use `{ concurrent: false }`:

```ts
describe.concurrent('mostly parallel', () => {
  test('parallel 1', async () => {})

  // Opt this test out of inherited concurrency
  test('must run alone', { concurrent: false }, async () => {})
})

// Or an entire suite
describe('sequential suite', { concurrent: false }, () => {
  test('first', () => {})
  test('second', () => {})
})
```

Set `sequence.concurrent: true` to make all tests concurrent by default.

## Max Concurrency

Limit concurrent tests:

```ts
defineConfig({
  test: {
    maxConcurrency: 5, // Max concurrent tests per file
  },
})
```

## Isolation

Each file runs in isolated environment by default:

```ts
defineConfig({
  test: {
    // Disable isolation for faster runs (less safe)
    isolate: false,
  },
})
```

## Sharding

Split tests across machines:

```bash
# Machine 1
vitest run --shard=1/3

# Machine 2
vitest run --shard=2/3

# Machine 3
vitest run --shard=3/3
```

### CI Example (GitHub Actions)

```yaml
jobs:
  test:
    strategy:
      matrix:
        shard: [1, 2, 3]
    steps:
      - run: vitest run --shard=${{ matrix.shard }}/3 --reporter=blob
      
  merge:
    needs: test
    steps:
      - run: vitest --merge-reports --reporter=junit
```

### Merge Reports

```bash
# Each shard outputs blob
vitest run --shard=1/3 --reporter=blob --coverage
vitest run --shard=2/3 --reporter=blob --coverage

# Merge all blobs
vitest --merge-reports --reporter=json --coverage
```

## Test Sequence

Control test order:

```ts
defineConfig({
  test: {
    sequence: {
      // Run tests in random order
      shuffle: true,
      
      // Seed for reproducible shuffle
      seed: 12345,
      
      // Hook execution order
      hooks: 'stack', // 'stack', 'list', 'parallel'
      
      // All tests concurrent by default
      concurrent: true,

      // Order projects/groups run in (3.2+); lower runs first
      groupOrder: 0,
    },
  },
})
```

## Shuffle Tests

Randomize to catch hidden dependencies:

```ts
// Via CLI
vitest --shuffle

// Per suite
describe.shuffle('random order', () => {
  test('test 1', () => {})
  test('test 2', () => {})
  test('test 3', () => {})
})
```

## Pools (v4)

`poolOptions` was removed; pool settings are now top-level and can be set per project:

```ts
defineConfig({
  test: {
    pool: 'forks',     // 'forks' (default) | 'threads' | 'vmForks' | 'vmThreads'
    maxWorkers: 8,
    isolate: true,     // threads/forks only; vm* pools are always isolated
    vmMemoryLimit: '512MB',
  },
})
```

For per-project parallelism/isolation settings, see [advanced-projects](advanced-projects.md).

## Bail on Failure

Stop after first failure:

```bash
vitest --bail 1    # Stop after 1 failure
vitest --bail      # Stop on first failure (same as --bail 1)
```

## Key Points

- Files run in parallel by default (`pool: 'forks'`); tests within a file run sequentially unless `.concurrent`
- `concurrent` only speeds up tests that **await** (I/O, timers); pure sync tests still block the thread
- Always use context's `expect` in concurrent tests
- Use `{ concurrent: false }` (not `.sequential`) to opt out
- `maxWorkers` (not `maxThreads`/`maxForks`); `poolOptions` removed in v4
- Sharding splits tests across CI machines; `--merge-reports` combines blob results

<!-- 
Source references:
- https://vitest.dev/guide/parallelism.html
- https://vitest.dev/guide/improving-performance.html
-->
