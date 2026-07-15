---
name: benchmarking
description: Write benchmarks with the v5 bench test-context fixture (Tinybench)
---

# Benchmarking (v5)

In v5 the benchmark API was rewritten: `bench` is **no longer a top-level import**. It is a [test-context fixture](features-context.md) used inside a regular `test()`, available only in files matched by `benchmark.include` (default `**/*.{bench,benchmark}.?(c|m)[jt]s?(x)`). Benchmarks are powered by [Tinybench](https://github.com/tinylibs/tinybench).

## Defining & Running

```ts
import { expect, test } from 'vitest'

test('parse performance', async ({ bench }) => {
  // bench() registers; .run() executes and returns the result
  const result = await bench('parse', () => {
    const data = JSON.parse('{"key":"value"}')
    use(data) // consume the result — engines may eliminate dead code
  }).run()

  expect(result.throughput.mean).toBeGreaterThan(10_000)
})
```

Run benchmarks:

```bash
vitest bench           # only benchmarks (implicitly enables them)
vitest bench parser    # filter by filename
vitest bench -t JSON   # filter by test name
```

Set `benchmark: { enabled: true }` to run them alongside regular tests in a separate isolated group.

## Comparing Implementations

```ts
test('compare parsers', async ({ bench }) => {
  const result = await bench.compare(
    bench('JSON.parse', () => { JSON.parse(input) }),
    bench('custom', { beforeEach: () => reset() }, () => { customParse(input) }),
    { iterations: 100, time: 1000 }, // shared Tinybench options (last arg)
  )

  // Assertion matchers (delta avoids flaky failures)
  expect(result.get('JSON.parse')).toBeFasterThan(result.get('custom'), { delta: 0.1 })
  expect(result.get('custom')).toBeSlowerThan(result.get('JSON.parse'))
})
```

`bench.compare` interleaves iterations to reduce environmental bias and prints a comparison table after the test.

## Storing & Replaying Baselines

```ts
test('compare against baseline', async ({ bench }) => {
  await bench.compare(
    bench('current', { writeResult: './benchmarks/parse.json' }, () => parse(input)),
    bench.from('previous', './benchmarks/parse.json'),       // reads a stored result, no run
    bench.from('remote', () => fetch(url).then(r => r.json())),
  )
})
```

- `writeResult` overwrites the JSON file on every successful run (no skip-when-cached).
- `bench.from(name, source)` reads a stored result without invoking any function.
- For multi-project workspaces, pass `{ perProject: true }` and use `${projectName}` in `writeResult` paths to collect a cross-project comparison table.

## Stability Notes

- Benchmark files run sequentially and never in parallel; `retry` and the `delta` option reduce flakiness.
- Consume the result inside the bench fn — JS engines eliminate side-effect-free code.
- In Node mode every imported binding goes through Vite's module-runner getter; store hot references locally (`const _parse = parse`), benchmark the built package, or disable `experimental.viteModuleRunner` for the bench project.

## v5 Migration

- `bench` top-level import → `({ bench })` from the test context
- `bench.skip/only/todo` removed → use `test.skip/only/todo` on the surrounding test
- `benchmark.reporters`/`outputFile`/`compare`/`outputJson` and `--compare`/`--outputJson` removed → use `--reporter=json --outputFile` (JSON now has a `benchmarks` field)

## Key Points

- Benchmarks live in `*.bench.ts` files and run inside `test()` via `{ bench }`
- Use `bench.compare` + `toBeFasterThan`/`toBeSlowerThan` (with `delta`) for relative perf
- Persist baselines with `writeResult` and replay with `bench.from`

<!--
Source references:
- https://vitest.dev/guide/benchmarking
- https://vitest.dev/guide/test-context#bench
-->
