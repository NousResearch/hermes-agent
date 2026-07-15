---
name: vitest-configuration
description: Configure Vitest with vite.config.ts or vitest.config.ts
---

# Configuration

Vitest reads configuration from `vitest.config.ts` or `vite.config.ts`. It shares the same config format as Vite.

## Basic Setup

```ts
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // test options
  },
})
```

## Using with Existing Vite Config

Add Vitest types reference and use the `test` property:

```ts
// vite.config.ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
  },
})
```

## Merging Configs

If you have separate config files, use `mergeConfig`:

```ts
// vitest.config.ts
import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config'

export default mergeConfig(viteConfig, defineConfig({
  test: {
    environment: 'jsdom',
  },
}))
```

## Common Options

```ts
defineConfig({
  test: {
    // Enable global APIs (describe, it, expect) without imports
    globals: true,
    
    // Test environment: 'node', 'jsdom', 'happy-dom'
    environment: 'node',
    
    // Setup files to run before each test file
    setupFiles: ['./tests/setup.ts'],
    
    // Include patterns for test files
    include: ['**/*.{test,spec}.{js,ts,jsx,tsx}'],
    
    // Exclude patterns
    exclude: ['**/node_modules/**', '**/dist/**'],
    
    // Limit test discovery to a directory (faster than broad excludes)
    dir: './src',

    // Test timeout in ms
    testTimeout: 5000,
    
    // Hook timeout in ms
    hookTimeout: 10000,
    
    // Coverage configuration (v4+: define `include`, no more `all`)
    coverage: {
      provider: 'v8', // or 'istanbul'
      reporter: ['text', 'html'],
      include: ['src/**/*.ts'],
    },
    
    // Run each file in an isolated module graph (threads/forks pools only)
    isolate: true,
    
    // Pool: 'forks' (default), 'threads', 'vmForks', 'vmThreads'
    pool: 'forks',
    
    // v4+: pool options are top-level (poolOptions was removed)
    maxWorkers: 4,
    fileParallelism: true,
    
    // Automatically clear mocks between tests
    clearMocks: true,
    
    // Restore spies created with vi.spyOn between tests
    restoreMocks: true,
    
    // Retry failed tests
    retry: 0,
    
    // Stop after first failure
    bail: 0,
  },
})
```

## v4/v5 Config Changes

- **Pool default is `forks`** (child processes), not `threads`.
- **`poolOptions` removed** — `maxThreads`/`maxForks` are now top-level `maxWorkers`; `singleThread`/`singleFork` become `maxWorkers: 1, isolate: false`; VM `memoryLimit` is `vmMemoryLimit`. `minWorkers` was removed.
- **`workspace` removed** — use [`projects`](advanced-projects.md). `vitest.workspace.ts` no longer supported.
- **`coverage.all` and `coverage.extensions` removed** — by default only covered files are reported; set `coverage.include` explicitly.
- **Simplified `exclude`** — only `node_modules`/`.git` excluded by default. Use `test.dir` to scope discovery, or spread `configDefaults.exclude`.
- **Config not looked up from parent dirs** — pass `--config` explicitly when running from a subdirectory.
- **`.vitest` artifact dir** — blob reports (`.vitest/blob/`), attachments (`.vitest/attachments/`), and HTML report now live under a single `.vitest/` directory; add one entry to `.gitignore`.
- `deps.optimizer.web` renamed to `deps.optimizer.client`; `deps.inline`/`deps.external` moved under `server.deps`.

## Conditional Configuration

Use `mode` or `process.env.VITEST` for test-specific config:

```ts
export default defineConfig(({ mode }) => ({
  plugins: mode === 'test' ? [] : [myPlugin()],
  test: {
    // test options
  },
}))
```

## Projects (Monorepos)

Run different configurations in the same Vitest process:

```ts
defineConfig({
  test: {
    projects: [
      'packages/*',
      {
        test: {
          name: 'unit',
          include: ['tests/unit/**/*.test.ts'],
          environment: 'node',
        },
      },
      {
        test: {
          name: 'integration',
          include: ['tests/integration/**/*.test.ts'],
          environment: 'jsdom',
        },
      },
    ],
  },
})
```

## Key Points

- Vitest uses Vite's transformation pipeline - same `resolve.alias`, plugins work
- `vitest.config.ts` takes priority over `vite.config.ts`
- Use `--config` flag to specify a custom config path (required from subdirectories in v5)
- `process.env.VITEST` is set to `true` when running tests
- Test config uses `test` property, rest is Vite config
- v4 requires **Vite >= 6** and **Node >= 20**; v5 is currently in beta

<!-- 
Source references:
- https://vitest.dev/guide/#configuring-vitest
- https://vitest.dev/config/
-->
