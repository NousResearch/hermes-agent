import type { TestProjectConfiguration } from 'vitest/config'
import { defineConfig } from 'vitest/config'

// The desktop UI suite has hundreds of jsdom files. On the shared GitHub
// runner, Vitest's CPU-sized fork pool can exhaust the process/memory budget
// and terminate without a test summary. Keep CI deterministic while leaving
// local runs at Vitest's normal parallelism.
const ciUiWorkerLimit = process.env.CI ? { maxWorkers: 2, minWorkers: 1 } : {}

const reactUi: TestProjectConfiguration = {
  extends: './vite.config.ts',
  test: {
    name: 'ui',
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    // The first test in each file pays jsdom env init + full module transform,
    // which can exceed vitest's 5000ms default under CI/load. 15s gives the
    // cold start headroom without masking genuinely hung tests.
    testTimeout: 15_000,
    ...ciUiWorkerLimit
  }
}

const electronNative: TestProjectConfiguration = {
  test: {
    name: 'electron',
    environment: 'node',
    // `e2e/**/*.unit.test.ts` is the e2e HELPERS, not the specs: plain node
    // modules that should be provable without booting Electron. Playwright
    // ignores the same pattern so they run in exactly one runner.
    include: ['electron/**/*.test.ts', 'scripts/**.test.{ts,mjs}', 'e2e/**/*.unit.test.ts'],
    exclude: ['scripts/run-short-session-hang-repro.test.mjs']
  }
}

export default defineConfig({
  test: {
    projects: [reactUi, electronNative]
  }
})
