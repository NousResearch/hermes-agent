import type { TestProjectConfiguration } from 'vitest/config'
import { defineConfig } from 'vitest/config'

const reactUi: TestProjectConfiguration = {
  extends: './vite.config.ts',
  test: {
    name: 'ui',
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    // The first test in each file pays jsdom env init + full module transform,
    // which can exceed vitest's 5000ms default under CI/load. 15s gave the
    // cold start headroom without masking genuinely hung tests.
    //
    // TEMPORARY: raised again to 30s because this job moved from a 32-core
    // Larger Runner to the standard 4-core `ubuntu-latest` (GitHub Free has
    // no Larger Runners) and 15s was no longer enough headroom — e.g.
    // toolset-config-panel.test.tsx hit ~15.1s on a single render-heavy
    // test alone under the new, much weaker CPU budget.
    testTimeout: 30_000
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
