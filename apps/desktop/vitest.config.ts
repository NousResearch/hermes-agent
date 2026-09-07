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
    // which can exceed vitest's 5000ms default under CI/load. 15s gives the
    // cold start headroom without masking genuinely hung tests.
    testTimeout: 15_000
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
    // These use node:test and have dedicated npm scripts, not Vitest suites.
    // `*.windows-live.test.ts` runs in the dedicated `electron-live` project
    // below (serial files, not here) so no file executes twice.
    exclude: [
      'scripts/run-short-session-hang-repro.test.mjs',
      'scripts/tasks-scroll.test.mjs',
      'electron/**/*.windows-live.test.ts'
    ]
  }
}

const electronLive: TestProjectConfiguration = {
  test: {
    name: 'electron-live',
    environment: 'node',
    // These spawn real PowerShell/WMI process trees against the live OS
    // process table. Vitest's default file parallelism ran several of them
    // concurrently and starved the process-table probes under load; running
    // the files serially removes that contention. Tests within one file
    // still run concurrently as usual.
    include: ['electron/**/*.windows-live.test.ts'],
    exclude: ['scripts/run-short-session-hang-repro.test.mjs', 'scripts/tasks-scroll.test.mjs'],
    fileParallelism: false,
    // Matches the floor the live suites already assert per-test (e.g.
    // windows-update-force-release.windows-live.test.ts, 60_000 for its
    // un-annotated-longest arms; several go to 120_000 explicitly and keep
    // that override). This only raises the default for arms that don't
    // already declare their own timeout — no existing per-test timeout is
    // lowered.
    testTimeout: 60_000
  }
}

export default defineConfig({
  test: {
    projects: [reactUi, electronNative, electronLive]
  }
})
