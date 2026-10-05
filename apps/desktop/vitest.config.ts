import type { TestProjectConfiguration } from 'vitest/config'
import { defineConfig } from 'vitest/config'

const reactUi: TestProjectConfiguration = {
  extends: './vite.config.ts',
  test: {
    name: 'ui',
    environment: 'jsdom',
    // Keep padding regressions observable instead of mocking the stylesheet away.
    css: { include: [/status-stack\.css$/] },
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    // React 19.1+ stubs `react`.act to undefined in the production build;
    // @testing-library/react's act() delegates to it and throws
    // "React.act is not a function" when NODE_ENV=production. Force the
    // development build for the test worker so `act` resolves to a real
    // function, regardless of any ambient NODE_ENV (e.g. a shell that
    // inherited NODE_ENV=production from a desktop launch).
    env: { NODE_ENV: 'development' },
    // The first test in each file pays jsdom env init + full module transform,
    // which can exceed vitest's 5000ms default under CI/load. 15s gives the
    // cold start headroom without masking genuinely hung tests. Hooks pay the
    // same cold cost when a beforeEach does `vi.resetModules()` + `await
    // import(...)` (65 files); one timed out at 10s on CI (#120318).
    testTimeout: 15_000,
    hookTimeout: 30_000
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
    // `*.windows-live.test.ts` runs in the dedicated `electron-live` project
    // below (serial files, not here) so no file executes twice.
    exclude: ['electron/**/*.windows-live.test.ts'],
    // Several suites here shell out to real `git` many times per test. Process
    // spawn on Windows costs far more than on POSIX, so the 5s default times
    // out work that is progressing normally rather than hung.
    testTimeout: process.platform === 'win32' ? 30_000 : 5_000
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
    globalSetup: ['./vitest.run-tmp.ts'],
    projects: [reactUi, electronNative, electronLive]
  }
})
