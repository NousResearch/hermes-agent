import '../fix-electron-tracing'

import { defineConfig } from '@playwright/test'

/**
 * The core Desktop suite: a small, deterministic, REQUIRED lane.
 *
 * Deliberately different from ../../playwright.config.ts:
 *  - retries: 0 — a required job that retries hides exactly the flake it
 *    should expose (the old lane retried and still went red for weeks).
 *  - no visual baselines / always-on screenshots; artifacts only on failure.
 *  - one worker: every spec owns a real Electron + `hermes serve`; running
 *    them concurrently on a loaded runner is the timing margin we refuse.
 *  - 180 s per test (green runs take 6-36 s): a stalled stream fails the one
 *    test fast instead of the 30-min job timeout cancelling the whole lane
 *    before it reports.
 */
export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.ts',
  timeout: 180_000,
  expect: { timeout: 60_000 },
  retries: 0,
  workers: 1,
  fullyParallel: false,
  reporter: [['list'], ['html', { open: 'never', outputFolder: '../../playwright-report/core' }]],
  outputDir: '../../test-results/core',
  use: {
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure'
  }
})
