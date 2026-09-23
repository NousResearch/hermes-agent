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
 *  - generous per-test timeout; every wait inside is event-driven with its
 *    own deadline, so a long timeout never slows a green run.
 */
export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.ts',
  timeout: 600_000,
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
