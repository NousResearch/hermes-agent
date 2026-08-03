import './e2e/fix-electron-tracing'

import { defineConfig, type ReporterDescription } from '@playwright/test'

/**
 * Visual regression testing config.
 *
 * Screenshots are compared against baselines.  On `main`, baselines are
 * generated with `--update-snapshots` and cached.  On PRs, the cached
 * baselines are restored and screenshots are compared — but tests DON'T
 * fail on visual diffs (see `expectVisualSnapshot` in visual-snapshot.ts).
 * Instead, diffs are surfaced in the CI step summary and uploaded as
 * artifacts for human review.
 *
 * To update baselines after an intentional UI change:
 *   npx playwright test --update-snapshots
 */
const reporters: ReporterDescription[] = [
  ['list'],
  ['html', { open: 'never', outputFolder: 'playwright-report' }],
]

if (process.env.CI) {
  reporters.push(['json', { outputFile: 'playwright-report/results.json' }])
}

export default defineConfig({
  /* Test files live under e2e/ so they never collide with the vitest suite
   * under src/ or the node:test files under electron/. */
  testDir: './e2e',
  /* The desktop app can take a while to bootstrap on cold CI runners — 90 s
   * per test gives us headroom without masking real hangs. */
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  /* Each test gets its own worker so the Electron process is fully isolated. */
  fullyParallel: false,
  reporter: reporters,
  use: {
    screenshot: 'on',
    trace: { mode: 'on', screenshots: true, snapshots: true, sources: true },
    // Emulate prefers-reduced-motion: reduce so all CSS transitions and
    // animations resolve instantly. This prevents boot/connecting overlays
    // from being mid-fade when a screenshot fires, and skips JS-driven exit
    // choreography in components that check matchMedia (onboarding, connecting
    // overlay, DecodeText). Without this, screenshots capture the loading bar
    // or overlay at a transient opacity because the text-content check fires
    // before the visual transition finishes.
    contextOptions: {
      reducedMotion: 'reduce',
    },
  },
  expect: {
    toHaveScreenshot: {
      // 1% of pixels may differ — absorbs sub-pixel font rendering variance
      // between local and CI environments.
      maxDiffPixelRatio: 0.01,
      animations: 'disabled',
      caret: 'hide',
      // Per-channel threshold for "close enough" — anti-aliasing differences.
      threshold: 0.2,
    },
  },
})
