/**
 * Visual snapshot helper — wraps `toHaveScreenshot` so visual diffs are
 * reported without failing the test suite.
 *
 * On CI, the JSON reporter + post-test script parse the results and post a
 * summary to the GitHub Actions step output, and diff images are uploaded
 * as artifacts.  This keeps visual regressions visible without gating PRs
 * on pixel-perfect matches.
 *
 * When a screenshot matches the baseline, nothing happens.  When it
 * differs, Playwright writes three images to the test's output dir:
 *   <name>-actual.png, <name>-expected.png, <name>-diff.png
 * These are picked up by the "Upload test results" artifact step.
 */
import { type ElectronApplication, expect, type Page } from '@playwright/test'

/** Fixed window dimensions for visual regression screenshots. */
export const VISUAL_WINDOW_WIDTH = 1220
export const VISUAL_WINDOW_HEIGHT = 800

export interface VisualSnapshotOptions {
  /** Snapshot name — defaults to the test title. */
  name?: string
  /** Full page screenshot (default) vs. viewport-only. */
  fullPage?: boolean
  /** Timeout in ms. */
  timeout?: number
  /** The Electron app handle — needed to force a fixed window size. */
  app?: ElectronApplication
}

/**
 * Force the Electron window to a fixed size so screenshots are comparable
 * across runs and CI environments.  Window managers (Hyprland, etc.) may
 * auto-tile or resize windows after launch; calling this right before the
 * screenshot ensures the viewport is always the expected size.
 */
async function forceFixedSize(app: ElectronApplication): Promise<void> {
  await app.evaluate(({ BrowserWindow }, { width, height }) => {
    const win = BrowserWindow.getAllWindows()[0]

    if (win) {
      win.unmaximize()
      // setMinimumSize must be ≤ the target, otherwise setSize is clamped.
      win.setMinimumSize(width, height)
      win.setSize(width, height, false)
      win.setBounds({ x: 0, y: 0, width, height })
    }
  }, { width: VISUAL_WINDOW_WIDTH, height: VISUAL_WINDOW_HEIGHT })
}

/**
 * Take a screenshot and compare it against the baseline.
 *
 * If the baseline doesn't exist yet (first run), Playwright creates it.
 * If it differs, the test logs a soft warning but does NOT fail — the diff
 * images are still generated for CI to surface.
 */
export async function expectVisualSnapshot(
  page: Page,
  options: VisualSnapshotOptions = {},
): Promise<void> {
  const { name, fullPage = false, timeout = 30_000, app } = options

  // Force the window to a fixed size right before the screenshot so it's
  // always comparable, regardless of WM resizing during the test.
  if (app) {
    await forceFixedSize(app)
    // Give the renderer a moment to relayout after the resize.
    await page.waitForTimeout(500)
  }

  // Playwright appends a platform suffix (e.g. "-linux") and requires
  // a .png extension on the name argument.  Auto-append it if missing.
  const snapshotName = name ? (name.endsWith('.png') ? name : `${name}.png`) : undefined

  try {
    if (snapshotName) {
      await expect(page).toHaveScreenshot(snapshotName, { fullPage, timeout })
    } else {
      await expect(page).toHaveScreenshot({ fullPage, timeout })
    }
  } catch (err) {
    // Don't fail the test — just log that a diff was detected.
    // The diff/actual/expected images are already written to the test
    // output directory by Playwright for the CI workflow to pick up.
    console.log(`[visual-diff] ${name ?? '(unnamed)'} — screenshot differs from baseline`)
    console.log(`  ${err instanceof Error ? err.message.split('\n')[0] : String(err)}`)
  }
}
