/**
 * Visual snapshot helper — wraps `toHaveScreenshot` so visual diffs are
 * reported without failing the test suite.
 *
 * On CI, the JSON reporter + post-test script parse the results and post a
 * summary to the GitHub Actions step output, and diff images are uploaded
 * as artifacts.  This keeps visual regressions visible without gating PRs
 * on pixel-perfect matches.
 *
 * The actual screenshot is always written to the test output dir so CI
 * artifacts include every screenshot — not just the ones that diffed.
 * When it differs, this helper also writes expected and diff images:
 *   <name>-actual.png, <name>-expected.png, <name>-diff.png
 */
import fs from 'node:fs'
import path from 'node:path'

import { type ElectronApplication, type Page, test } from '@playwright/test'

/** Fixed window dimensions for visual regression screenshots. */
export const VISUAL_WINDOW_WIDTH = 1220
export const VISUAL_WINDOW_HEIGHT = 800

export interface VisualSnapshotOptions {
  /** Snapshot name — defaults to the test title. */
  name?: string
  /** Full page screenshot vs. viewport-only (default). */
  fullPage?: boolean
  /** Timeout in ms. */
  timeout?: number
  /** The Electron app handle — used to size and decode screenshots. */
  app: ElectronApplication
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
  options: VisualSnapshotOptions,
): Promise<void> {
  const { name, fullPage = false, timeout = 30_000, app } = options

  // Force the window to a fixed size right before the screenshot so it's
  // always comparable, regardless of WM resizing during the test.
  await forceFixedSize(app)
  // Give the renderer a moment to relayout after the resize.
  await page.waitForTimeout(500)

  // Playwright appends a platform suffix (e.g. "-linux") and requires
  // a .png extension on the name argument.  Auto-append it if missing.
  const snapshotName = name ? (name.endsWith('.png') ? name : `${name}.png`) : undefined

  const info = test.info()
  const actual = await page.screenshot({ animations: 'disabled', caret: 'hide', fullPage, timeout })
  const baselinePath = info.snapshotPath(snapshotName ?? `${info.title}.png`)
  const outputName = (snapshotName ?? 'snapshot.png').replace(/\.png$/, '')

  if (info.config.updateSnapshots === 'all' || info.config.updateSnapshots === 'changed') {
    fs.mkdirSync(path.dirname(baselinePath), { recursive: true })
    fs.writeFileSync(baselinePath, actual)
    // Also write to the output dir so CI artifacts include the screenshot.
    fs.writeFileSync(info.outputPath(`${outputName}-actual.png`), actual)
    console.log(`[visual-baseline] updated ${baselinePath}`)
    return
  }

  if (!fs.existsSync(baselinePath)) {
    fs.writeFileSync(info.outputPath(`${outputName}-actual.png`), actual)
    console.log(`[visual-diff] ${name ?? '(unnamed)'} — no baseline available`)
    return
  }

  const expected = fs.readFileSync(baselinePath)
  const comparison = await app.evaluate(
    ({ nativeImage }, images) => {
      const actualImage = nativeImage.createFromBuffer(Buffer.from(images.actual, 'base64'))
      const expectedImage = nativeImage.createFromBuffer(Buffer.from(images.expected, 'base64'))
      const actualSize = actualImage.getSize()
      const expectedSize = expectedImage.getSize()

      if (actualSize.width !== expectedSize.width || actualSize.height !== expectedSize.height) {
        return { mismatchRatio: 1, diff: images.actual }
      }

      const actualPixels = actualImage.toBitmap()
      const expectedPixels = expectedImage.toBitmap()
      const diffPixels = Buffer.alloc(actualPixels.length)
      let mismatched = 0

      for (let i = 0; i < actualPixels.length; i += 4) {
        const different =
          Math.abs(actualPixels[i] - expectedPixels[i]) > 51 ||
          Math.abs(actualPixels[i + 1] - expectedPixels[i + 1]) > 51 ||
          Math.abs(actualPixels[i + 2] - expectedPixels[i + 2]) > 51 ||
          Math.abs(actualPixels[i + 3] - expectedPixels[i + 3]) > 51

        if (different) {
          mismatched++
          diffPixels[i + 2] = 255
        }
        diffPixels[i + 3] = 255
      }

      return {
        mismatchRatio: mismatched / (actualPixels.length / 4),
        diff: nativeImage.createFromBitmap(diffPixels, actualSize).toPNG().toString('base64'),
      }
    },
    { actual: actual.toString('base64'), expected: expected.toString('base64') },
  )

  // Always write the actual screenshot to the output dir so CI artifacts
  // include every screenshot — not just the ones that diffed.
  fs.writeFileSync(info.outputPath(`${outputName}-actual.png`), actual)

  if (comparison.mismatchRatio <= 0.01) {
    return
  }

  fs.writeFileSync(info.outputPath(`${outputName}-expected.png`), expected)
  fs.writeFileSync(info.outputPath(`${outputName}-diff.png`), Buffer.from(comparison.diff, 'base64'))
  console.log(
    `[visual-diff] ${name ?? '(unnamed)'} — ${(comparison.mismatchRatio * 100).toFixed(2)}% of pixels differ`,
  )
}
