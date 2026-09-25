import {
  PACKAGED_BINARY_PATH,
  type PackagedAppFixture,
  packagedBinaryExists,
  setupPackagedApp,
  waitForAppReady
} from './fixtures'
import { expect, test } from './test'

/**
 * E2E smoke tests for the packaged Hermes desktop app.
 *
 * Launches the real packaged Electron binary (produced by `npm run pack` →
 * `electron-builder --dir`) with a real isolated backend and local mock
 * provider. The packaged renderer, sandboxed HERMES_HOME, and user data are
 * all independent from the developer's live desktop session.
 *
 * Skips if the packaged binary doesn't exist — run `npm run pack` first.
 */

let fixture: PackagedAppFixture | null = null

test.beforeAll(async () => {
  test.skip(!packagedBinaryExists(), `Built app binary not found: ${PACKAGED_BINARY_PATH}. Run 'npm run pack' first.`)

  fixture = await setupPackagedApp()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('HUD composer remains fully inside the transparent window', async () => {
  const geometryEpsilonPx = 1
  const hudPagePromise = fixture!.app.waitForEvent('window', { timeout: 30_000 })

  const opened = await fixture!.page.evaluate(async () => {
    const hud = (
      window as typeof window & {
        hermesDesktop?: { hud?: { open: (options: { sessionId: null }) => Promise<{ ok: boolean }> } }
      }
    ).hermesDesktop?.hud

    if (!hud) {
      throw new Error('HUD bridge is unavailable in the packaged app')
    }

    return hud.open({ sessionId: null })
  })

  expect(opened).toEqual({ ok: true })

  const hudPage = await hudPagePromise

  try {
    await waitForAppReady({ app: fixture!.app, page: hudPage }, 120_000)
    await hudPage.waitForSelector('[data-slot="composer-rich-input"]', { state: 'visible' })

    const geometry = await hudPage.evaluate(() => {
      const dock = document.querySelector<HTMLElement>('[data-slot="composer-dock"]')
      const input = document.querySelector<HTMLElement>('[data-slot="composer-rich-input"]')

      if (!dock || !input) {
        throw new Error('HUD composer did not render')
      }

      const dockRect = dock.getBoundingClientRect()
      const inputRect = input.getBoundingClientRect()

      return {
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
        dockLeft: dockRect.left,
        dockRight: dockRect.right,
        dockTop: dockRect.top,
        dockBottom: dockRect.bottom,
        inputLeft: inputRect.left,
        inputRight: inputRect.right,
        inputTop: inputRect.top,
        inputBottom: inputRect.bottom,
        // The bug class this guards: a build-time CSS optimization folding the
        // dock's identity `translate` override into `transform`, leaving
        // Tailwind's standalone `translate: -50%` live and shifting the dock
        // half a window off-screen. Surface the computed value so a failure
        // says WHY the dock moved, not just that it did.
        dockTranslate: getComputedStyle(dock).translate
      }
    })

    // Horizontal containment — the composer shifted half a window left when the
    // standalone `translate: -50%` survived optimization (#82214, #82233).
    // Windows DPI rounding may leave an edge within one CSS pixel of the
    // viewport without a visible escape; the regression shape is much larger.
    expect(geometry.dockLeft).toBeGreaterThanOrEqual(-geometryEpsilonPx)
    expect(geometry.inputLeft).toBeGreaterThanOrEqual(-geometryEpsilonPx)
    expect(geometry.dockRight).toBeLessThanOrEqual(geometry.viewportWidth + geometryEpsilonPx)
    expect(geometry.inputRight).toBeLessThanOrEqual(geometry.viewportWidth + geometryEpsilonPx)

    // Vertical containment — the toolbar/transcript clipping reported on
    // Windows (#82203) and macOS (#82214) is the same "composer escapes the
    // window" class on the other axis.
    expect(geometry.dockTop).toBeGreaterThanOrEqual(-geometryEpsilonPx)
    expect(geometry.inputTop).toBeGreaterThanOrEqual(-geometryEpsilonPx)
    expect(geometry.dockBottom).toBeLessThanOrEqual(geometry.viewportHeight + geometryEpsilonPx)
    expect(geometry.inputBottom).toBeLessThanOrEqual(geometry.viewportHeight + geometryEpsilonPx)

    // The dock's centering translate must be fully neutralized. Any live
    // percentage translate means the HUD override lost to the app's centering.
    // (Computed `translate` keeps percentages as-is, so this is assertable;
    // computed `transform` resolves to a matrix and is covered by the
    // geometric containment checks above.)
    expect(geometry.dockTranslate ?? 'none').not.toContain('%')
  } finally {
    await hudPage.close().catch(() => undefined)
  }
})
