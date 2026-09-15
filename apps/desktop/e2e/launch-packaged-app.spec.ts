import {
  PACKAGED_BINARY_PATH,
  type PackagedAppFixture,
  packagedBinaryExists,
  setupPackagedApp,
  waitForAppReady,
} from './fixtures'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

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
  test.skip(
    !packagedBinaryExists(),
    `Built app binary not found: ${PACKAGED_BINARY_PATH}. Run 'npm run pack' first.`,
  )

  fixture = await setupPackagedApp()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('window opens with the Hermes title', async () => {
  const title = await fixture!.page.title()
  expect(title).toContain('Hermes')
})

test('renderer loads and shows DOM content', async () => {
  const page = fixture!.page
  await page.waitForSelector('#root', { state: 'attached', timeout: 30_000 })
  const childCount = await page.locator('#root > *').count()
  expect(childCount).toBeGreaterThan(0)
})

test('boots to the app UI, not the QueryClient error boundary (#95560)', async () => {
  const page = fixture!.page
  await page.waitForSelector('#root', { state: 'attached', timeout: 30_000 })

  // Wait until the root has real content (boot overlay fades, app paints) —
  // the error boundary also paints, so assert on its absence explicitly.
  await page.waitForFunction(
    () => (document.getElementById('root')?.textContent ?? '').trim().length > 0,
    undefined,
    { timeout: 60_000 },
  )

  const text = await page.locator('#root').textContent()
  // The #95560 crash: a duplicate @tanstack/react-query runtime made the
  // QueryClientProvider's context invisible to useQuery, so the app hit the
  // error boundary at launch. Neither the boundary headline nor the throw
  // message may appear on a healthy boot.
  expect(text).not.toContain('No QueryClient set')
  expect(text).not.toContain('Something broke in the interface')
})

test('HUD composer remains fully inside the transparent window', async () => {
  const geometryEpsilonPx = 1
  const hudPagePromise = fixture!.app.waitForEvent('window', { timeout: 30_000 })

  const opened = await fixture!.page.evaluate(async () => {
    const hud = (window as typeof window & {
      hermesDesktop?: { hud?: { open: (options: { sessionId: null }) => Promise<{ ok: boolean }> } }
    }).hermesDesktop?.hud

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
        dockTranslate: getComputedStyle(dock).translate,
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

test('boot progress overlay fades out or shows error state', async () => {
  const page = fixture!.page
  await page.waitForFunction(
    () => {
      const root = document.getElementById('root')

      if (!root) {
        return false
      }

      const text = root.textContent ?? ''

      // Error path: boot failure overlay renders an error message.
      if (text.includes('error') || text.includes('Error') || text.includes('failed')) {
        return true
      }

      // Success path: overlay disappears and the app renders. If there's
      // no "boot" / "starting" / "installing" text visible, boot has
      // completed (either to the main UI or to onboarding).
      const bootIndicators = ['starting', 'resolving', 'spawning', 'waiting', 'installing']
      const lower = text.toLowerCase()

      return !bootIndicators.some((word) => lower.includes(word))
    },
    undefined,
    { timeout: 60_000 },
  )
})

test('can capture a screenshot for the CI artifact', async () => {
  if (!fixture) {
    test.skip(true, 'Previous test failed — no app running')

    return
  }

  // Visual snapshot — won't fail on diff, just logs + generates diff image
  await expectVisualSnapshot(fixture!.page, { name: 'packaged-app-booted', timeout: 10_000, app: fixture!.app })
})
