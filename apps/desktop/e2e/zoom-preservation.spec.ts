/**
 * E2E regression: in-page route navigation must preserve the chosen UI scale.
 *
 * Desktop is a HashRouter over one file:// document, so every route is a
 * distinct URL to Chromium's per-URL zoom store, and a route with no record of
 * its own resolves to the host default (100%). In-page navigation fires no load
 * or window event, so nothing re-asserted the persisted level: switching
 * sessions dropped the window to 100% while Appearance kept reading the chosen
 * scale (#48658, #38854, #79863).
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const SCALE = 110

let fixture: MockBackendFixture | null = null

async function readZoomPercent(): Promise<number> {
  return fixture!.page.evaluate(async () => {
    const desktop = window as unknown as {
      hermesDesktop: { zoom: { get: () => Promise<{ percent: number }> } }
    }

    return (await desktop.hermesDesktop.zoom.get()).percent
  })
}

async function setZoomPercent(percent: number): Promise<void> {
  await fixture!.page.evaluate(target => {
    const desktop = window as unknown as {
      hermesDesktop: { zoom: { setPercent: (percent: number) => void } }
    }

    desktop.hermesDesktop.zoom.setPercent(target)
  }, percent)
  await expect.poll(readZoomPercent).toBe(percent)
}

async function gotoRoute(route: string): Promise<void> {
  const page = fixture!.page

  await page.evaluate(target => {
    window.location.hash = target
  }, route)
  await page.waitForFunction(target => window.location.hash === `#${target}`, route)
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a non-default UI scale survives navigation to never-zoomed routes', async () => {
  await setZoomPercent(SCALE)

  // Routes Chromium has no zoom record for — what opening a new session looks
  // like to the per-URL store. Pre-fix, the first hop reports 100%.
  const fresh = `/e2e-zoom-${Date.now()}`

  for (const route of [`${fresh}-one`, `${fresh}-two`, '/settings?tab=config%3Aappearance']) {
    await gotoRoute(route)
    await expect.poll(readZoomPercent, { message: `UI scale after navigating to ${route}` }).toBe(SCALE)
  }
})

test('Cmd/Ctrl+N preserves a non-default UI scale', async () => {
  const page = fixture!.page

  await gotoRoute('/settings')
  await setZoomPercent(SCALE)

  await page.evaluate(() => {
    ;(document.activeElement as HTMLElement | null)?.blur()
  })
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+N' : 'Control+N')
  await page.waitForFunction(() => window.location.hash === '#/')

  await expect.poll(readZoomPercent).toBe(SCALE)
})
