/**
 * E2E regression: hash-route navigation must preserve the chosen UI scale.
 *
 * Electron resets file:// zoom to level 0 during in-page hash navigation on
 * macOS. Cmd+N moves Hermes from the current route to a fresh chat route, so
 * this exercises the exact user path rather than calling the zoom helper.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

let fixture: MockBackendFixture | null = null

async function readZoomPercent(): Promise<number> {
  return fixture!.page.evaluate(async () => {
    const desktop = window as unknown as {
      hermesDesktop: { zoom: { get: () => Promise<{ percent: number }> } }
    }

    return (await desktop.hermesDesktop.zoom.get()).percent
  })
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('Cmd+N preserves a non-default UI scale', async () => {
  const page = fixture!.page

  await page.evaluate(() => {
    window.location.hash = '/settings'
  })
  await page.waitForFunction(() => window.location.hash === '#/settings')

  await page.evaluate(() => {
    const desktop = window as unknown as {
      hermesDesktop: { zoom: { setPercent: (percent: number) => void } }
    }

    desktop.hermesDesktop.zoom.setPercent(110)
  })
  await expect.poll(readZoomPercent).toBe(110)

  await page.evaluate(() => {
    ;(document.activeElement as HTMLElement | null)?.blur()
  })
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+N' : 'Control+N')
  await page.waitForFunction(() => window.location.hash === '#/')

  await expect.poll(readZoomPercent).toBe(110)
})
