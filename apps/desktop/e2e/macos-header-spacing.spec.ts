import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

test.skip(process.platform !== 'darwin', 'Native macOS window controls')

let fixture: MockBackendFixture

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
})

test('traffic lights and toolbar share a center without touching sidebar tabs', async () => {
  const info = test.info()

  const { app, page } = fixture
  const sidebar = page.getByRole('tab', { name: 'Sessions', exact: true })
  const toggle = page.getByRole('button', { name: /^Hide sidebar/ })
  await expect(sidebar).toBeVisible()
  await info.attach('macos-header', { body: await page.screenshot(), contentType: 'image/png' })

  for (const zoom of [1, 0.5, 0.8, 1.25]) {
    // Use the same IPC as Settings → Scale.
    await page.evaluate(zoom => window.hermesDesktop!.zoom!.setPercent(zoom * 100), zoom)

    await expect.poll(async () => {
      const native = await app.evaluate(({ BrowserWindow }) => {
        const win = BrowserWindow.getAllWindows()[0]

        return { position: win.getWindowButtonPosition(), zoom: win.webContents.getZoomFactor() }
      })

      const button = await toggle.boundingBox()

      return button && native.position
        ? Math.abs((button.y + button.height / 2) * native.zoom - (native.position.y + 7))
        : Infinity
    }).toBeLessThanOrEqual(1)

    const native = await app.evaluate(({ BrowserWindow }) => {
      const win = BrowserWindow.getAllWindows()[0]

      return { position: win.getWindowButtonPosition(), zoom: win.webContents.getZoomFactor() }
    })

    const button = await toggle.boundingBox()
    expect(button!.x * native.zoom - native.position!.x - 58).toBeGreaterThanOrEqual(7.9)
    const tab = await sidebar.boundingBox()
    expect(tab!.y).toBeGreaterThanOrEqual(43.9)
  }

  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setFullScreen(true))
  await expect.poll(() => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isFullScreen())).toBe(true)
  await expect(sidebar).toBeVisible()
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setFullScreen(false))
  await expect.poll(() => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isFullScreen())).toBe(false)
  await expect(toggle).toBeVisible()
})
