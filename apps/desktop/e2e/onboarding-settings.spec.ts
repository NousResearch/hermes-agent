import { readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

import { buildAppEnv, createSandbox, launchDesktop, setupNoProvider } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const { prepareWindowForInput } = createRequire(import.meta.url)(
  '../../../tests/install/e2e-assets/window-input.cjs',
) as { prepareWindowForInput: (app: ElectronApplication, page: Page) => Promise<void> }

test('input setup survives a fresh-install zoom restore before onboarding', async () => {
  const sandbox = createSandbox('cold-input')
  unlinkSync(path.join(sandbox.userDataDir, 'zoom-state.json'))
  writeFileSync(path.join(sandbox.hermesHome, 'config.yaml'), '# no provider\n', 'utf8')
  let app: ElectronApplication | undefined

  try {
    const launched = await launchDesktop(buildAppEnv(sandbox))
    app = launched.app
    const page = launched.page
    await page.waitForSelector('button', { state: 'attached' })
    await prepareWindowForInput(app, page)
    const later = page.getByRole('button', { name: /choose a provider later/i })
    await expect(later).toBeVisible({ timeout: 60_000 })
    const appWindow = await app.browserWindow(page)
    await appWindow.evaluate(win => win.emit('focus'))
    await expect.poll(() => appWindow.evaluate(win => win.webContents.getZoomFactor())).toBeCloseTo(1)
    await later.click({ timeout: 5_000 })
    await expect(later).toBeHidden()
  } finally {
    await app?.close().catch(() => undefined)
    sandbox.cleanup()
  }
})

// Exercise the install driver's input setup against the real renderer/backend,
// with no installer, update, credentials, or live user data.
for (const lifecycleEvent of ['focus', 'navigation'] as const) {
  test(`onboarding input zoom survives ${lifecycleEvent} and opens Settings`, async () => {
    const fixture = await setupNoProvider()
    const { app, page, sandbox } = fixture

    try {
      await prepareWindowForInput(app, page)
      const later = page.getByRole('button', { name: /choose a provider later/i })
      await expect(later).toBeVisible({ timeout: 60_000 })
      const zoomFile = path.join(sandbox.userDataDir, 'zoom-state.json')
      const savedLevel = () => JSON.parse(readFileSync(zoomFile, 'utf8')).zoomLevel as number
      await page.evaluate(() => {
        const desktop = (window as unknown as { hermesDesktop: { zoom: { setPercent: (percent: number) => void } } }).hermesDesktop
        desktop.zoom.setPercent(90)
      })
      await expect.poll(savedLevel).toBeCloseTo(Math.log(0.9) / Math.log(1.2))

      await prepareWindowForInput(app, page)
      const appWindow = await app.browserWindow(page)

      // The same lifecycle callback that fires when another window takes focus
      // must restore our input scale, not the original 90% preference.
      if (lifecycleEvent === 'focus') {
        await appWindow.evaluate(win => win.emit('focus'))
      } else {
        await page.evaluate(() => { window.location.hash = '#/settings' })
      }

      await expect.poll(() => appWindow.evaluate(win => win.webContents.getZoomFactor())).toBeCloseTo(1)
      expect(savedLevel()).toBe(0)
      await later.click({ timeout: 5_000 })
      await expect(later).toBeHidden()

      if (lifecycleEvent === 'navigation') {
        await page.evaluate(() => { window.location.hash = '#/' })
      }

      await page.getByRole('button', { name: 'Open settings', exact: true }).click({ timeout: 5_000 })
      await expect(page).toHaveURL(/settings/)
    } finally {
      await fixture.cleanup()
    }
  })
}
