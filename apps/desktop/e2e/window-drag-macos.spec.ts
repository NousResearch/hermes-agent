import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  findElectron,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { _electron, type ElectronApplication, expect, installErrorBannerGuard, test } from './test'

// This moves the system pointer. Run alone in an unlocked macOS desktop:
// HERMES_E2E_NATIVE_DRAG=1 npx playwright test e2e/window-drag-macos.spec.ts --workers=1
test('native macOS top-edge drag survives session typing', async () => {
  test.skip(process.platform !== 'darwin', 'Exercises native macOS window movement')
  test.skip(process.env.HERMES_E2E_NATIVE_DRAG !== '1', 'Native pointer input requires an explicit local run')
  const testInfo = test.info()
  test.skip(testInfo.config.workers !== 1, 'Native pointer input must run with --workers=1')
  test.setTimeout(150_000)
  const sandbox = createSandbox('native-drag')
  const nativeDriver = resolve(sandbox.root, 'native-drag')
  let mock: Awaited<ReturnType<typeof startMockServer>> | undefined
  let app: ElectronApplication | undefined

  const cleanup = async () => {
    try {
      await app?.close().catch(() => undefined)
      await mock?.close()
    } finally {
      sandbox.cleanup()
    }
  }

  try {
    execFileSync('swiftc', [resolve(import.meta.dirname, 'fixtures/window-drag-macos.swift'), '-o', nativeDriver], {
      timeout: 60_000
    })
    const trusted = execFileSync(nativeDriver, ['--check-accessibility'], { encoding: 'utf8' }).trim() === 'true'
    test.skip(!trusted, 'Native drag requires existing macOS Accessibility permission')

    mock = await startMockServer()
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)
    // Exercise the shipping 90% zoom with native GPU compositing enabled.
    writeFileSync(
      resolve(sandbox.userDataDir, 'zoom-state.json'),
      JSON.stringify({ zoomLevel: Math.log(0.9) / Math.log(1.2) })
    )
    app = await _electron.launch({
      executablePath: findElectron(),
      args: [resolve(import.meta.dirname, '..')],
      env: buildAppEnv(sandbox),
      cwd: resolve(import.meta.dirname, '..')
    })
    const page = await app.firstWindow()
    installErrorBannerGuard(page)
    const fixture = { app, page, mock, mockUrl: mock.url, sandbox, cleanup }
    await waitForAppReady(fixture, 120_000)
    const win = await app.browserWindow(page)
    await win.evaluate(w => {
      w.setBounds({ x: 300, y: 200, width: 1220, height: 800 })
      w.show()
      w.focus()
    })

    const probe = async (state: string, point: { x: number; y: number }) => {
      const before = await win.evaluate(w => ({
        bounds: w.getBounds(),
        pid: process.pid,
        zoom: w.webContents.getZoomFactor()
      }))

      execFileSync(
        nativeDriver,
        [
          String(before.pid),
          String(before.bounds.x + point.x * before.zoom),
          String(before.bounds.y + point.y * before.zoom),
          '80',
          '0'
        ],
        { timeout: 10_000 }
      )
      const after = await win.evaluate(w => w.getBounds())
      const evidence = { state, point, before, after, delta: after.x - before.bounds.x }
      console.log('NATIVE_DRAG', JSON.stringify(evidence))
      writeFileSync(testInfo.outputPath(state + '.json'), JSON.stringify(evidence, null, 2))
      await page.screenshot({ path: testInfo.outputPath(state + '.png') })

      return evidence.delta
    }

    expect(await probe('empty-top-edge', { x: 390, y: 4 })).toBeGreaterThan(60)
    const composer = page.locator('[data-slot="composer-rich-input"]:visible').first()
    await composer.click()
    await page.keyboard.press('Meta+t')
    const tab = page.locator('[data-zone-tabstrip="grp-main"] [data-tree-tab]').last()
    await expect(page.locator('[data-zone-tabstrip="grp-main"] [data-tree-tab]')).toHaveCount(2)
    await composer.click()
    await composer.pressSequentially('Keep this draft while moving the window')

    const point = await tab.evaluate(el => {
      const bounds = el.getBoundingClientRect()
      const header = el.closest('[data-zone-tabstrip]')!.getBoundingClientRect()

      return { x: bounds.left + bounds.width / 2, y: header.top + 4 }
    })

    expect(await probe('typing-top-edge', point)).toBeGreaterThan(60)
    await expect(composer).toHaveText('Keep this draft while moving the window')
  } finally {
    await cleanup()
  }
})
