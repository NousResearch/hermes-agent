/**
 * The plugin contract in the real app: the contract probe (src/contrib/__fixtures__/contract-probe,
 * one contribution in every SDK area) is dropped into the sandbox's `desktop-plugins/` folder the
 * way a user or agent installs a disk plugin, and each shell × interface mode is checked for where
 * its contributions land. Real Electron and real `hermes serve`; the probe needs no backend.
 *
 * Status-bar contributions are asserted as they CURRENTLY behave (the redesign has no strip; its
 * items sit in the rail's System popover) — their home in the redesign is an open decision.
 */
import fs from 'node:fs'
import path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const PROBE_SOURCE = path.join(__dirname, '..', 'src', 'contrib', '__fixtures__', 'contract-probe', 'plugin.js')
const SHELL_KEY = 'hermes.desktop.agentShell.v1'
const MODE_KEY = 'hermes.desktop.interfaceMode.v1'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  const folder = path.join(fixture.sandbox.hermesHome, 'desktop-plugins', 'contract-probe')
  fs.mkdirSync(folder, { recursive: true })
  fs.copyFileSync(PROBE_SOURCE, path.join(folder, 'plugin.js'))
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

const marker = (area: string) => fixture!.page.getByTestId(`probe-${area}`)

for (const shell of ['classic', 'redesign'] as const) {
  for (const mode of ['simple', 'advanced'] as const) {
    test(`the probe's contributions reach the ${shell} shell in ${mode} mode`, async () => {
      test.setTimeout(240_000)
      const { page } = fixture!
      await waitForAppReady(fixture!, 120_000)
      await page.evaluate(
        ([shellKey, modeKey, classic, simple]) => {
          if (classic) {
            localStorage.setItem(shellKey, 'false')
          } else {
            localStorage.removeItem(shellKey)
          }

          if (simple) {
            localStorage.setItem(modeKey, 'simple')
          } else {
            localStorage.removeItem(modeKey)
          }

          location.hash = '#/'
        },
        [SHELL_KEY, MODE_KEY, shell === 'classic', mode === 'simple'] as const
      )
      await page.reload()
      await waitForAppReady(fixture!, 120_000)

      // The chat: titlebar slots, composer render areas, the floating pane.
      for (const area of ['titleBar.left', 'titleBar.right', 'composer.top', 'composer.bottom', 'pane-floating']) {
        await expect(marker(area).first(), area).toBeVisible({ timeout: 60_000 })
      }

      await expect(page.getByLabel('Probe tool').first()).toBeVisible()

      // The page, reached from its nav row (the rail in the redesign, the sidebar in classic).
      await page.getByRole('button', { name: 'Probe page' }).first().click()
      await expect(marker('route')).toBeVisible()
      await expect(marker('workspace.pageHeader')).toBeVisible()
      await expect(marker('i18n')).toHaveText(shell === 'redesign' ? 'Contract probe (redesign)' : 'Contract probe')
      await expect(marker('viewport')).toContainText('"narrow":false')

      // Settings ▸ Plugins and Appearance.
      await page.evaluate(() => (location.hash = '#/settings?tab=plugins&plugin=contract-probe'))
      await expect(marker('settings.plugins')).toBeVisible()
      await page.evaluate(() => (location.hash = '#/settings?tab=appearance'))
      await expect(marker('appearance.extra')).toBeVisible()

      // Titlebar slots survive the overlay destinations in the redesign (parked), as permanent mounts.
      await expect(marker('titleBar.right')).toHaveCount(shell === 'redesign' ? 1 : 0)

      // host.openWorkspace through the palette command.
      await page.evaluate(() => (location.hash = '#/'))
      await page.keyboard.press('ControlOrMeta+k')
      await page.keyboard.type('Probe: open workspace')
      await page.keyboard.press('Enter')
      await expect(marker('openWorkspace')).toBeVisible()

      // CURRENT behaviour, pending the redesign's status-bar decision: classic shows the strip's
      // items in Advanced; the redesign lists them only in the rail's System popover.
      if (shell === 'classic' && mode === 'advanced') {
        await expect(marker('statusBar.right')).toBeVisible()
      }

      if (shell === 'redesign') {
        await expect(marker('statusBar.right')).toHaveCount(0)
      }
    })
  }
}
