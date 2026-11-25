/**
 * Every gateway's default profile has a door in the condensed profile menu
 * (#106017, #131632). A second registered gateway puts the sidebar rail in fleet mode,
 * which drops the default ↔ all home pill; past the condensed threshold the
 * menu is then the only way to reach a default by pointer. It listed every
 * AT-REST gateway's default (the remote's, or "This device") but never the
 * ACTIVE gateway's own, so on the usual setup the local default vanished
 * while the remote one stayed.
 *
 * Real app, real local backend, real second `hermes serve` registered as a
 * remote connection; checked from both sides of a switch.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import { coreAppEnv, createCoreSandbox, launchCoreApp, waitForInteractive, writeProviderHome } from './harness'
import { startScriptedProvider } from './provider'
import { startRemoteBackend } from './remote-helpers'

const REMOTE_ID = 'linuxhost'
const REMOTE_LABEL = 'Linux host'
// 13 local profiles + the remote's default square already crosses the
// condensed threshold (13) before the roster enumerates the remote's others.
const LOCAL_NAMED = Array.from({ length: 12 }, (_, i) => `local${i + 1}`)
const REMOTE_NAMED = ['inbox', 'research']

function seedProfiles(hermesHome: string, names: string[]): void {
  for (const name of names) {
    fs.mkdirSync(path.join(hermesHome, 'profiles', name), { recursive: true })
    fs.writeFileSync(path.join(hermesHome, 'profiles', name, 'config.yaml'), '')
  }
}

function writeConnectionsRegistry(userDataDir: string, url: string, token: string): void {
  fs.writeFileSync(
    path.join(userDataDir, 'connections.json'),
    JSON.stringify({
      version: 2,
      primary: 'local',
      launchMode: 'primary',
      lastUsed: 'local',
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        {
          id: REMOTE_ID,
          kind: 'remote',
          label: REMOTE_LABEL,
          url,
          authMode: 'token',
          token: { encoding: 'plain', value: token }
        }
      ]
    }),
    { encoding: 'utf8', mode: 0o600 }
  )
}

const trigger = (page: Page) => page.locator('[data-slot="profile-rail"] [data-slot="profile-dropdown"]')

async function openMenu(page: Page) {
  await expect(trigger(page), 'the fleet rail condensed to its menu').toBeVisible()
  await trigger(page).click()
  const menu = page.getByRole('menu')
  await expect(menu).toBeVisible()

  return menu
}

/** The active gateway's own default: a checked radio row with the home glyph. */
async function expectActiveDefault(page: Page, where: string) {
  const menu = await openMenu(page)
  const home = menu.getByRole('menuitemradio', { name: 'default', exact: true })
  await expect(home, `${where}: the active gateway's default has a row`).toBeVisible()
  await expect(home).toHaveAttribute('aria-checked', 'true')
  await expect(home.locator('.codicon-home')).toHaveCount(1)

  return menu
}

test("the condensed fleet menu lists the active gateway's default on both sides of a switch (#131632)", async () => {
  test.setTimeout(240_000)
  const provider = await startScriptedProvider()
  const clientBox = createCoreSandbox('fleet-default-client')
  const remoteBox = createCoreSandbox('fleet-default-remote')
  writeProviderHome(clientBox.hermesHome, provider.url)
  writeProviderHome(remoteBox.hermesHome, provider.url)
  seedProfiles(clientBox.hermesHome, LOCAL_NAMED)
  seedProfiles(remoteBox.hermesHome, REMOTE_NAMED)
  const remote = await startRemoteBackend(remoteBox)
  writeConnectionsRegistry(clientBox.userDataDir, remote.url, remote.token)

  const { app, page } = await launchCoreApp(coreAppEnv(clientBox))

  try {
    await waitForInteractive(app, page)

    await test.step('on This device: its default is listed beside the remote one', async () => {
      await expect(trigger(page)).toContainText('default')
      const menu = await expectActiveDefault(page, 'This device active')
      await expect(menu.getByRole('menuitemradio', { name: LOCAL_NAMED[0], exact: true })).toBeVisible()
      const remoteDefault = menu.getByRole('menuitem', { name: `default · ${REMOTE_LABEL}` })
      await expect(remoteDefault, 'the at-rest remote lists its default').toBeVisible({ timeout: 90_000 })
      await remoteDefault.click()
    })

    await test.step(`on ${REMOTE_LABEL}: its default is listed beside This device`, async () => {
      await expect
        .poll(
          async () => {
            const menu = await openMenu(page)
            const atRestLocal = await menu
              .locator('[data-slot="profile-dropdown-gateway"][data-connection-id="local"]')
              .count()
            await page.keyboard.press('Escape')

            return atRestLocal
          },
          { message: `the window re-homed onto ${REMOTE_LABEL}`, timeout: 120_000, intervals: [1_000, 2_000] }
        )
        .toBe(1)

      const menu = await expectActiveDefault(page, `${REMOTE_LABEL} active`)
      await expect(menu.getByRole('menuitemradio', { name: REMOTE_NAMED[0], exact: true })).toBeVisible()
      await expect(menu.getByRole('menuitem', { name: /^This device/ }), 'This device at rest').toBeVisible()
    })
  } finally {
    await app.close().catch(() => undefined)
    await remote.kill()
    await provider.close()
    clientBox.cleanup()
    remoteBox.cleanup()
  }
})
