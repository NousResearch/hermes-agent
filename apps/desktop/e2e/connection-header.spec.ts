/**
 * Real built Electron shell + real Hermes backend + mock inference provider.
 * Run from apps/desktop:
 *   npm run build && npx playwright test e2e/connection-header.spec.ts --workers=1 --reporter=list
 */
import * as fs from 'node:fs'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, waitForAppReady } from './fixtures'
import { collectErrorBanners, expect, test } from './test'

let fixture: MockBackendFixture | undefined

test.setTimeout(180_000)

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = undefined
})

async function launchWithConnections(multiple: boolean, localLabel = 'This device'): Promise<MockBackendFixture> {
  const sandbox = createSandbox('connection-header')
  const mock = await startMockServer()
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'connections.json'),
    JSON.stringify({
      version: 2,
      primary: 'local',
      launchMode: 'primary',
      lastUsed: 'local',
      connections: [
        { id: 'local', kind: 'local', label: localLabel },
        // Saved but not selected: this test opens the selector, never dials it.
        ...(multiple ? [{ id: 'homelab', kind: 'remote', label: 'Homelab', url: mock.url, authMode: 'token' }] : [])
      ]
    })
  )

  try {
    const { app, page } = await launchDesktop(buildAppEnv(sandbox))

    return {
      app,
      page,
      mock,
      mockUrl: mock.url,
      sandbox,
      cleanup: async () => {
        try {
          expect(await collectErrorBanners(page)).toEqual([])
        } finally {
          try {
            await app.close()
          } finally {
            await mock.close()
            sandbox.cleanup()
          }
        }
      }
    }
  } catch (error) {
    await mock.close()
    sandbox.cleanup()
    throw error
  }
}

test('connection header stays above Sessions and Bots and leaves titlebar tools clickable', async () => {
  fixture = await launchWithConnections(true)
  await waitForAppReady(fixture, 120_000)
  const { page } = fixture
  const header = page.locator('[data-tree-group] [data-slot="connection-switcher"]')
  const switcher = header.getByRole('button', { name: /^Registered gateways:/ })
  const sessions = page.getByRole('tab', { name: 'Sessions', exact: true })
  const bots = page.getByRole('tab', { name: 'Bots', exact: true })

  await expect(switcher).toBeVisible()
  await sessions.click()
  await expect(sessions).toHaveAttribute('aria-selected', 'true')

  for (const tab of [sessions, bots]) {
    const headerBox = await header.boundingBox()
    const tabBox = await tab.boundingBox()
    expect(headerBox).not.toBeNull()
    expect(tabBox).not.toBeNull()
    expect(headerBox!.y + headerBox!.height).toBeLessThanOrEqual(tabBox!.y)
  }

  // Compare rendered columns, rather than freezing an absolute x coordinate.
  const navLabel = page.locator('[data-tour="sidebar-nav-new-session"]')
  const navButton = page.locator('[data-sidebar="menu-button"]').filter({ has: navLabel })
  const headerIcon = await header.locator('[data-slot="connection-glyph"] > svg').boundingBox()
  const navIcon = await navButton.locator(':scope > .codicon').boundingBox()
  const headerLabel = await switcher.locator('[data-connection-name]').boundingBox()
  const navLabelBox = await navLabel.boundingBox()
  expect(headerIcon).not.toBeNull()
  expect(navIcon).not.toBeNull()
  expect(headerLabel).not.toBeNull()
  expect(navLabelBox).not.toBeNull()
  expect(headerIcon!.x).toBeCloseTo(navIcon!.x, 0)
  expect(headerLabel!.x).toBeCloseTo(navLabelBox!.x, 0)

  await bots.click()
  await expect(bots).toHaveAttribute('aria-selected', 'true')
  await expect(switcher).toBeVisible()
  await switcher.click()
  await expect(page.getByRole('menuitemradio', { name: 'Homelab', exact: true })).toBeVisible()
  await page.screenshot({ path: test.info().outputPath('connection-header-bots-menu.png') })
  // The open menu is modal (the rest of the page is aria-hidden), so close it with an outside pointer press
  // (no re-selection side effects) before asserting the tab state.
  // Electron has no emulated viewport (viewportSize() is null), so read the window size from the page.
  const viewport = await page.evaluate(() => ({ height: window.innerHeight, width: window.innerWidth }))
  await page.mouse.click(Math.round(viewport.width * 0.7), Math.round(viewport.height * 0.3))
  await expect(page.getByRole('menu')).toBeHidden()
  await expect(bots).toHaveAttribute('aria-selected', 'true')

  // Use actual pointer input and observable effects, never force/DOM click.
  const titlebar = page.locator('[data-titlebar-cluster]')
  await titlebar.getByRole('button', { name: 'Hide sidebar', exact: true }).click()
  await expect(header).toBeHidden()
  await titlebar.getByRole('button', { name: 'Show sidebar', exact: true }).click()
  await expect(switcher).toBeVisible()
  await expect(bots).toHaveAttribute('aria-selected', 'true')
  await page.screenshot({ path: test.info().outputPath('connection-header-bots.png') })
  await titlebar.getByRole('button', { name: 'Open settings', exact: true }).click()
  await expect(page).toHaveURL(/\/settings/)
})

test('one saved connection has no zone header', async () => {
  fixture = await launchWithConnections(false)
  await waitForAppReady(fixture, 120_000)
  const { page } = fixture
  const bots = page.getByRole('tab', { name: 'Bots', exact: true })
  await expect(page.getByRole('tab', { name: 'Sessions', exact: true })).toBeVisible()
  await expect(bots).toBeVisible()
  await expect(page.locator('[data-tree-group] [data-slot="connection-switcher"]')).toHaveCount(0)
  await bots.click()
  await expect(bots).toHaveAttribute('aria-selected', 'true')
  await expect(page.locator('[data-tree-group] [data-slot="connection-switcher"]')).toHaveCount(0)
  await page.screenshot({ path: test.info().outputPath('connection-header-single.png') })
})

test('a long local name truncates with visible suffixes in the sidebar and status bar', async () => {
  // Registry labels are capped at 64 characters, so stay under the cap to seed the label verbatim.
  const localLabel = 'Studio workstation with an extraordinarily long device name'
  fixture = await launchWithConnections(true, localLabel)
  await waitForAppReady(fixture, 120_000)
  const { page } = fixture
  const header = page.locator('[data-tree-group] [data-slot="connection-switcher"]')
  const row = header.getByRole('button', { name: `Registered gateways: ${localLabel} (this device)` })
  const suffix = row.locator('[data-connection-suffix]')
  const name = row.locator('[data-connection-name]')
  await expect(row).toBeVisible()
  await expect(suffix).toHaveText('(this device)')
  await expect(suffix).toBeVisible()
  await expect(name).toHaveText(localLabel)

  const sidebarBox = await header.evaluate(element => {
    const box = element.closest('[data-tree-group]')!.getBoundingClientRect()

    return { x: box.x, y: box.y, width: box.width, height: box.height }
  })

  const statusbar = page.locator('[data-slot="statusbar"]')
  const compact = statusbar.getByRole('button', { name: `Registered gateways: ${localLabel} (this device)` })
  const compactName = compact.locator('[data-connection-name]')
  const compactSuffix = compact.locator('[data-connection-suffix]')
  await expect(compact).toBeVisible()
  await expect(compactName).toHaveText(localLabel)
  await expect(compactSuffix).toHaveText('(this device)')
  await expect(compactSuffix).toBeVisible()

  const compactBox = (await compact.boundingBox())!
  const compactSuffixBox = (await compactSuffix.boundingBox())!
  const statusbarBox = (await statusbar.boundingBox())!
  expect(compactBox.height).toBeCloseTo(statusbarBox.height, 1)

  const rowBox = (await row.boundingBox())!
  const suffixBox = (await suffix.boundingBox())!

  for (const [inner, outer] of [
    [rowBox, sidebarBox],
    [suffixBox, rowBox],
    [compactBox, statusbarBox],
    [compactSuffixBox, compactBox]
  ]) {
    expect(inner.x).toBeGreaterThanOrEqual(outer.x)
    expect(inner.x + inner.width).toBeLessThanOrEqual(outer.x + outer.width)
    expect(inner.y).toBeGreaterThanOrEqual(outer.y)
    expect(inner.y + inner.height).toBeLessThanOrEqual(outer.y + outer.height)
  }

  for (const [nameElement, suffixElement] of [
    [name, suffix],
    [compactName, compactSuffix]
  ]) {
    await expect.poll(() => nameElement.evaluate(element => element.scrollWidth > element.clientWidth)).toBe(true)
    // Truncated, not collapsed: part of the name stays readable next to the suffix.
    expect((await nameElement.boundingBox())!.width).toBeGreaterThanOrEqual(24)
    await expect(nameElement).toHaveCSS('text-overflow', 'ellipsis')
    await expect(nameElement).toHaveCSS('overflow', 'hidden')
    await expect(suffixElement).toHaveCSS('flex-shrink', '0')
    expect(await suffixElement.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true)
  }

  await page.screenshot({ path: test.info().outputPath('connection-header-long-name.png') })
  await name.hover()
  await expect(page.getByRole('tooltip')).toHaveText(`${localLabel} (this device)`)
})
