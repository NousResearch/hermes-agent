import { type MockBackendFixture, setupMockBackend, waitForDesktopShellReady } from './fixtures'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null

const modes = [
  { id: 'skills', heading: 'Skills' },
  { id: 'toolsets', heading: 'Tools' },
  { id: 'connectors', heading: 'Connectors' },
  { id: 'plugins', heading: 'Plugins' }
] as const

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForDesktopShellReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('Capabilities desktop golden path', () => {
  test('keeps navigation, accessibility and visual shell consistent across all capability modes', async () => {
    const page = fixture!.page

    await page.locator('[data-tour="sidebar-nav-capabilities"]').click()

    const pageTabs = page.locator('[data-tour="page-tabs"]')
    await expect(pageTabs).toBeVisible()

    const capabilitiesPage = pageTabs.locator('xpath=ancestor::section[1]')
    const pageHeaderRow = pageTabs.locator('xpath=..')

    for (const mode of modes) {
      const tab = page.locator(`[data-tour="tab-${mode.id}"]`)

      await expect(tab).toBeVisible()
      await tab.focus()
      await expect(tab).toBeFocused()
      await tab.click()

      await expect(capabilitiesPage.getByRole('heading', { level: 1, name: mode.heading })).toBeVisible()

      const breadcrumb = capabilitiesPage.getByLabel('Capabilities')
      await expect(breadcrumb).toContainText('Capabilities')
      await expect(breadcrumb).toContainText(mode.heading)

      // PageSearchShell owns the shared capability search. A mode with no
      // available rows may hide it, but nested tabs must never render a
      // second competing SearchField.
      const searchInputs = pageHeaderRow.locator('input[type="text"]')
      expect(await searchInputs.count()).toBeLessThanOrEqual(1)

      await expectVisualSnapshot(page, {
        app: fixture!.app,
        name: `capabilities-${mode.id}`
      })
    }
  })
})
