import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'


test.describe('Sessions gateway switcher', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('shows the current gateway in the Sessions sidebar above the separate profile rail', async () => {
    const page = fixture.page
    const profileRail = page.locator('[data-slot=profile-rail]')
    const sessionsSidebar = page.locator('[data-slot=sidebar]').filter({ has: profileRail }).first()
    const switcher = sessionsSidebar.locator('[data-slot=connection-switcher]')

    await expect(profileRail).toBeVisible()
    await expect(switcher).toBeVisible()
    await expect(page.locator('[data-slot=connection-switcher]')).toHaveCount(1)

    const trigger = switcher.getByRole('button', { name: 'Registered gateways: This device' })
    await expect(trigger).toContainText('This device')

    const gatewayBox = await switcher.boundingBox()
    const profileBox = await profileRail.boundingBox()

    expect(gatewayBox).not.toBeNull()
    expect(profileBox).not.toBeNull()
    expect(gatewayBox!.y + gatewayBox!.height).toBeLessThanOrEqual(profileBox!.y)

    await trigger.click()
    await expect(page.getByRole('menuitem', { name: 'Manage gateways…' })).toBeVisible()
  })
})
