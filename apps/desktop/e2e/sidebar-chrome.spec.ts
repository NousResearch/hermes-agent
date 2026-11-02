import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
})

test('sidebar tabs align with navigation and keep one collapse control', async () => {
  const info = test.info()

  const page = fixture.page
  const sessions = page.getByRole('tab', { name: 'Sessions', exact: true })
  const bots = page.getByRole('tab', { name: 'Bots', exact: true })
  const newSession = page.locator('[data-tour="sessions-sidebar"]').getByRole('button', { name: /^New session/ })
  await expect(sessions).toBeVisible()
  await expect(bots).toBeVisible()
  await newSession.hover()
  await info.attach('sidebar-navigation', { body: await page.screenshot(), contentType: 'image/png' })

  for (const width of [1220, 960]) {
    await fixture.app.evaluate(({ BrowserWindow }, width) => {
      BrowserWindow.getAllWindows()[0].setSize(width, 800)
    }, width)

    await expect.poll(async () => {
      const s = await sessions.boundingBox()
      const b = await bots.boundingBox()
      const n = await newSession.boundingBox()

      return s && b && n ? Math.max(Math.abs(s.x - n.x), Math.abs(b.x + b.width - n.x - n.width)) : Infinity
    }).toBeLessThanOrEqual(1)
  }

  const strip = sessions.locator('xpath=ancestor::*[@data-zone-tabstrip]')
  await strip.hover()
  await expect(strip.getByRole('button', { name: /minimize|hide/i })).toHaveCount(0)

  await bots.click()
  await expect(bots).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
  await sessions.click()
  await expect(newSession).toBeVisible()

  await page.getByRole('button', { name: /^Hide sidebar/ }).click()
  await expect(newSession).toBeHidden()
  await page.getByRole('button', { name: /^Show sidebar/ }).click()
  await expect(newSession).toBeVisible()
  await newSession.hover()
  await info.attach('sidebar-after-interaction', { body: await page.screenshot(), contentType: 'image/png' })
})
