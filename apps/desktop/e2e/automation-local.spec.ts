import { test, expect } from '@playwright/test'
import { setupMockBackend, waitForAppReady } from './fixtures'

test('automation creation is reachable through composer plus', async () => {
  test.setTimeout(180_000)
  const fixture = await setupMockBackend()
  try {
    await waitForAppReady(fixture, 120_000)
    const { page } = fixture
    page.on('console', msg => { if (msg.type() === 'error' || msg.type() === 'warning') console.log('RENDER:', msg.text()) })
    page.on('pageerror', error => console.log('PAGE:', error.stack))
    const input = page.locator('[data-slot="composer-rich-input"]').first()
    await input.fill('Hello. This is a local automation test.')
    await input.press('Enter')
    await expect(page.getByText('Hello from the mock inference server! The full boot chain is working.', { exact: true })).toBeVisible({ timeout: 30000 })
    await page.getByRole('button', { name: 'Add files and actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: /Create automation/ }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await page.screenshot({ path: '../../.automation-evidence/composer-dialog.png' })
    await page.getByLabel('Goal prompt', { exact: true }).fill('Write a short greeting for the automation test')
    await page.getByRole('button', { name: 'Start goal', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })
    await expect(page.locator('body')).toContainText('Write a short greeting for the automation test')
    await expect.poll(() => fixture.mock.receivedPrompts.some(text => text.includes('Write a short greeting for the automation test')), { timeout: 60_000 }).toBe(true)
    await page.screenshot({ path: '../../.automation-evidence/goal-created.png' })
    await page.getByRole('button', { name: 'Add files and actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: /Create automation/ }).click()
    await expect(page.getByRole('button', { name: 'Manage existing', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Start goal', exact: true })).toBeDisabled()
  } finally {
    await fixture.cleanup()
  }
})
