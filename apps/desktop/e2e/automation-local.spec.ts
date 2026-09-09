import { expect, test } from '@playwright/test'

import { setupMockBackend, waitForAppReady } from './fixtures'

test('automation creation is reachable through composer plus', async () => {
  test.setTimeout(180_000)
  const fixture = await setupMockBackend()

  try {
    await waitForAppReady(fixture, 120_000)
    const { page } = fixture
    page.on('console', msg => { if (msg.type() === 'error' || msg.type() === 'warning') {console.log('RENDER:', msg.text())} })
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

    // An active goal chains continuation turns while the judge says "continue",
    // so the session stays busy and the backend (correctly) refuses a goal edit
    // mid-turn. Pause the goal first: the pause command is not gated on the live
    // turn and stops the chain, leaving the session idle and the goal editable.
    await page.getByRole('button', { name: 'Goal actions', exact: true }).click()
    await page.getByRole('menuitem', { name: 'Pause goal', exact: true }).click()
    await expect(page.getByText(/Goal paused/, { exact: false }).first()).toBeVisible({ timeout: 30_000 })
    await page.screenshot({ path: '../../.automation-evidence/goal-paused.png' })

    const promptsBeforeEdit = fixture.mock.receivedPrompts.length
    await page.getByRole('button', { name: 'Goal actions', exact: true }).click()
    await page.getByRole('menuitem', { name: 'Edit goal', exact: true }).click()
    const goalPrompt = page.getByLabel('Goal prompt', { exact: true })
    await expect(goalPrompt).toHaveValue('Write a short greeting for the automation test')
    await goalPrompt.fill('Write a concise greeting for the edited automation test')
    await page.getByRole('button', { name: 'Save goal', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })
    await expect(page.locator('body')).toContainText('Write a concise greeting for the edited automation test')
    await expect.poll(() => fixture.mock.receivedPrompts.length).toBe(promptsBeforeEdit)
    await page.screenshot({ path: '../../.automation-evidence/goal-edited.png' })

    await page.getByRole('button', { name: 'Add files and actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: /Create automation/ }).click()
    await expect(page.getByRole('button', { name: 'Manage existing', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Start goal', exact: true })).toBeDisabled()
  } finally {
    await fixture.cleanup()
  }
})
