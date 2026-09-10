/**
 * E2E batch clarify CONFIRM CHORD — Ctrl/Cmd+Enter locks the whole batch
 * without touching the "Confirm and continue" button, including from inside
 * an "Other" text field mid-typing (where plain Enter is a newline).
 *
 * Each test gets its OWN app fixture (one describe per test): the mock scripts
 * the batch clarify only on a conversation's FIRST completion — a second
 * identical trigger in the same session falls through to the canned reply so
 * the quiz cannot loop forever. The button-driven card flow lives in
 * batch-clarify.spec.ts; this file exercises the keyboard path end to end:
 * composer → gateway → agent → clarify tool → clarify.request → renderer.
 */

import { type Page } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { BATCH_CLARIFY_QUESTIONS, BATCH_CLARIFY_TRIGGER } from './mock-server'
import { expect, test } from './test'

/** Send the trigger and wait for the single live batch card to mount. */
async function openBatchCard(page: Page) {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })

  await composer.click()
  await composer.type(BATCH_CLARIFY_TRIGGER, { delay: 20 })
  await page.keyboard.press('Enter')

  const batchCard = page.locator('form[data-clarify-batch]')
  await batchCard.first().waitFor({ state: 'visible', timeout: 60_000 })
  await expect(batchCard).toHaveCount(1)

  return batchCard
}

test.describe('batch clarify confirm chord — lock from an Other box', () => {
  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('Ctrl+Enter from inside an Other box locks the whole batch', async () => {
    const { page } = fixture
    const batchCard = await openBatchCard(page)

    const confirmButton = batchCard.locator('button[type="submit"]')
    await expect(confirmButton).toContainText('Confirm and continue')
    // The chord hint rides the button, matching the approval bar's convention.
    await expect(confirmButton).toContainText(/⌘⏎|Ctrl⏎/u)

    // Stage q0 by pick, q1 by typing into its "Other" row…
    await batchCard.getByRole('button', { name: /Coffee/ }).click()
    const otherBoxes = batchCard.getByPlaceholder('Other (type your answer)')
    await expect(otherBoxes).toHaveCount(2)
    await otherBoxes.nth(1).click()
    await otherBoxes.nth(1).type('evening', { delay: 15 })

    // …and confirm from INSIDE that text field with the chord, never the button.
    await otherBoxes.nth(1).press('Control+Enter')

    // The settled card lists both questions with their locked answers.
    const settled = page.locator('[data-clarify-settled]')
    await settled.waitFor({ state: 'visible', timeout: 30_000 })
    await expect(settled.getByText(BATCH_CLARIFY_QUESTIONS[0].question)).toBeVisible()
    await expect(settled.getByText('Coffee', { exact: true })).toBeVisible()
    await expect(settled.getByText(BATCH_CLARIFY_QUESTIONS[1].question)).toBeVisible()
    await expect(settled.getByText('evening')).toBeVisible()

    // No live card lingers after settle.
    await expect(page.locator('form[data-clarify-batch]')).toHaveCount(0)
  })
})

test.describe('batch clarify confirm chord — park on the open question', () => {
  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('Ctrl+Enter with a question still open parks on it instead of locking', async () => {
    const { page } = fixture
    const batchCard = await openBatchCard(page)

    // Stage only q0, then fire the chord from the transcript with q1 blank.
    await batchCard.getByRole('button', { name: /Coffee/ }).click()
    await page.keyboard.press('Control+Enter')

    // Nothing locked yet — the first unanswered question owns the caret.
    const otherBoxes = batchCard.getByPlaceholder('Other (type your answer)')
    await expect(otherBoxes.nth(1)).toBeFocused()
    await expect(batchCard.locator('button[type="submit"]')).toBeDisabled()

    // Answer it and confirm: the same chord now completes the batch.
    await otherBoxes.nth(1).type('evening', { delay: 15 })
    await otherBoxes.nth(1).press('Control+Enter')

    const settled = page.locator('[data-clarify-settled]')
    await settled.waitFor({ state: 'visible', timeout: 30_000 })
    await expect(settled.getByText('evening')).toBeVisible()
  })
})
