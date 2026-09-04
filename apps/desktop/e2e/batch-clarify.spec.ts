/**
 * E2E batch clarify test — the multi-question clarify card must mount ONCE.
 *
 * Regression coverage for the duplicated-card bug: `tool.start` carries the
 * model's tool_call_id while `clarify.request` carries a gateway-generated
 * request_id. A batch payload has no top-level `question`, so the two rows
 * only merge when the correlation key comes from the question list
 * (`batchClarifyMatchValue` in lib/chat-messages/tool-parts.ts). Before that
 * fix this exact flow rendered two identical interactive cards.
 *
 * The flow runs the real chain: composer → gateway → agent → clarify tool →
 * clarify.request event → renderer, against the mock inference server.
 */

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { BATCH_CLARIFY_QUESTIONS, BATCH_CLARIFY_TRIGGER } from './mock-server'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('batch clarify card', () => {
  test('renders exactly one card and completes via per-question locks', async () => {
    const page = fixture!.page
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })

    await composer.click()
    await composer.type(BATCH_CLARIFY_TRIGGER, { delay: 20 })
    await page.keyboard.press('Enter')

    // The live batch form marks itself with data-clarify-batch=<count>.
    const batchCard = page.locator('form[data-clarify-batch]')
    await batchCard.first().waitFor({ state: 'visible', timeout: 60_000 })

    // THE regression assertion: one card, not two.
    await expect(batchCard).toHaveCount(1)
    await expect(batchCard).toHaveAttribute('data-clarify-batch', String(BATCH_CLARIFY_QUESTIONS.length))

    // Both questions render inside the single card.
    for (const entry of BATCH_CLARIFY_QUESTIONS) {
      await expect(batchCard.getByText(entry.question)).toHaveCount(1)
    }

    // Each question text also appears exactly once in the whole transcript —
    // catches a duplicate that mounts outside a form[data-clarify-batch].
    for (const entry of BATCH_CLARIFY_QUESTIONS) {
      await expect(page.getByText(entry.question)).toHaveCount(1)
    }

    // Answer both questions: stage picks locally (no server traffic yet).
    const confirmButton = batchCard.locator('button[type="submit"]')
    await expect(confirmButton).toContainText('Confirm and continue')
    await expect(confirmButton).toBeDisabled()

    await batchCard.getByRole('button', { name: /Coffee/ }).click()
    await expect(confirmButton).toBeDisabled()

    await batchCard.getByRole('button', { name: /Morning/ }).click()
    await expect(confirmButton).toBeEnabled()

    // ONE confirm submits the whole batch.
    await confirmButton.click()

    // The settled card lists both questions with their locked answers.
    const settled = page.locator('[data-clarify-settled]')
    await settled.waitFor({ state: 'visible', timeout: 30_000 })
    await expect(settled.getByText(BATCH_CLARIFY_QUESTIONS[0].question)).toBeVisible()
    await expect(settled.getByText('Coffee', { exact: true })).toBeVisible()
    await expect(settled.getByText(BATCH_CLARIFY_QUESTIONS[1].question)).toBeVisible()
    await expect(settled.getByText('Morning', { exact: true })).toBeVisible()

    // And still no duplicate live card lingering after settle.
    await expect(page.locator('form[data-clarify-batch]')).toHaveCount(0)
  })
})
