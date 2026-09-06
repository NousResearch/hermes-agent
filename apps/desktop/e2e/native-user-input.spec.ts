/**
 * Native asynchronous user-input E2E.
 *
 * This is intentionally a real agent → gateway → renderer journey. The mock
 * model emits `request_user_input`; the durable core request becomes a native
 * card, and the answer must reach the next model turn without inventing a
 * second composer submission.
 */

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { NATIVE_USER_INPUT_QUESTIONS, NATIVE_USER_INPUT_TRIGGER } from './mock-server'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('renders the native card above the composer and resumes after one answer', async () => {
  const page = fixture!.page
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })

  await composer.click()
  await composer.type(NATIVE_USER_INPUT_TRIGGER, { delay: 10 })
  await page.keyboard.press('Enter')

  const card = page.locator('[data-user-input-request]')
  await card.waitFor({ state: 'visible', timeout: 30_000 }).catch(async (error) => {
    const toolError = page.getByRole('button', { name: /Error Request User Input/ })
    if (await toolError.isVisible().catch(() => false)) {
      await toolError.click()
      const body = await page.locator('body').innerText()
      throw new Error(`Native user-input tool error state:\n${body}`, { cause: error })
    }
    throw error
  })
  await expect(card).toHaveCount(1)
  const placement = await card.evaluate(element => ({
    inComposerDock: Boolean(element.closest('[data-slot="composer-dock"]')),
    position: getComputedStyle(element).position
  }))
  expect(placement.inComposerDock).toBe(true)
  expect(placement.position).not.toBe('fixed')

  try {
    await expect.poll(async () => {
      const currentCard = await card.boundingBox()
      const currentComposer = await composer.boundingBox()
      if (!currentCard || !currentComposer) return Number.POSITIVE_INFINITY
      return currentCard.y + currentCard.height - currentComposer.y
    }, { timeout: 5_000 }).toBeLessThanOrEqual(1)
  } catch (error) {
    const geometry = await page.evaluate(() => {
      const describe = (element: Element | null) => {
        if (!element) return null
        const rect = element.getBoundingClientRect()
        const style = getComputedStyle(element)
        return {
          className: element.className,
          display: style.display,
          position: style.position,
          rect: { bottom: rect.bottom, height: rect.height, top: rect.top, width: rect.width },
          tag: element.tagName
        }
      }
      const card = document.querySelector('[data-user-input-request]')
      const composer = document.querySelector('[data-slot="composer-root"] [contenteditable="true"]')
      return {
        card: describe(card),
        composer: describe(composer),
        dock: describe(card?.closest('[data-slot="composer-dock"]') ?? null),
        cardParent: describe(card?.parentElement ?? null),
        composerParent: describe(composer?.parentElement ?? null)
      }
    })
    throw new Error(`Native user-input geometry:\n${JSON.stringify(geometry, null, 2)}`, { cause: error })
  }

  const cardBox = await card.boundingBox()
  const composerBox = await composer.boundingBox()
  expect(cardBox).not.toBeNull()
  expect(composerBox).not.toBeNull()
  const cardBottom = cardBox!.y + cardBox!.height
  const composerTop = composerBox!.y
  expect(cardBottom).toBeLessThanOrEqual(composerTop + 1)

  await expect(card.getByText(NATIVE_USER_INPUT_QUESTIONS[0].text)).toBeVisible()
  await expect(card.getByText(NATIVE_USER_INPUT_QUESTIONS[1].text)).toBeVisible()
  await expect(card.getByText('Coffee', { exact: true })).toBeVisible()
  await expect(card.getByText('Tea', { exact: true })).toBeVisible()

  const freeText = card.locator('input[data-user-input-kind="text"]')
  await freeText.focus()
  await freeText.fill('Keep this note for the next turn')
  await expect(freeText).toBeFocused()
  await expect(freeText).toHaveValue('Keep this note for the next turn')

  await card.getByText('Coffee', { exact: true }).click()
  const submit = card.locator('button[type="submit"]')
  await expect(submit).toBeEnabled()

  await page.screenshot({ path: test.info().outputPath('native-user-input-card-open.png'), fullPage: false })
  await submit.click()

  await expect(page.getByText('Native user input was recorded and the turn resumed.')).toBeVisible({ timeout: 60_000 })
  await expect(page.locator('[data-user-input-request]')).toHaveCount(0)
})
