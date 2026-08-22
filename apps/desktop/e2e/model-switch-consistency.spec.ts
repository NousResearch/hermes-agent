/**
 * Model-switch E2E — prove the model painted by Desktop is the model sent to
 * inference, through the real Electron → hermes serve → provider path.
 */

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const ORIGINAL_MODEL = 'mock-model'
const SELECTED_MODEL = 'mock-model-alt'
const SELECTED_PROMPT = 'E2E model switch selected route'

let fixture: MockBackendFixture | null = null

test.setTimeout(180_000)

test.beforeAll(async () => {
  test.setTimeout(180_000)
  fixture = await setupMockBackend({
    mockServer: { modelIds: [ORIGINAL_MODEL, SELECTED_MODEL] }
  })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  test.setTimeout(180_000)
  await fixture?.cleanup()
  fixture = null
})

test('the selected model is acknowledged, painted, and used for inference', async () => {
  test.setTimeout(180_000)
  const { mock, page } = fixture!
  const composer = page.locator('[contenteditable="true"]').first()
  const submit = page.locator('[data-slot="composer-root"] button[type="submit"]')

  await composer.waitFor({ state: 'visible', timeout: 10_000 })

  const modelPill = page.locator('[data-slot="composer-root"] button[aria-label*="Model ·"]').first()
  await expect(modelPill).toHaveAttribute('aria-label', new RegExp(`${ORIGINAL_MODEL}$`))
  await modelPill.click()

  const search = page.getByRole('textbox', { name: 'Search models' })
  await search.fill(SELECTED_MODEL)
  await expect(page.getByRole('menuitem', { name: /Mock Model Alt/ })).toBeVisible({ timeout: 30_000 })

  await search.press('Enter')

  await expect(modelPill).toHaveAttribute('aria-label', new RegExp(`${SELECTED_MODEL}(?:\\s|$)`), {
    timeout: 30_000
  })

  expect(mock.receivedModels).toHaveLength(0)
  await composer.click()
  await composer.type(SELECTED_PROMPT)
  await expect(composer).toContainText(SELECTED_PROMPT)
  await submit.click()

  await expect(page.getByText(SELECTED_PROMPT, { exact: true })).toBeVisible({ timeout: 30_000 })
  await expect.poll(() => mock.receivedModels.length, { timeout: 60_000 }).toBeGreaterThan(0)
  expect(mock.receivedModels.at(-1)).toBe(SELECTED_MODEL)
  await expect(page.getByText(/mock inference server/i).last()).toBeVisible({ timeout: 60_000 })
})
