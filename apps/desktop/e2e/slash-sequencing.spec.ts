import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const FIRST_SESSION_MARKER = 'E2E_SLASH_SEQUENCE_FIRST_SESSION'
const GOAL_TEXT = 'E2E slash sequence goal'
const ACTIVE_SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'

let fixture: MockBackendFixture | null = null

const activeSurface = () => fixture!.page.locator(ACTIVE_SURFACE).last()

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('desktop runs /goal on the fresh session created by a leading /new', async () => {
  test.setTimeout(180_000)
  const page = fixture!.page
  const composer = activeSurface().locator('[contenteditable="true"]').first()
  const initialTranscript = activeSurface().locator('[data-slot="aui_thread-viewport"]')

  await composer.click()
  await composer.type(FIRST_SESSION_MARKER, { delay: 2 })
  await page.keyboard.press('Enter')
  await expect(initialTranscript).toContainText(FIRST_SESSION_MARKER, { timeout: 15_000 })
  await expect(initialTranscript).toContainText('mock inference server', { timeout: 60_000 })

  await composer.click()
  await composer.type(`/new /goal ${GOAL_TEXT}`, { delay: 2 })
  await page.keyboard.press('Enter')

  await expect.poll(() => fixture!.mock.receivedPrompts.some(prompt => prompt.includes(GOAL_TEXT)), {
    timeout: 60_000
  }).toBe(true)
  await expect(page.locator('body')).toContainText(GOAL_TEXT, { timeout: 30_000 })
  const goalSurface = page.locator('[data-composer-target]').filter({ hasText: GOAL_TEXT }).last()
  await expect(goalSurface).not.toContainText(FIRST_SESSION_MARKER)
})
