/**
 * Run Board acceptance coverage.
 *
 * Reproduces the original failure mode with real streamed assistant output and
 * repeated todo tool calls. The last plan must remain in its independent pane
 * after the final message.complete event.
 */

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('persists independently after streamed messages and turn completion', async () => {
  const page = fixture!.page
  const board = page.getByTestId('run-board')

  await expect(board).toBeVisible()
  await expect(board).toContainText('No task plan yet')

  const composer = page.locator('[contenteditable="true"]').first()

  await composer.click()
  await composer.type('E2E_INTERIM_TRIGGER')
  await page.keyboard.press('Enter')

  await page
    .getByRole('paragraph')
    .filter({ hasText: 'All done! Here is the complete summary of what I found.' })
    .last()
    .waitFor({ state: 'visible', timeout: 60_000 })

  // The final scripted todo update is completed before message.complete. The
  // board must still display it after the final assistant message has rendered.
  await expect(board).toBeVisible()
  await expect(board).toContainText('DONE')
  await expect(board).toContainText('1 of 1 resolved')
  await expect(board).toContainText('Support needed: No')
  await expect(board).toContainText('Note finding')

  const bounds = await board.boundingBox()

  expect(bounds).not.toBeNull()
  expect(bounds!.width).toBeGreaterThan(180)
  expect(bounds!.height).toBeGreaterThan(120)

  await expectVisualSnapshot(page, { app: fixture!.app, name: 'run-board-persistent' })
})
