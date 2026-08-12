/**
 * E2E coverage for session compression, which rotates a live backend session.
 */

import { expect, test, type Page } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { receivedUserTexts, restartMockServer } from './mock-server'

async function send(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.click()
  await composer.type(text, { delay: 15 })
  await page.keyboard.press('Enter')
}

async function waitForTranscript(page: Page, text: string, timeout = 90_000): Promise<void> {
  await page.waitForFunction(
    expected => document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes(expected) ?? false,
    text,
    { timeout },
  )
}

test.describe('session compression', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('compresses an existing session and accepts a follow-up turn on its continuation', async () => {
    const { page } = fixture
    const reply = 'Hello from the mock inference server! The full boot chain is working.'

    // Three completed exchanges leave a compressible middle after the
    // compressor's protected head/tail boundaries.
    await send(page, 'E2E_COMPRESSION_FIRST')
    await waitForTranscript(page, reply)
    await send(page, 'E2E_COMPRESSION_SECOND')
    await expect.poll(() => receivedUserTexts().filter(text => text === 'E2E_COMPRESSION_SECOND').length).toBe(1)
    await send(page, 'E2E_COMPRESSION_THIRD')
    await expect.poll(() => receivedUserTexts().filter(text => text === 'E2E_COMPRESSION_THIRD').length).toBe(1)

    // Commit the command before typing its argument. This waits for the async
    // completion request on cold CI workers, then uses the composer's own
    // keyboard accept path to replace the `/compress` trigger with a command
    // chip. Clicking a later completion after typing the argument can insert a
    // second command token (for example `//compress ...`) as plain text.
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.click()
    await composer.type('/compress', { delay: 15 })
    await page.getByText('/compress').first().waitFor({ state: 'visible' })
    await page.keyboard.press('Enter')
    await composer.type(' preserve the three test turns', { delay: 15 })
    await page.keyboard.press('Enter')
    await expect
      .poll(
        () => page.locator('[data-slot="aui_thread-viewport"]').textContent(),
        { timeout: 90_000 },
      )
      .toMatch(/Compressed|No changes from compression/)

    // Compression rotates the agent's live session id. A post-compression
    // ordinary turn proves the desktop's runtime binding followed that child.
    await send(page, 'E2E_COMPRESSION_FOLLOW_UP')
    await expect.poll(() => receivedUserTexts().filter(text => text === 'E2E_COMPRESSION_FOLLOW_UP').length).toBe(1)
    await waitForTranscript(page, reply)
    await page.screenshot({ path: 'test-results/session-compression-continuation.png' })
  })
})
