/**
 * E2E chat tests — send a message and verify a response appears.
 *
 * Requires the full boot chain to complete (hermes serve + mock inference
 * provider). The mock server returns a canned reply, so we verify the
 * response text shows up in the chat transcript.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { test } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('chat interaction with mock backend', () => {
  test('send a message and receive a response', async () => {
    const page = fixture!.page

    // Find the composer — it's a contenteditable textbox.
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })

    // Click to focus, then type the message character by character.
    // Using `type` instead of `fill` because the composer is a
    // contenteditable div with custom keydown handling that tracks
    // IME composition state — `fill` bypasses the event chain.
    await composer.click()
    await composer.type('Hello, can you hear me?', { delay: 20 })

    // Submit with Enter — the composer's keydown handler intercepts
    // plain Enter (without Shift) and calls submitDraft().
    await page.keyboard.press('Enter')

    // Wait for the user's message to appear in the transcript.
    // The message renders as an assistant-ui message in the chat view.
    await page.waitForFunction(
      () => {
        const body = document.body

        if (!body) {
          return false
        }

        return (body.textContent ?? '').includes('Hello, can you hear me?')
      },
      undefined,
      { timeout: 15_000 },
    )

    // Wait for the mock response to appear. The canned reply is:
    // "Hello from the mock inference server! The full boot chain is working."
    // Give it a generous timeout — the inference request goes through the
    // gateway → hermes serve → mock server → streaming SSE back.
    await page.waitForFunction(
      () => {
        const body = document.body

        if (!body) {
          return false
        }

        const text = body.textContent ?? ''

        return text.includes('mock inference server') || text.includes('boot chain is working')
      },
      undefined,
      { timeout: 60_000 },
    )
  })

  test('screenshot of chat with messages', async () => {
    await expectVisualSnapshot(fixture!.page, { name: 'chat-with-messages', app: fixture!.app })
  })
})
