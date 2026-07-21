/**
 * E2E tests asserting the mock backend gets the app past the setup/onboarding
 * screen.
 *
 * The mock backend fixture writes a config.yaml with a pre-configured mock
 * provider pointing at a mock inference server. When the app boots, the
 * runtime readiness check should detect the working provider and dismiss the
 * onboarding overlay — landing straight on the chat UI without ever showing
 * the "Let's get you setup with Hermes Agent" screen.
 *
 * If these tests fail, the mock backend config isn't getting the app past
 * onboarding — the chat interaction tests (chat.spec.ts) will also fail
 * because the composer is blocked by the setup overlay.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from '@playwright/test'

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

test.describe('mock backend gets past setup screen', () => {
  test('onboarding overlay is not shown', async () => {
    const page = fixture!.page

    // The onboarding overlay renders "Let's get you setup with Hermes Agent"
    // when the runtime check fails to find a working provider. With the mock
    // backend configured, the runtime check should pass and the overlay
    // returns null — this text should NOT be present in the DOM.
    await page.waitForFunction(
      () => {
        const text = document.body.textContent ?? ''

        return !text.includes("Let's get you setup")
      },
      undefined,
      { timeout: 30_000 },
    )
  })

  test('chat composer is visible', async () => {
    const page = fixture!.page

    // The composer (contenteditable div) should be visible and not blocked
    // by the onboarding overlay. If the first test passed, the overlay is
    // gone and the composer is the primary interactive surface.
    const composer = page.locator('[contenteditable="true"]').first()
    await expect(composer).toBeVisible()
  })

  test('can type into the composer', async () => {
    const page = fixture!.page

    // If the setup overlay is truly gone, the composer accepts input.
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.click()
    await composer.type('hello mock backend', { delay: 20 })

    // Verify the typed text appears in the DOM.
    await page.waitForFunction(
      () => (document.body.textContent ?? '').includes('hello mock backend'),
      undefined,
      { timeout: 10_000 },
    )
  })

  test('screenshot shows chat UI without setup screen', async () => {
    await expectVisualSnapshot(fixture!.page, { name: 'mock-backend-chat-ready', app: fixture!.app })
  })
})
