/**
 * A session's running highlight must end when its background work ends, even
 * while the user is looking at another session.
 *
 * The background process here is spawned WITHOUT `notify_on_complete`, so its
 * exit sends no event: the only things that can clear session A's row are the
 * app-wide live-work roster read and the per-pane process poll, and the pane
 * poll stops once A's pane is hidden. Before the fix the row kept its
 * "Background task running" state until the user clicked back into A.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from '@playwright/test'

import {
  createBackgroundReleaseHandle,
  restartMockServer,
  SIDEBAR_CROSS_TEXTS
} from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

const BG_DOT_LABEL = 'Background task running'
const SESSION_RUNNING_DOT_LABEL = 'Session running'
// Script indices must line up with real chat turns (see sidebar-states.spec.ts).
const DISABLE_AUTO_TITLE = 'auxiliary:\n  title_generation:\n    enabled: false'

test.describe('sidebar — background work finishing off screen', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture
  const bgRelease = createBackgroundReleaseHandle()

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({
      extraConfig: DISABLE_AUTO_TITLE,
      mockServer: { backgroundReleasePath: bgRelease.path, backgroundSilent: true }
    })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    bgRelease.release()
    await fixture?.cleanup()
    bgRelease.cleanup()
  })

  test('a silent background process that exits while another session is open clears the row', async () => {
    const page = fixture.page

    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })
    await composer.click()
    await composer.type('E2E_SIDEBAR_CROSS', { delay: 20 })
    await page.keyboard.press('Enter')

    await page.waitForFunction(
      text => (document.body.textContent ?? '').includes(text),
      SIDEBAR_CROSS_TEXTS.finalText,
      { timeout: 90_000 }
    )
    await expect
      .poll(() => page.locator(`[aria-label="${SESSION_RUNNING_DOT_LABEL}"]`).count(), { timeout: 30_000 })
      .toBe(0)
    await expect
      .poll(() => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(), {
        timeout: 30_000,
        message: 'background dot should be visible after the turn completes'
      })
      .toBeGreaterThan(0)

    await page.locator('button:has-text("New session")').first().click()
    await page.waitForTimeout(2000)
    await page.screenshot({ path: 'test-results/offscreen-bg-before-release.png' })

    bgRelease.release()

    // The roster is read every 10 s; two reads is ample. The screenshot is the
    // evidence either way, so it is taken even when the row stays lit.
    try {
      await expect
        .poll(() => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(), {
          timeout: 35_000,
          message: 'background dot should clear without clicking back into the session'
        })
        .toBe(0)
    } finally {
      await page.screenshot({ path: 'test-results/offscreen-bg-after-release.png' })
    }
  })
})
