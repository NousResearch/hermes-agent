/**
 * E2E test for PERSISTED unread state — the green "Finished — unread" dot
 * must survive an app restart.
 *
 * Regression coverage for the reset-on-restart bug: the unread flag used to
 * live only in the in-memory `$unreadFinishedSessionIds` atom, so closing and
 * reopening the desktop app grayed out every green dot. The persisted layer
 * (src/store/session-unread.ts) now rebuilds the dot from localStorage-backed
 * finish markers + seen-count watermarks.
 *
 * The scenario uses TWO sessions on purpose: with a single session the app
 * can reopen straight into it after a restart, which acks the session (the
 * user is looking at it) and would mask the dot. Session A finishes in the
 * background while session B is the selected one; the restart reopens into
 * B, so A's dot is observable.
 *
 *  1. Session A: start a turn, hold its stream open.
 *  2. Session B: new session, send a message, let it finish while SELECTED.
 *  3. Release A's stream → its background finish paints A's green dot.
 *  4. QUIT the app, relaunch on the SAME sandbox → A's dot must still be
 *     there (previously it was lost).
 *  5. Open session A → dot clears.
 *  6. Restart once more → the cleared state must persist too (no zombie dot).
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, type Page, test } from '@playwright/test'
import { type ElectronApplication } from '@playwright/test'

import {
  buildAppEnv,
  launchDesktop,
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { restartMockServer } from '../../../tests-js/scripts/mock-server'

/** Finished-unread dot aria-label (from i18n en.ts). */
const UNREAD_DOT_LABEL = 'Finished — unread'

/** Held prompt — the mock pauses this stream until we release it, so the
 *  turn is deterministically still running when we switch away. */
const HELD_PROMPT = 'E2E unread restart: hold this stream until released.'
/** Second session's prompt — completes normally while selected. */
const SECOND_PROMPT = 'E2E unread restart: second session, read while open.'

const unreadDots = (page: Page) => page.locator(`[aria-label="${UNREAD_DOT_LABEL}"]`)

async function sendMessage(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await page.keyboard.press('Enter')
}

test.describe('unread dot survives app restart', () => {
  test.describe.configure({ mode: 'serial' })
  // Three full app boots in one scenario — give it more than the global 90s.
  test.setTimeout(300_000)

  let fixture: MockBackendFixture
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({
      mockServer: { holdFirstStreamForPrompt: HELD_PROMPT },
    })
    app = fixture.app
    page = fixture.page
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    // The fixture's own app handle may already be closed by the restart
    // steps — close whatever is current, then drop the sandbox + mock.
    try {
      await app?.close()
    } catch {
      // already closed
    }

    fixture?.mock.close()
    fixture?.sandbox.cleanup()
  })

  /** Relaunch the desktop app against the SAME sandbox (same userData →
   *  same localStorage, same HERMES_HOME → same session store). */
  async function restartApp(): Promise<void> {
    await app.close()

    const relaunched = await launchDesktop(buildAppEnv(fixture.sandbox))
    app = relaunched.app
    page = relaunched.page
    await waitForAppReady({ ...fixture, app, page }, 120_000)
  }

  test('green dot persists across restart and its clear persists too', async () => {
    // ── 1. Session A: start a turn whose stream the mock holds open ────
    await sendMessage(page, HELD_PROMPT)
    await fixture.mock.waitForHeldStream()

    // ── 2. Session B: complete a turn while SELECTED (stays read) ──────
    await page.locator('button:has-text("New session")').first().click()
    await sendMessage(page, SECOND_PROMPT)

    // B's reply lands in the open transcript — this session is "read".
    await page.waitForFunction(
      () => (document.body.textContent ?? '').includes('Hello from the mock inference server'),
      undefined,
      { timeout: 30_000 },
    )

    // ── 3. Release A's stream → background finish paints A's dot ──────
    fixture.mock.releaseHeldStream()

    await expect
      .poll(() => unreadDots(page).count(), {
        timeout: 30_000,
        message: 'green unread dot should appear after the background finish',
      })
      .toBeGreaterThan(0)

    // ── 4. Restart the app — the dot must survive ──────────────────────
    // The app reopens into session B (or a fresh draft), NOT session A, so
    // A's dot is observable rather than being acked by the route restore.
    await restartApp()

    await expect
      .poll(() => unreadDots(page).count(), {
        timeout: 60_000,
        message: 'green unread dot should be rebuilt from persisted state after restart',
      })
      .toBeGreaterThan(0)

    // ── 5. Open session A — the dot clears ─────────────────────────────
    // The dot sits inside A's sidebar row button; click that row.
    await unreadDots(page)
      .first()
      .locator('xpath=ancestor::button[1]')
      .click()

    await expect
      .poll(() => unreadDots(page).count(), {
        timeout: 15_000,
        message: 'opening the session should clear its unread dot',
      })
      .toBe(0)

    // ── 6. Restart again — the CLEARED state must persist as well ──────
    await restartApp()

    // Give the sidebar a moment to load rows, then assert no dot returns.
    await page.waitForSelector('[data-slot="sidebar"]', { timeout: 60_000 })
    await page.waitForTimeout(5_000)
    expect(await unreadDots(page).count(), 'acked session must stay read after restart').toBe(0)
  })
})
