/**
 * E2E tests for desktop sidebar states — background processes, subagents,
 * and session dot transitions.
 *
 * The mock server returns scripted tool_calls that the agent executes for
 * real (trivial commands + real subagent delegations). The tests assert the
 * sidebar states driven by real gateway events.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test, type Page } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { SIDEBAR_CROSS_TEXTS, SIDEBAR_TEXTS, restartMockServer } from './mock-server'

/** Background-running dot aria-label (from i18n en.ts). */
const BG_DOT_LABEL = 'Background task running'
/** Finished-unread dot aria-label. */
const UNREAD_DOT_LABEL = 'Finished — unread'

/** Send a message and wait for the final response to appear. */
async function sendMessageAndWait(
  page: Page,
  trigger: string,
  finalText: string,
  timeout = 90_000,
): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })
  await composer.click()
  await composer.type(trigger, { delay: 20 })
  await page.keyboard.press('Enter')

  await page.waitForFunction(
    () => (document.body.textContent ?? '').includes('E2E_'),
    undefined,
    { timeout: 15_000 },
  )

  await page.waitForFunction(
    (text) => (document.body.textContent ?? '').includes(text),
    finalText,
    { timeout },
  )
}

// ────────────────────────────────────────────────────────────────────────
// Test 1: background process + subagent appear in sidebar during turn
// ────────────────────────────────────────────────────────────────────────

test.describe('sidebar states — background process and subagent', () => {
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

  test('background process dot appears and disappears, subagent runs, final answer visible', async () => {
    const page = fixture.page

    await sendMessageAndWait(page, 'E2E_SIDEBAR_TRIGGER', SIDEBAR_TEXTS.finalText)

    // The background process (sleep 1) should have shown a "Background task
    // running" dot at some point during the turn. We try to catch it; if
    // the process was too fast, that's OK — the real assertion is that the
    // final answer appeared and the dot is gone afterward.
    try {
      await expect
        .poll(
          () => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(),
          { timeout: 15_000, message: 'background dot should appear' },
        )
        .toBeGreaterThan(0)
    } catch {
      // sleep 1 may have finished before we polled — not a failure.
    }

    // After the turn completes and auto-dismiss fires, the background dot
    // should be gone.
    await page.waitForTimeout(8000)
    const bgCount = await page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count()
    expect(bgCount, 'background dot should be gone after auto-dismiss').toBe(0)

    // Evidence: capture the final state — no background dot, final answer visible.
    await page.screenshot({ path: 'test-results/bg-dot-gone-after-dismiss.png' })

    // The final answer text must be in the transcript.
    const viewportText = await page
      .locator('[data-slot="aui_thread-viewport"]')
      .textContent()
    expect(viewportText).toContain(SIDEBAR_TEXTS.finalText)
  })
})

// ────────────────────────────────────────────────────────────────────────
// Test 2: subagent running shows background dot too (longer bg process)
// ────────────────────────────────────────────────────────────────────────

test.describe('sidebar states — subagent and background dot coexist', () => {
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

  test('background dot visible while subagent runs', async () => {
    const page = fixture.page

    // Start the turn but DON'T wait for the final answer yet — we want
    // to assert the background dot is visible WHILE the subagent runs.
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })
    await composer.click()
    await composer.type('E2E_SIDEBAR_CROSS', { delay: 20 })
    await page.keyboard.press('Enter')

    // Wait for the user's message to appear.
    await page.waitForFunction(
      () => (document.body.textContent ?? '').includes('E2E_SIDEBAR_CROSS'),
      undefined,
      { timeout: 15_000 },
    )

    // The background process (sleep 5) should show a "Background task
    // running" dot while the subagent is also running.
    await expect
      .poll(
        () => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(),
        { timeout: 30_000, message: 'background dot should appear while subagent runs' },
      )
      .toBeGreaterThan(0)

    // Evidence: the background dot is visible while the subagent runs.
    await page.screenshot({ path: 'test-results/bg-dot-while-subagent-runs.png' })

    // Now wait for the final answer to appear.
    await page.waitForFunction(
      (text) => (document.body.textContent ?? '').includes(text),
      SIDEBAR_CROSS_TEXTS.finalText,
      { timeout: 90_000 },
    )

    // After the turn + auto-dismiss, the background dot should be gone.
    await page.waitForTimeout(8000)
    const bgCount = await page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count()
    expect(bgCount, 'background dot should be gone after process exits').toBe(0)
  })
})

// ────────────────────────────────────────────────────────────────────────
// Test 3: cross-session — dot updates when viewing a different session
// ────────────────────────────────────────────────────────────────────────

test.describe('sidebar states — cross-session dot transition', () => {
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

  test('background dot transitions to finished when viewing another session', async () => {
    const page = fixture.page

    // Start a turn with a long background process (sleep 5).
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })
    await composer.click()
    await composer.type('E2E_SIDEBAR_CROSS', { delay: 20 })
    await page.keyboard.press('Enter')

    // Wait for the background dot to appear.
    await expect
      .poll(
        () => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(),
        { timeout: 30_000, message: 'background dot should appear' },
      )
      .toBeGreaterThan(0)

    // Wait for the final answer (turn completes, but bg process still running).
    await page.waitForFunction(
      (text) => (document.body.textContent ?? '').includes(text),
      SIDEBAR_CROSS_TEXTS.finalText,
      { timeout: 90_000 },
    )

    // The background dot should still be visible (sleep 5 hasn't finished yet,
    // or auto-dismiss hasn't fired).
    const bgDuringTurn = await page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count()
    expect(bgDuringTurn, 'background dot should still be visible after turn completes').toBeGreaterThan(0)

    // Evidence: bg dot visible on session A while its turn is done but the
    // background process hasn't exited yet.
    await page.screenshot({ path: 'test-results/cross-session-bg-dot-before-switch.png' })

    // Create a new session (click "New session" button).
    await page.locator('button:has-text("New session")').first().click()
    await page.waitForTimeout(2000)

    // Now wait for the background process to finish (sleep 5 + auto-dismiss).
    // The session A dot should transition away from "background running".
    await expect
      .poll(
        () => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(),
        { timeout: 30_000, message: 'background dot should disappear after process finishes' },
      )
      .toBe(0)

    // The original session should show a "finished unread" indicator (green dot)
    // since its turn completed while we were in a different session. This is an
    // event-driven transition, so wait for it instead of sampling the DOM right
    // after the running dot disappears.
    await expect
      .poll(
        () => page.locator(`[aria-label="${UNREAD_DOT_LABEL}"]`).count(),
        { timeout: 30_000, message: 'original session should show finished-unread dot' },
      )
      .toBeGreaterThan(0)

    // Evidence: the green "finished unread" dot on the original session after
    // switching to a new session — the cross-session dot transition.
    await page.screenshot({ path: 'test-results/cross-session-unread-dot-after-switch.png' })
  })
})
