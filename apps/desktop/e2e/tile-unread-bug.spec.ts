/**
 * E2E tests for the tile-unread bug — two scenarios:
 *
 * 1. TAB (stacked, not visible) — a session opened as a tab via ⌃-click is
 *    NOT visible on screen. When it finishes, the green "unread" dot IS
 *    correct — the user isn't looking at it. This test PASSES.
 *
 * 2. SPLIT (side-by-side, visible) — a session dragged to the edge of the
 *    workspace zone opens as a split tile, visible on screen at the same time
 *    as the main session. When it finishes, it should NOT get the green
 *    "unread" dot — the user is looking right at it. This test FAILS until
 *    the fix in session-states.ts:174 lands (the unread check only compares
 *    against $selectedStoredSessionId and ignores $sessionTiles).
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { SIDEBAR_CROSS_TEXTS, restartMockServer } from './mock-server'

/** Finished-unread dot aria-label. */
const UNREAD_DOT_LABEL = 'Finished — unread'
/** Background-running dot aria-label. */
const BG_DOT_LABEL = 'Background task running'

/** Locate a session's sidebar row by its preview text. */
function sessionRow(page: import('@playwright/test').Page, text: string) {
  return page.locator('[data-slot="sidebar"] button').filter({ hasText: text }).first()
}

/** Common setup: start a turn with a sleep 5 bg process + subagent, wait for
 *  the turn to complete, then switch to a new session so the first session is
 *  no longer $selectedStoredSessionId (required before opening a tile). */
async function startTurnAndSwitchAway(page: import('@playwright/test').Page) {
  // Send E2E_SIDEBAR_CROSS — starts a turn with sleep 5 + subagent.
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

  // Wait for the background dot — confirms the turn is running.
  await expect
    .poll(
      () => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(),
      { timeout: 30_000, message: 'background dot should appear' },
    )
    .toBeGreaterThan(0)

  // Wait for the turn to complete (final answer visible).
  await page.waitForFunction(
    (text) => (document.body.textContent ?? '').includes(text),
    SIDEBAR_CROSS_TEXTS.finalText,
    { timeout: 90_000 },
  )

  // The background dot should still be visible (sleep 5 hasn't finished).
  const bgDuringTurn = await page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count()
  expect(bgDuringTurn, 'background dot should still be visible after turn completes').toBeGreaterThan(0)

  // Switch to a new session — session A is no longer $selectedStoredSessionId.
  // This is required: openSessionTile bails if the session is already selected.
  await page.locator('button:has-text("New session")').first().click()
  await page.waitForTimeout(2000)
}

/** Wait for the background process to finish (sleep 5 + auto-dismiss). */
async function waitForBgProcessToFinish(page: import('@playwright/test').Page) {
  await expect
    .poll(
      () => page.locator(`[aria-label="${BG_DOT_LABEL}"]`).count(),
      { timeout: 30_000, message: 'background dot should disappear after process finishes' },
    )
    .toBe(0)
}

// ────────────────────────────────────────────────────────────────────────
// Test 1: TAB (not visible) — unread dot IS correct (PASSES)
// ────────────────────────────────────────────────────────────────────────

test.describe('sidebar states — tab (hidden) unread is correct', () => {
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

  test('session opened as a tab (not visible) correctly gets unread dot', async () => {
    const page = fixture.page

    await startTurnAndSwitchAway(page)

    // Evidence: session A is in the background (bg dot in sidebar).
    await page.screenshot({ path: 'test-results/tile-bug-tab-switched-away.png' })

    // ⌃-click opens the session as a TAB (center dock = stacked, not visible
    // unless it's the active tab). The session is NOT on screen.
    const row = sessionRow(page, SIDEBAR_CROSS_TEXTS.finalText)
    await row.click({ modifiers: ['Control'] })
    await page.waitForTimeout(2000)

    // Evidence: the tab is open but the session is not visible on screen.
    await page.screenshot({ path: 'test-results/tile-bug-tab-opened.png' })

    await waitForBgProcessToFinish(page)

    // A tab that's not the active tab IS hidden — the unread dot is correct.
    // The user is NOT looking at it, so marking it "unread" is right.
    const unreadCount = await page.locator(`[aria-label="${UNREAD_DOT_LABEL}"]`).count()
    expect(unreadCount, 'hidden tab should be marked unread').toBeGreaterThan(0)

    await page.screenshot({ path: 'test-results/tile-bug-tab-unread-correct.png' })
  })
})

// ────────────────────────────────────────────────────────────────────────
// Test 2: SPLIT (visible) — unread dot is WRONG (FAILS until fix)
// ────────────────────────────────────────────────────────────────────────

test.describe.skip('sidebar states — split (visible) unread bug (RED)', () => {
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

  test('session visible in a split tile does NOT get unread dot when it finishes', async () => {
    const page = fixture.page

    await startTurnAndSwitchAway(page)

    // Evidence: session A is in the background (bg dot in sidebar).
    await page.screenshot({ path: 'test-results/tile-bug-split-switched-away.png' })

    // Drag the session row from the sidebar to the right edge of the workspace
    // zone to create a SPLIT (side-by-side) tile. This triggers the real
    // startSessionDrag → onCommit → openSessionTile(id, 'right', anchor) path.
    const row = sessionRow(page, SIDEBAR_CROSS_TEXTS.finalText)
    const rowBox = await row.boundingBox()
    expect(rowBox, 'session row must be visible').not.toBeNull()

    // Find the workspace zone — the main chat area. We drop on its right edge.
    const workspace = page.locator('[data-session-anchor="workspace"]')
    const wsBox = await workspace.boundingBox()
    expect(wsBox, 'workspace zone must be visible').not.toBeNull()

    // Drag from the session row to the right edge of the workspace.
    // The drag-session's subZonePosition resolves a right-edge drop as 'right'
    // (a split dock), not 'center' (which would be a composer link).
    await page.mouse.move(rowBox!.x + rowBox!.width / 2, rowBox!.y + rowBox!.height / 2)
    await page.mouse.down()
    // Move in steps so the drag-session's pointermove handler tracks the
    // position and resolves the drop zone (a single jump can miss the
    // threshold/engage logic).
    const targetX = wsBox!.x + wsBox!.width - 20
    const targetY = wsBox!.y + wsBox!.height / 2
    const steps = 10
    for (let i = 1; i <= steps; i++) {
      const x = rowBox!.x + rowBox!.width / 2 + (targetX - (rowBox!.x + rowBox!.width / 2)) * (i / steps)
      const y = rowBox!.y + rowBox!.height / 2 + (targetY - (rowBox!.y + rowBox!.height / 2)) * (i / steps)
      await page.mouse.move(x, y)
      await page.waitForTimeout(30)
    }
    await page.mouse.up()
    await page.waitForTimeout(2000)

    // Evidence: the split tile is now open side-by-side — both sessions visible.
    await page.screenshot({ path: 'test-results/tile-bug-split-opened.png' })

    await waitForBgProcessToFinish(page)

    // THE BUG: the session visible in the split tile should NOT have the green
    // "finished unread" dot — the user is looking right at it. This assertion
    // FAILS until the fix in session-states.ts:174 lands.
    const unreadCount = await page.locator(`[aria-label="${UNREAD_DOT_LABEL}"]`).count()
    expect(unreadCount, 'session visible in a split tile should NOT be marked unread').toBe(0)

    // Evidence: the green dot should NOT be here — this screenshot shows the bug.
    await page.screenshot({ path: 'test-results/tile-bug-split-unread-should-not-exist.png' })
  })
})
