/**
 * E2E: reorder pinned sessions by dragging within the sidebar.
 *
 * Regression for the two-drag-systems conflict (#47728 follow-up): the row body
 * is a session-drag source (drag onto chat → link), and reorder used to live on
 * a separate, near-invisible dnd-kit grab handle that the session-drag stole
 * from. Now one pointer drag routes by DROP LOCATION — released within the
 * pinned list it reorders, released on a chat surface it links. This test drives
 * a REAL pointer drag (mouse.move/down/up in steps, like tile-unread-bug.spec)
 * and asserts the pinned order actually changes.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test, type Page } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { restartMockServer } from './mock-server'

/** A pinned row's durable order is read from the DOM: the tagged reorder rows
 *  in document order carry the session id in data-reorder-row. */
async function pinnedOrder(page: Page): Promise<string[]> {
  return page.locator('[data-reorder-row]').evaluateAll(nodes =>
    nodes.map(n => (n as HTMLElement).dataset.reorderRow ?? '')
  )
}

/** Create a fresh session with a distinct first message so its row is findable. */
async function createSession(page: Page, marker: string): Promise<void> {
  await page.locator('button:has-text("New session")').first().click()
  await page.waitForTimeout(500)

  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })
  await composer.click()
  await composer.type(marker, { delay: 15 })
  await page.keyboard.press('Enter')

  await page.waitForFunction(
    text => (document.body.textContent ?? '').includes(text),
    marker,
    { timeout: 20_000 }
  )
}

/** Shift-click a session row to pin it (the row's onClick pin shortcut). */
async function shiftClickPin(page: Page, marker: string): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: marker }).first()
  await row.click({ modifiers: ['Shift'] })
  await page.waitForTimeout(300)
}

test.describe('sidebar — pinned reorder via pointer drag', () => {
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

  test('dragging a pinned row within the sidebar reorders it', async () => {
    const page = fixture.page

    // Two pinned sessions, distinct markers so their rows are addressable.
    await createSession(page, 'E2E_PIN_ALPHA')
    await shiftClickPin(page, 'E2E_PIN_ALPHA')
    await createSession(page, 'E2E_PIN_BETA')
    await shiftClickPin(page, 'E2E_PIN_BETA')

    // Both rows now carry data-reorder-row (the pinned list is a reorder zone).
    await expect
      .poll(() => pinnedOrder(page).then(o => o.length), { timeout: 10_000 })
      .toBe(2)

    const before = await pinnedOrder(page)
    expect(before).toHaveLength(2)

    const topId = before[0]!
    const topRow = page.locator(`[data-reorder-row="${topId}"]`)
    const box = await topRow.boundingBox()
    expect(box, 'top pinned row must be visible').not.toBeNull()

    // Drag the top row down past the second row's midpoint, in steps so the
    // pointer session engages (threshold) and resolveMove tracks the slot.
    const startX = box!.x + box!.width / 2
    const startY = box!.y + box!.height / 2
    const targetY = box!.y + box!.height * 1.6

    await page.mouse.move(startX, startY)
    await page.mouse.down()
    for (let i = 1; i <= 10; i++) {
      await page.mouse.move(startX, startY + (targetY - startY) * (i / 10))
      await page.waitForTimeout(25)
    }
    await page.mouse.up()
    await page.waitForTimeout(500)

    // The order flipped — the dragged row is no longer first.
    const after = await pinnedOrder(page)
    expect(after, 'pinned order should have the same two ids').toHaveLength(2)
    expect(after[0], 'the dragged top row should now be second').not.toBe(topId)
    expect(after).toContain(topId)
  })
})
