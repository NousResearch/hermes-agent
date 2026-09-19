import type { Locator } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { MOCK_REPLY } from './mock-server'
import { expect, type Page, test } from './test'

let fixture: MockBackendFixture | null = null

async function send(page: Page, text: string): Promise<void> {
  const composer = page.locator('[data-slot="composer-rich-input"]:visible').last()
  const surface = composer.locator('xpath=ancestor::*[@data-session-anchor][1]')

  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.fill(text)
  await page.keyboard.press('Enter')
  await expect(surface).toContainText(text, { timeout: 30_000 })
  await expect(surface).toContainText(MOCK_REPLY, { timeout: 60_000 })
}

async function drag(page: Page, source: Locator, targetX: number, targetY: number) {
  const box = await source.boundingBox()
  expect(box).not.toBeNull()

  const startX = box!.x + box!.width / 2
  const startY = box!.y + box!.height / 2

  await page.mouse.move(startX, startY)
  await page.mouse.down()

  for (let step = 1; step <= 12; step++) {
    await page.mouse.move(startX + (targetX - startX) * (step / 12), startY + (targetY - startY) * (step / 12))
    await page.waitForTimeout(20)
  }

  await page.mouse.up()
}

async function groupIdFor(page: Page, paneId: string): Promise<string | null> {
  return page.locator(`[data-tree-tab="${paneId}"]`).evaluate(tab => {
    return tab.closest<HTMLElement>('[data-tree-group]')?.dataset.treeGroup ?? null
  })
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)

  // Keep the primary workspace's tab visible after it becomes a lone pane in
  // the lower split, matching the reporter's always-visible tab bars.
  await fixture.page.evaluate(() => localStorage.setItem('hermes.desktop.tabStripDefault', 'always'))
  await fixture.page.reload()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a lower primary chat merges into an upper session strip', async () => {
  const page = fixture!.page
  const mainPrompt = 'E2E lower primary chat A'
  const tilePrompt = 'E2E upper session tile B'

  await send(page, mainPrompt)

  // New-session intent opens a session TILE in the same strip, preserving the
  // loaded primary workspace behind it. Give that tile a real session too.
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
  await send(page, tilePrompt)

  const tileTab = page.locator('[data-tree-tab^="session-tile:"]').first()
  const workspaceTab = page.locator('[data-tree-tab="workspace"]')
  await expect(tileTab).toBeVisible({ timeout: 30_000 })
  await expect(workspaceTab).toBeVisible({ timeout: 30_000 })

  const tilePaneId = await tileTab.getAttribute('data-tree-tab')
  expect(tilePaneId).not.toBeNull()

  // Tear B into a top split, leaving the primary workspace chat A below it.
  const sharedGroup = tileTab.locator('xpath=ancestor::*[@data-tree-group][1]')
  const sharedBox = await sharedGroup.boundingBox()
  expect(sharedBox).not.toBeNull()
  await drag(page, tileTab, sharedBox!.x + sharedBox!.width / 2, sharedBox!.y + 54)

  await expect
    .poll(() => groupIdFor(page, 'workspace'), { timeout: 15_000 })
    .not.toBe(await groupIdFor(page, tilePaneId!))
  expect(await groupIdFor(page, 'workspace')).not.toBe(await groupIdFor(page, tilePaneId!))

  // This is the user's failing gesture: drag the bottom primary chat onto the
  // upper chat's tab strip. It must move `workspace` itself, not attempt to
  // open the already-selected stored session as a duplicate tile.
  const targetStrip = tileTab.locator('xpath=ancestor::*[@data-zone-tabstrip][1]')
  const stripBox = await targetStrip.boundingBox()
  expect(stripBox).not.toBeNull()
  await drag(page, workspaceTab, stripBox!.x + stripBox!.width - 24, stripBox!.y + stripBox!.height / 2)

  await expect.poll(() => groupIdFor(page, 'workspace'), { timeout: 15_000 }).toBe(await groupIdFor(page, tilePaneId!))
  await expect(page.locator('[data-tree-tab="workspace"]')).toHaveAttribute('aria-selected', 'true')
  await expect(page.locator('[data-session-anchor="workspace"]:visible')).toContainText(mainPrompt)
})
