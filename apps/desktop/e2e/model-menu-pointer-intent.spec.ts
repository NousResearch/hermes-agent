import { expect, test } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

let fixture: MockBackendFixture | null = null

test.beforeEach(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a deliberate diagonal path through sibling rows reaches the lowest effort option', async () => {
  const page = fixture!.page
  const modelButton = page.getByRole('button', { name: /Model · mock:/i })

  await modelButton.click()

  const menu = page.locator('[data-slot="dropdown-menu-content"]:visible').last()
  const trigger = menu.getByRole('menuitem', { name: /X Preview F Free/i })
  const triggerBox = await trigger.boundingBox()

  expect(triggerBox).not.toBeNull()
  await page.mouse.move(triggerBox!.x - 8, triggerBox!.y + triggerBox!.height / 2)
  await page.mouse.move(triggerBox!.x + triggerBox!.width / 2, triggerBox!.y + triggerBox!.height / 2)

  const lowestEffort = page.getByRole('menuitemradio', { name: 'Ultra' })
  await expect(lowestEffort).toBeVisible()

  const from = await trigger.boundingBox()
  const to = await lowestEffort.boundingBox()

  expect(from).not.toBeNull()
  expect(to).not.toBeNull()

  for (let step = 1; step <= 30; step += 1) {
    const progress = step / 30
    await page.mouse.move(
      from!.x + from!.width / 2 + (to!.x + to!.width / 2 - (from!.x + from!.width / 2)) * progress,
      from!.y + from!.height / 2 + (to!.y + to!.height / 2 - (from!.y + from!.height / 2)) * progress,
    )
    await page.waitForTimeout(30)
  }

  await expect(trigger).toHaveAttribute('data-state', 'open')
  await lowestEffort.click()
  await expect(trigger).toContainText('Ultra')
})

test('Escape dismisses a submenu after a deliberate diagonal path', async () => {
  const page = fixture!.page
  const modelButton = page.getByRole('button', { name: /Model · mock:/i })

  await modelButton.click()

  const menu = page.locator('[data-slot="dropdown-menu-content"]:visible').last()
  const trigger = menu.getByRole('menuitem', { name: /X Preview F Free/i })
  const triggerBox = await trigger.boundingBox()

  expect(triggerBox).not.toBeNull()
  await page.mouse.move(triggerBox!.x - 8, triggerBox!.y + triggerBox!.height / 2)
  await page.mouse.move(triggerBox!.x + triggerBox!.width / 2, triggerBox!.y + triggerBox!.height / 2)

  const lowestEffort = page.getByRole('menuitemradio', { name: 'Ultra' })
  await expect(lowestEffort).toBeVisible()

  const from = await trigger.boundingBox()
  const to = await lowestEffort.boundingBox()

  expect(from).not.toBeNull()
  expect(to).not.toBeNull()

  for (let step = 1; step <= 30; step += 1) {
    const progress = step / 30
    await page.mouse.move(
      from!.x + from!.width / 2 + (to!.x + to!.width / 2 - (from!.x + from!.width / 2)) * progress,
      from!.y + from!.height / 2 + (to!.y + to!.height / 2 - (from!.y + from!.height / 2)) * progress,
    )
    await page.waitForTimeout(30)
  }

  await expect(lowestEffort).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(lowestEffort).not.toBeVisible()
  await expect(menu).not.toBeVisible()
  await expect(page.getByRole('textbox', { name: 'Message' })).toBeFocused()

  await modelButton.click()
  const reopenedMenu = page.locator('[data-slot="dropdown-menu-content"]:visible').last()
  await expect(reopenedMenu).toBeVisible()
  await expect(lowestEffort).not.toBeVisible()
  await page.keyboard.press('Escape')
  await expect(reopenedMenu).not.toBeVisible()
  await expect(page.getByRole('textbox', { name: 'Message' })).toBeFocused()
})

test('intentional vertical movement switches to the sibling submenu promptly', async () => {
  const page = fixture!.page

  await page.getByRole('button', { name: /Model · mock:/i }).click()

  const menu = page.locator('[data-slot="dropdown-menu-content"]:visible').last()
  const first = menu.getByRole('menuitem', { name: /X Preview F Free/i })
  const sibling = menu.getByRole('menuitem', { name: /Hy3 Free/i })

  await first.focus()
  await first.press('ArrowRight')
  await expect(first).toHaveAttribute('data-state', 'open')

  const from = await first.boundingBox()
  const to = await sibling.boundingBox()

  expect(from).not.toBeNull()
  expect(to).not.toBeNull()

  await page.mouse.move(from!.x + from!.width / 2, from!.y + from!.height / 2)
  await page.mouse.move(to!.x + to!.width / 2, to!.y + to!.height / 2)

  await expect(sibling).toHaveAttribute('data-state', 'open', { timeout: 500 })
})