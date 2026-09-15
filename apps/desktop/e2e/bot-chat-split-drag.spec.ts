import fs from 'node:fs'
import path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
  type MockBackendFixture
} from './fixtures'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Regression for upstream #109684 (Desktop Bot Mode):
//
//   1. a bot chat is an ordinary tab in the main strip, so dragging its tab
//      onto the workspace edge must produce a usable, visible split pane --
//      never a hole in the pane tree, never a chat that silently vanishes;
//   2. closing that tab and reopening the bot from the BOTS roster must bring
//      the same chat back.
//
// A `+` side thread is opened first (Control+t, same trick as
// bot-mode-tab-shows-bot-name.spec.ts) so the main zone has a tab strip: with a
// single lone pane the strip is hidden and there is nothing to drag.

type Page = MockBackendFixture['page']

type ZoneDump = {
  group: string
  rect: [number, number, number, number]
  tabs: { id: string; label: string }[]
  hasStrip: boolean
  stripId: string | null
  hasContent: boolean
}

let fixture: MockBackendFixture | null = null

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible({ timeout: 60_000 })
}

async function openUntil(action: () => Promise<void>, expected: () => Promise<void>, attempts = 3): Promise<void> {
  for (let attempt = 1; ; attempt += 1) {
    await action()

    try {
      await expected()

      return
    } catch (error) {
      if (attempt >= attempts) {
        throw error
      }
    }
  }
}

async function seedBot(hermesHome: string, mockUrl: string, name: string): Promise<void> {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  const builder = await RealSessionBuilder.start(dir)

  try {
    await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${name}`] })
  } finally {
    await builder.close()
  }
}

const ZONES = `(() => {
  return [...document.querySelectorAll('[data-tree-group]')].map(el => {
    const r = el.getBoundingClientRect()
    return {
      group: el.getAttribute('data-tree-group'),
      rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
      tabs: [...el.querySelectorAll('[data-tree-tab]')].map(t => ({
        id: t.getAttribute('data-tree-tab'),
        label: (t.textContent || '').trim().slice(0, 30)
      })),
      hasStrip: Boolean(el.querySelector('[data-zone-tabstrip]')),
      stripId: el.querySelector('[data-zone-tabstrip]')?.getAttribute('data-zone-tabstrip') ?? null,
      hasContent: Boolean(el.querySelector('textarea, [contenteditable="true"], [data-pane-self-label]'))
    }
  })
})()`

const zones = (page: Page) => page.evaluate(ZONES) as Promise<ZoneDump[]>

/** A zone with real estate but nothing in it: the pane tree grew a hole. */
const deadZones = (zs: ZoneDump[]) =>
  zs.filter(z => z.rect[2] > 40 && z.rect[3] > 40 && z.tabs.length === 0 && !z.hasStrip && !z.hasContent)

const liveZones = (zs: ZoneDump[]) => zs.filter(z => z.rect[2] > 40 && z.rect[3] > 40)

const visibleTabs = (page: Page) => page.locator('[data-zone-tabstrip] [data-tree-tab]').filter({ visible: true })

const chatVisible = (page: Page, text: string) =>
  page.getByText(text, { exact: true }).filter({ visible: true }).first()

/** Open the bot's chat from the BOTS roster. */
async function openBotChatFromRoster(page: Page, name: string): Promise<void> {
  await openBots(page)

  const row = page.getByRole('button', { name: new RegExp(`^${name}\\b`, 'i') }).filter({ visible: true }).first()
  await expect(row).toBeVisible({ timeout: 30_000 })

  await openUntil(
    () => row.click(),
    () => expect(chatVisible(page, `Hello ${name}`)).toBeVisible({ timeout: 45_000 })
  )
}

/** Same, plus a `+` side thread: with a single lone pane the strip is hidden
 *  and there is nothing to drag. */
async function openBotChatWithStrip(page: Page, name: string): Promise<void> {
  await openBotChatFromRoster(page, name)

  await openUntil(
    () => page.keyboard.press('Control+t'),
    async () => {
      await expect.poll(() => visibleTabs(page).count(), { timeout: 15_000 }).toBeGreaterThanOrEqual(2)
    }
  )
}

/** Slow, stepped pointer drag: the pane drag tracks pointermove. */
async function dragTo(page: Page, from: { x: number; y: number }, to: { x: number; y: number }): Promise<void> {
  await page.mouse.move(from.x, from.y)
  await page.mouse.down()
  const steps = 16
  for (let i = 1; i <= steps; i += 1) {
    await page.mouse.move(from.x + ((to.x - from.x) * i) / steps, from.y + ((to.y - from.y) * i) / steps)
    await page.waitForTimeout(40)
  }
  await page.waitForTimeout(400)
  await page.mouse.up()
  await page.waitForTimeout(1500)
}

test.beforeEach(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bot-chat-split')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  await seedBot(sandbox.hermesHome, mock.url, 'alpha')

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  fixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('dragging a bot chat tab to the workspace edge yields a usable split pane', async () => {
  test.setTimeout(600_000)
  const page = fixture!.page

  await openBotChatWithStrip(page, 'alpha')
  expect(deadZones(await zones(page))).toEqual([])

  const tab = visibleTabs(page).filter({ hasText: /alpha/i }).first()
  await expect(tab).toBeVisible({ timeout: 30_000 })
  const box = (await tab.boundingBox())!
  const win = await page.evaluate(() => ({ w: window.innerWidth, h: window.innerHeight }))

  await dragTo(
    page,
    { x: box.x + box.width / 2, y: box.y + box.height / 2 },
    { x: win.w - 16, y: Math.round(win.h / 3) }
  )

  console.log('AFTER DRAG:', JSON.stringify(await zones(page), null, 1))

  // Split view means BOTH surfaces on screen at the same time: the bot chat in
  // its own pane, the side thread still in the other.
  await expect(chatVisible(page, 'Hello alpha')).toBeVisible({ timeout: 30_000 })

  const after = await zones(page)
  // No hole where the pane was dropped...
  expect(deadZones(after)).toEqual([])
  // ...and the drop really split the workspace in two.
  expect(liveZones(after).length).toBeGreaterThanOrEqual(2)
  await page.screenshot({ path: '/tmp/e2e-bot-drag-split.png' })
})

test('closing the bot tab and reopening it from the BOTS roster brings the chat back', async () => {
  test.setTimeout(600_000)
  const page = fixture!.page

  await openBotChatWithStrip(page, 'alpha')

  // Close the bot chat's own tab...
  const tab = visibleTabs(page).filter({ hasText: /alpha/i }).first()
  await expect(tab).toBeVisible({ timeout: 30_000 })
  await tab.locator('button[aria-label*="lose" i]').first().click({ force: true })
  await page.waitForTimeout(1500)

  // ...reopening it from the roster must bring the very same chat back, and
  // leave no hole behind in the pane tree.
  await openBotChatFromRoster(page, 'alpha')
  expect(deadZones(await zones(page))).toEqual([])
})
