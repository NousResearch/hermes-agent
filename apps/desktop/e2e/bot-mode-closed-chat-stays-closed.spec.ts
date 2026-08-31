import fs from 'node:fs'
import path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { MOCK_REPLY, startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// A bot row click is "go to this bot", not "open its Bot Chat". Before the
// fix, every click resolved the canonical chat by name and opened it as a tab
// again — a Bot Chat the user had closed came back beside every newer thread
// on every bot switch, because nothing records a close (the plugin keeps no
// closed set; core's tile bucket only forgets). Now a bot whose workspace
// already holds tabs comes back to the one the user left; the forever-chat is
// re-opened only by the explicit asks (row menu "Open Bot Chat", Bots home
// "Open chat").

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

/** A bot's backend spawns on its first open; give the wake a real chance to
 *  clear before the next gesture races it. Tolerant: the mock backend can
 *  keep a tile's "Waking up…" notice around. */
async function settle(page: Page, timeout = 90_000): Promise<void> {
  await page
    .getByText(/Waking up/i)
    .first()
    .waitFor({ state: 'hidden', timeout })
    .catch(() => undefined)
  await page.waitForTimeout(500)
}

/** A first open right after a bot's backend spawned can strand on the
 *  profile socket (a separate, pre-existing reconnect race); a newer click
 *  supersedes it. Retry the gesture like a user would before giving up. */
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

const SCREENSHOT_DIR = process.env.BOT_MODE_SCREENSHOT_DIR

async function snap(page: Page, name: string): Promise<void> {
  if (SCREENSHOT_DIR) {
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${name}.png` })
  }
}

/** The session tabs on the main strip (the Bots home tab may sit beside them). */
const mainTabs = (page: Page) =>
  page.evaluate(() =>
    [...document.querySelectorAll<HTMLElement>('[data-zone-tabstrip="grp-main"] [data-tree-tab]')]
      .map(element => element.getAttribute('data-tree-tab') ?? '')
      .filter(id => id.startsWith('session-tile:'))
  )

/** Bots are profiles. Seeding one on disk before launch — with the mock
 *  provider so its own backend can answer, and a real, durable "Bot Chat"
 *  row (the plugin's canonical forever-chat, found by exact title) — keeps
 *  in-app creation and the intro turn it fires out of a scenario that is
 *  about the row click. With the row present, the click takes the open-as-
 *  tab path; without it, it would mint the chat into the workspace pane. */
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

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  await seedBot(sandbox.hermesHome, mock.url, 'alpha')
  await seedBot(sandbox.hermesHome, mock.url, 'beta')

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

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a bot row click returns to the open thread and does not re-open a closed Bot Chat', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)

  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  const betaRow = page.getByRole('button', { name: /^beta\b/i }).filter({ visible: true }).first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await expect(betaRow).toBeVisible({ timeout: 30_000 })
  const botChatTab = page.getByRole('tab', { name: /Bot Chat/ }).filter({ visible: true })

  // The first click on a bot with nothing open lands on its canonical chat.
  await openUntil(
    () => alphaRow.click(),
    () => expect(botChatTab.first()).toBeVisible({ timeout: 45_000 })
  )
  await settle(page, 15_000)
  await snap(page, '01-first-click-opens-bot-chat')

  // Close it, then start a fresh thread for Alpha (⌘/Ctrl+T — the strip's
  // "+" leaves with the zone's last tab).
  await botChatTab.first().hover()
  await botChatTab.first().getByRole('button', { name: 'Close' }).click({ force: true })
  await expect(botChatTab).toHaveCount(0)

  await page.keyboard.press('Control+t')
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('hello alpha thread')
  await page.keyboard.press('Enter')
  await expect(page.getByText('hello alpha thread').filter({ visible: true }).first()).toBeVisible({ timeout: 15_000 })
  // The reply also becomes the tab's (clipped) title — match the visible copy.
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  await snap(page, '02-closed-bot-chat-new-thread')

  const threadTabs = await mainTabs(page)
  expect(threadTabs).toHaveLength(1)
  const [threadTab] = threadTabs
  expect(threadTab).toMatch(/^session-tile:/)

  // Switch to Beta: Alpha's thread leaves the strip (scoped away, not closed).
  await betaRow.click()
  await expect(page.locator(`[data-zone-tabstrip="grp-main"] [data-tree-tab="${threadTab}"]`)).toHaveCount(0, {
    timeout: 60_000
  })
  await settle(page)

  // Back to Alpha: the thread is fronted, and the closed Bot Chat STAYS closed.
  await alphaRow.click()
  const threadTabLocator = page.locator(`[data-zone-tabstrip="grp-main"] [data-tree-tab="${threadTab}"]`)
  await expect(threadTabLocator).toBeVisible({ timeout: 30_000 })
  await expect(threadTabLocator).toHaveAttribute('aria-selected', 'true')
  await page.waitForTimeout(3000)
  await expect(botChatTab).toHaveCount(0)
  expect(await mainTabs(page)).toEqual([threadTab])
  await snap(page, '03-back-to-alpha-bot-chat-stays-closed')

  // The explicit ask still opens the forever-chat, beside the thread.
  await openUntil(
    async () => {
      await alphaRow.click({ button: 'right' })
      await page.getByRole('menuitem', { name: 'Open Bot Chat' }).click()
    },
    () => expect(botChatTab.first()).toBeVisible({ timeout: 45_000 })
  )
  expect(await mainTabs(page)).toHaveLength(2)
  expect(await mainTabs(page)).toContain(threadTab)
  await snap(page, '04-explicit-open-bot-chat')
})
