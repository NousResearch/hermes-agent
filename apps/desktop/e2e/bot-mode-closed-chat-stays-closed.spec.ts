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
// re-opened only by the explicit asks (row menu "Open Bot Chat").
//
// UI note (post design-system rework): the canonical Bot Chat opens INTO the
// main workspace pane (`data-tree-tab="workspace"`), and a lone uncloseable
// workspace pane renders chromeless — its "Bot Chat" tab only exists once a
// second pane (e.g. a ⌘/Ctrl+T thread tile) shares the main zone. Assertions
// about the lone open therefore read the transcript, not a tab.

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

/** The session tabs on the main strip (the Bot Chat workspace tab may sit
 *  beside them). The strip itself auto-hides when the workspace pane is the
 *  only pane in the zone, so an empty result also covers "no strip at all". */
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
 *  workspace path; without it, it would mint the chat into the pane. */
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
  // The seeded forever-chat's first turn — visible only while the Bot Chat
  // transcript is on screen. This is how a chromeless lone open is observed.
  const seededTurn = page.getByText('Hello alpha', { exact: true }).filter({ visible: true })

  // The first click on a bot with nothing open lands on its canonical chat.
  // It fills the lone main workspace pane, which renders without a tab strip.
  await openUntil(
    () => alphaRow.click(),
    () => expect(seededTurn.first()).toBeVisible({ timeout: 45_000 })
  )
  await settle(page, 15_000)
  await snap(page, '01-first-click-opens-bot-chat')

  // Start a fresh thread for Alpha (⌘/Ctrl+T). The thread tile joins the main
  // zone beside the Bot Chat workspace pane, which mounts the tab strip — the
  // "Bot Chat" tab exists now, and the close affordance with it.
  await page.keyboard.press('Control+t')
  await expect(botChatTab.first()).toBeVisible({ timeout: 15_000 })
  await expect.poll(() => mainTabs(page), { timeout: 15_000 }).toHaveLength(1)

  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('hello alpha thread')
  await page.keyboard.press('Enter')
  await expect(page.getByText('hello alpha thread').filter({ visible: true }).first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  await snap(page, '02-new-thread-beside-bot-chat')

  const threadTabs = await mainTabs(page)
  expect(threadTabs).toHaveLength(1)
  const [threadTab] = threadTabs
  expect(threadTab).toMatch(/^session-tile:/)

  // Close the Bot Chat. Its transcript leaves the screen; the thread stays.
  await botChatTab.first().hover()
  await botChatTab.first().getByRole('button', { name: 'Close' }).click({ force: true })
  await expect(botChatTab).toHaveCount(0)
  await expect(seededTurn).toHaveCount(0)
  await snap(page, '03-bot-chat-closed-thread-stays')

  // Switch to Beta: Alpha's thread leaves the strip (scoped away, not closed).
  await betaRow.click()
  await expect(page.locator(`[data-zone-tabstrip="grp-main"] [data-tree-tab="${threadTab}"]`)).toHaveCount(0, {
    timeout: 60_000
  })
  await settle(page)

  // Back to Alpha: the workspace comes back to what the user left, and the
  // closed Bot Chat STAYS closed. The regression this pins re-opened the
  // canonical chat beside the thread on every switch — two panes in the main
  // zone, which mounts the tab strip and puts the "Bot Chat" tab back on
  // screen. Its absence (with the transcript present, so the click landed) is
  // the observable "stays closed".
  await alphaRow.click()
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 30_000 })
  await page.waitForTimeout(3000)
  await expect(botChatTab).toHaveCount(0)
  await snap(page, '04-back-to-alpha-bot-chat-stays-closed')

  // The explicit ask still opens the forever-chat: its seeded first turn is
  // back on screen. (As the surviving main-workspace pane it may render
  // chromeless, so the transcript — not a tab — is the assertion.)
  await openUntil(
    async () => {
      await alphaRow.click({ button: 'right' })
      await page.getByRole('menuitem', { name: 'Open Bot Chat' }).click()
    },
    () => expect(seededTurn.first()).toBeVisible({ timeout: 45_000 })
  )
  await snap(page, '05-explicit-open-bot-chat')
})
