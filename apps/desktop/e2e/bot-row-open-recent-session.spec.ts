import fs from 'node:fs'
import path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Right-click a bot row → "Open recent session" opens the bot's most recently
// active ordinary session as a tab in its workspace, while the row's left click
// keeps landing on the canonical Bot Chat (hermes-agent#93054).

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

// BOT_RECENT_SESSION_SCREENSHOT_DIR=<dir> saves full-window captures at the
// key states — PR evidence; never part of the assertions.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.BOT_RECENT_SESSION_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })
}

const RECENT_TITLE = 'Deploy notes'

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-recent-session')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const dir = path.join(sandbox.hermesHome, 'profiles', 'alpha')
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mock.url)
  writeEnvFile(dir)
  const builder = await RealSessionBuilder.start(dir)

  try {
    // The canonical chat first, then an ordinary session so the latter is the
    // profile's newest listed conversation.
    await builder.createSession({ title: 'Bot Chat', turns: ['Hello alpha'] })
    await builder.createSession({ title: RECENT_TITLE, turns: ['Where are we on the deploy?'] })
  } finally {
    await builder.close()
  }

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

test('the bot row menu opens the most recent session as a tab beside the Bot Chat', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
  const row = page.locator('[data-slot="bots-roster"] [data-roster-key="local::alpha"]')
  await expect(row).toBeVisible({ timeout: 30_000 })

  // The row click is unchanged: the canonical Bot Chat opens, captioned with the bot's name.
  await row.click()
  const botChatTab = page.getByRole('tab', { name: /^Alpha\b/ })
  await expect(botChatTab).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('tab', { name: new RegExp(`^${RECENT_TITLE}\\b`) })).toHaveCount(0)
  await capture(page, '1-row-click-opens-bot-chat')

  await row.click({ button: 'right' })
  const item = page.getByRole('menuitem', { name: 'Open recent session' })
  await expect(item).toBeVisible()
  await expect(item).toBeEnabled()
  await capture(page, '2-row-menu-open-recent-session')
  await item.click()

  const recentTab = page.getByRole('tab', { name: new RegExp(`^${RECENT_TITLE}\\b`) })
  await expect(recentTab).toBeVisible({ timeout: 60_000 })
  await expect(recentTab).toHaveAttribute('aria-selected', 'true')
  // A tab beside the Bot Chat, not in its place.
  await expect(botChatTab).toBeVisible()
  await expect(page.getByText('Where are we on the deploy?').first()).toBeVisible({ timeout: 60_000 })
  await capture(page, '3-recent-session-tab-open')
})
