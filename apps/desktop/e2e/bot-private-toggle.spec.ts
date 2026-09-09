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
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Per-bot mesh visibility: the row menu gains Make private / Make public beside
// Pin to top. The toggle writes `ui_meta.hermes-bots.private` through the REAL
// gateway, so the assertion below reads it back off the profile.yaml on disk.

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

// BOT_PRIVATE_SCREENSHOT_DIR=<dir> saves full-window captures at each state.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.BOT_PRIVATE_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })
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

const roster = (page: Page) => page.locator('[data-slot="bots-roster"]')
const botRow = (page: Page, name: string) => roster(page).locator(`[data-roster-key="local::${name}"]`)

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bot-private')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  for (const name of ['lucky', 'researcher']) {
    await seedBot(sandbox.hermesHome, mock.url, name)
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

test('Make private writes the flag through the gateway and the menu flips to Make public', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const hermesHome = fixture!.sandbox.hermesHome

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(botRow(page, 'lucky')).toBeVisible({ timeout: 30_000 })
  await expect(botRow(page, 'researcher')).toBeVisible({ timeout: 30_000 })
  await capture(page, '1-roster')

  // Right-click lucky: the new item sits beside Pin to top.
  await botRow(page, 'lucky').click({ button: 'right' })
  await expect(page.getByRole('menuitem', { name: 'Pin to top' })).toBeVisible()
  const makePrivate = page.getByRole('menuitem', { name: 'Make private' })
  await expect(makePrivate).toBeVisible()
  await capture(page, '2-menu-make-private')

  await makePrivate.click()
  // The notice names the bot and says what changed for OTHER agents.
  await expect(page.getByText(/is now private/)).toBeVisible({ timeout: 15_000 })
  await capture(page, '3-toast-now-private')

  // End-to-end proof: the REAL gateway persisted it into the profile's ui_meta.
  const profileYaml = path.join(hermesHome, 'profiles', 'lucky', 'profile.yaml')
  await expect
    .poll(() => (fs.existsSync(profileYaml) ? fs.readFileSync(profileYaml, 'utf8') : ''), { timeout: 15_000 })
    .toMatch(/hermes-bots:[\s\S]*private: true/)

  // Re-open the menu: the label has flipped.
  await botRow(page, 'lucky').click({ button: 'right' })
  await expect(page.getByRole('menuitem', { name: 'Make public' })).toBeVisible()
  await expect(page.getByRole('menuitem', { name: 'Make private' })).toHaveCount(0)
  await capture(page, '4-menu-make-public')
  await page.keyboard.press('Escape')
})
