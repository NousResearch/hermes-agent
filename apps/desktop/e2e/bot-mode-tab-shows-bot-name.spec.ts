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
import { setupOwnedBotFixture } from './owned-bot-fixture'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Every bot's canonical chat is STORED under the same title ("Bot Chat" — the
// name the gateway resolves it by), so the main tab strip captioned every open
// bot chat identically and two bots' tabs were indistinguishable (#99152). The
// tab must read the bot's display name while the stored title stays canonical.

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

async function seedBot(hermesHome: string, mockUrl: string, name: string, markChildAttempted: () => void): Promise<void> {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  markChildAttempted()
  const builder = await RealSessionBuilder.start(dir)

  try {
    await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${name}`] })
  } finally {
    await builder.close()
  }
}

/** Every tab caption in the main strip (the main `workspace` tab + tiles). */
function mainStripTabTitles(page: Page): Promise<string[]> {
  return page.evaluate(() =>
    [...document.querySelectorAll<HTMLElement>('[data-zone-tabstrip="grp-main"] [data-tree-tab]')].map(element =>
      (element.textContent ?? '').trim()
    )
  )
}

// Seeding two durable sessions and booting the real Electron/backend chain can
// exceed Playwright's default 90-second hook budget on a cold Windows install.
test.beforeAll(async () => {
  test.setTimeout(300_000)
  const sandbox = createSandbox('bots-tabname')
  fixture = await setupOwnedBotFixture({
    sandbox,
    startMock: startMockServer,
    seed: async (mockUrl, markChildAttempted) => {
      writeMockProviderConfig(sandbox.hermesHome, mockUrl)
      writeEnvFile(sandbox.hermesHome)
      await seedBot(sandbox.hermesHome, mockUrl, 'alpha', markChildAttempted)
      await seedBot(sandbox.hermesHome, mockUrl, 'beta', markChildAttempted)
    },
    launch: onLaunched => launchDesktop(buildAppEnv(sandbox), onLaunched),
    ready: current => waitForAppReady(current, 120_000),
    onRetained: async () => {}
  })
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test("an open Bot Chat's tab reads the bot's name, not the canonical 'Bot Chat' title", async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)

  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })

  await openUntil(
    () => alphaRow.click(),
    () =>
      expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true }).first()).toBeVisible({
        timeout: 45_000
      })
  )

  // A `+` side thread beside the Bot Chat gives the main zone a tab strip —
  // the surface where every bot chat used to read "Bot Chat".
  await page.keyboard.press('Control+t')
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('hello alpha thread')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })

  await expect.poll(() => mainStripTabTitles(page), { timeout: 15_000 }).toHaveLength(2)
  const captions = await mainStripTabTitles(page)
  expect(captions.some(caption => /alpha/i.test(caption))).toBe(true)
  expect(captions.some(caption => /bot chat/i.test(caption))).toBe(false)
})
