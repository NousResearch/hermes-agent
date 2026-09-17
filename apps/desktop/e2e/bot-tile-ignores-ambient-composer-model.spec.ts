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
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// A manual model pick in the composer is sticky for ordinary new chats. A `+`
// chat opened beside a bot in Bot Mode targets that bot's profile without
// switching the window's ambient composer, so a pick made for some other open
// session must NOT ride into the bot's session.create: the bot profile's
// configured default model is what the new chat's inference request carries
// (#95264).

type Page = MockBackendFixture['page']

const AMBIENT_PICK = 'ambient-pick-model'

let fixture: MockBackendFixture | null = null

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function settle(page: Page, timeout = 90_000): Promise<void> {
  await page
    .getByText(/Waking up/i)
    .first()
    .waitFor({ state: 'hidden', timeout })
    .catch(() => undefined)
  await page.waitForTimeout(500)
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

test.beforeAll(async () => {
  const mock = await startMockServer({ extraModels: [AMBIENT_PICK] })
  const sandbox = createSandbox('bot-tile-model')
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

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test("a bot's `+` chat runs on the bot profile's model, not the ambient composer pick", async ({}, testInfo) => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const mock = fixture!.mock

  await openBots(page)

  const alphaRow = page
    .getByRole('button', { name: /^alpha\b/i })
    .filter({ visible: true })
    .first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await openUntil(
    () => alphaRow.click(),
    () =>
      expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true }).first()).toBeVisible({
        timeout: 45_000
      })
  )

  await settle(page, 15_000)

  // A manual pick while the bot's canonical chat is open: the window's ambient
  // composer now holds a model that is NOT the profile default. Rows are
  // hover-submenu triggers (reasoning/fast edit); Enter on the row commits.
  await page
    .getByRole('button', { name: /^Model ·/ })
    .first()
    .click()
  await page.getByRole('textbox', { name: 'Search models' }).fill('ambient pick')
  const row = page.getByRole('menuitem', { name: /^Ambient Pick Model/ }).first()
  await expect(row).toBeVisible()
  await row.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('button', { name: /^Model ·/ }).first()).toHaveAccessibleName(new RegExp(AMBIENT_PICK), {
    timeout: 15_000
  })
  await page.screenshot({ path: testInfo.outputPath('bot-chat-manual-pick.png') })

  // `+` beside the bot: a fresh bots-workspace tile for alpha.
  const before = mock.receivedModels.length
  await page.keyboard.press('Control+t')
  await settle(page)
  const composer = page
    .locator('[data-slot="composer-root"] [contenteditable="true"]')
    .filter({ visible: true })
    .first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('hello alpha thread')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: testInfo.outputPath('bot-tile-after-turn.png') })

  await expect.poll(() => mock.receivedModels.length, { timeout: 15_000 }).toBeGreaterThan(before)
  const models = mock.receivedModels.slice(before)
  // The bot profile's configured default (config.yaml `model.default`), never
  // the window's manual composer pick.
  expect(models).not.toContain(AMBIENT_PICK)
  expect(models).toContain('mock-model')
})
