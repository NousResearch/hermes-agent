import fs from 'node:fs'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, waitForAppReady } from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Bot role: Edit profile gains a Role field. Saving it writes
// `ui_meta.hermes-bots.role` through the REAL gateway (the assertion reads it back
// off the profile.yaml on disk) and the roster row shows it beside the name.

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

// BOT_ROLE_SCREENSHOT_DIR=<dir> saves full-window captures at each state.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.BOT_ROLE_SCREENSHOT_DIR

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
  const sandbox = createSandbox('bot-role')
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

test('Edit profile → Role writes ui_meta.hermes-bots.role through the gateway', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const hermesHome = fixture!.sandbox.hermesHome

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(botRow(page, 'lucky')).toBeVisible({ timeout: 30_000 })
  await expect(botRow(page, 'lucky').locator('[data-slot="bot-role"]')).toHaveCount(0)

  // Right-click lucky → Edit… → the Edit profile dialog with the new Role field.
  await botRow(page, 'lucky').click({ button: 'right' })
  await page.getByRole('menuitem', { name: /^Edit/ }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText('Edit profile')).toBeVisible()
  const roleField = dialog.getByPlaceholder('e.g. Deep Research')
  await expect(roleField).toBeVisible()
  await capture(page, '1-edit-dialog-role-field')

  await roleField.fill('Deep Research')
  await capture(page, '2-role-filled')
  await dialog.getByRole('button', { name: 'Save' }).click()

  // End-to-end proof: the REAL gateway persisted the role into the profile's ui_meta.
  const profileYaml = path.join(hermesHome, 'profiles', 'lucky', 'profile.yaml')
  await expect
    .poll(() => (fs.existsSync(profileYaml) ? fs.readFileSync(profileYaml, 'utf8') : ''), { timeout: 15_000 })
    .toMatch(/hermes-bots:[\s\S]*role: Deep Research/)

  // The roster shows the role beside the name once the write lands, and the other bot is untouched.
  await expect(botRow(page, 'lucky').locator('[data-slot="bot-role"]')).toHaveText('Deep Research', { timeout: 15_000 })
  await expect(botRow(page, 'researcher').locator('[data-slot="bot-role"]')).toHaveCount(0)
  await capture(page, '3-roster-with-role')
})
