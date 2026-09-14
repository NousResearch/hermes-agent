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

// A bot row previews the bot's canonical Bot Chat (the gateway resolves it by
// name on every roster poll). Clicking the row must land on THAT conversation.
// Before this fix a plain click fronted whatever bots-workspace tile the user
// last had open for that bot — a `+` side thread outlived every restart in
// Local Storage and won every click forever, while the row kept previewing the
// Bot Chat. The user saw the sidebar and the center describe two different
// conversations ("sessions not in sync"; support thread 1544460286084391043).

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
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-sync')
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

test('a bot row click lands on the Bot Chat the row previews, not a side thread', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)

  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  const betaRow = page.getByRole('button', { name: /^beta\b/i }).filter({ visible: true }).first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await expect(betaRow).toBeVisible({ timeout: 30_000 })
  const seededTurn = page.getByText('Hello alpha', { exact: true }).filter({ visible: true })

  await openUntil(
    () => alphaRow.click(),
    () => expect(seededTurn.first()).toBeVisible({ timeout: 45_000 })
  )
  await settle(page, 15_000)

  // A `+` side thread for alpha, with a real turn so it is a persisted tile.
  await page.keyboard.press('Control+t')
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('hello alpha thread')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })

  // Leave alpha on the side thread, go to beta, come back via the row.
  await betaRow.click()
  await expect(page.getByText('Hello beta', { exact: true }).filter({ visible: true }).first()).toBeVisible({
    timeout: 60_000
  })
  await settle(page)

  await alphaRow.click()
  // The row previews the Bot Chat; the click must front it.
  await expect(seededTurn.first()).toBeVisible({ timeout: 45_000 })
  // The side thread is still open beside it (scoped to alpha), not closed.
  await expect
    .poll(
      () =>
        page.evaluate(() =>
          [...document.querySelectorAll<HTMLElement>('[data-zone-tabstrip="grp-main"] [data-tree-tab]')]
            .map(element => element.getAttribute('data-tree-tab') ?? '')
            .filter(id => id.startsWith('session-tile:')).length
        ),
      { timeout: 15_000 }
    )
    .toBe(1)
})
