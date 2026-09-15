import fs from 'node:fs'
import path from 'node:path'

import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { setupOwnedBotFixture } from './owned-bot-fixture'
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
let canonicalAlphaId = ''

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function settle(page: Page, timeout = 90_000): Promise<void> {
  // The swap label stays mounted after fading out. Playwright still considers
  // opacity: 0 visible, so wait for the foreground wrapper's completed fade.
  const swapOverlay = page
    .locator('[data-chat-surface]:not([data-chat-unfocused])')
    .filter({ visible: true })
    .locator('div[aria-hidden="true"].transition-opacity')
    .filter({ hasText: /Waking up/i })

  await expect(swapOverlay).toHaveCount(1, { timeout })
  await expect(swapOverlay).toHaveCSS('opacity', '0', { timeout })
  await page.waitForTimeout(500)
}

async function seedBot(hermesHome: string, mockUrl: string, name: string, markChildAttempted: () => void): Promise<string> {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  markChildAttempted()
  const builder = await RealSessionBuilder.start(dir)

  try {
    const session = await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${name}`] })

    return session.sessionId
  } finally {
    await builder.close()
  }
}

// Seeding two durable sessions and booting the real Electron/backend chain can
// exceed Playwright's default 90-second hook budget on a cold Windows install.
test.beforeAll(async () => {
  test.setTimeout(300_000)
  const sandbox = createSandbox('bots-sync')
  fixture = await setupOwnedBotFixture({
    sandbox,
    startMock: startMockServer,
    seed: async (mockUrl, markChildAttempted) => {
      writeMockProviderConfig(sandbox.hermesHome, mockUrl)
      writeEnvFile(sandbox.hermesHome)
      canonicalAlphaId = await seedBot(sandbox.hermesHome, mockUrl, 'alpha', markChildAttempted)
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

test('a bot row click lands on the Bot Chat the row previews, not a side thread', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)

  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  const betaRow = page.getByRole('button', { name: /^beta\b/i }).filter({ visible: true }).first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await expect(betaRow).toBeVisible({ timeout: 30_000 })

  const canonical = page
    .locator('[data-chat-surface][data-session-anchor="workspace"]:not([data-chat-unfocused])')
    .filter({ visible: true })

  const seededTurn = canonical.getByText('Hello alpha', { exact: true })

  // One user selection must finish by itself. Retrying this click hid cold
  // activation failures even while the test runner reported zero retries.
  await alphaRow.click()
  await expect(seededTurn).toBeVisible({ timeout: 45_000 })
  await settle(page, 15_000)

  // A `+` side thread for alpha, populated through its own focused composer.
  await page.keyboard.press('Control+t')

  const newSide = page
    .locator('[data-chat-surface][data-session-anchor^="session-tile:"]:not([data-chat-unfocused])')
    .filter({ visible: true })

  await expect(newSide).toHaveCount(1, { timeout: 15_000 })

  const sideAnchor = await newSide.getAttribute('data-session-anchor')

  expect(sideAnchor).toMatch(/^session-tile:[a-zA-Z0-9_-]+$/)

  const sideId = sideAnchor!.slice('session-tile:'.length)
  const side = page.locator(`[data-chat-surface][data-session-anchor="${sideAnchor}"]`)
  const sideTab = page.locator(`[data-zone-tabstrip="grp-main"] [data-tree-tab="${sideAnchor}"]`)
  const sidePrompt = 'hello alpha thread'

  expect(sideId).not.toBe(canonicalAlphaId)
  await expect(side).toHaveAttribute('data-composer-target', `tile:${sideId}`)
  await expect(sideTab).toHaveAttribute('aria-selected', 'true')
  // No old seeded reply may satisfy the side thread's completion assertion.
  await expect(side.getByText('Hello alpha', { exact: true })).toHaveCount(0)
  await expect(side.getByText(MOCK_REPLY, { exact: true })).toHaveCount(0)

  const composer = side.locator('[data-slot="composer-root"] [contenteditable="true"]')

  await expect(composer).toHaveCount(1)
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await expect(composer).toBeFocused()
  await composer.fill(sidePrompt)
  await composer.press('Enter')
  await expect(side.getByText(sidePrompt, { exact: true })).toBeVisible({ timeout: 15_000 })
  await expect(side.getByText(MOCK_REPLY, { exact: true })).toBeVisible({ timeout: 60_000 })

  const sideCompletion = side.locator('[data-slot="aui_message-streaming-marker"]')

  await expect(sideCompletion).toHaveCount(1)
  await expect(sideCompletion).not.toHaveAttribute('data-message-streaming', 'true', { timeout: 60_000 })
  await expect(side.getByRole('alert')).toHaveCount(0)
  await test.info().attach('distinct-session-identities', {
    body: JSON.stringify({ canonicalAlphaId, sideId }),
    contentType: 'application/json'
  })
  await test.info().attach('populated-side-before-switch', {
    body: await page.screenshot(),
    contentType: 'image/png'
  })

  // Leave alpha on the side thread, go to beta, come back via the row.
  await betaRow.click()
  await expect(canonical.getByText('Hello beta', { exact: true })).toBeVisible({
    timeout: 60_000
  })
  await settle(page)

  await alphaRow.click()
  // The row previews the Bot Chat; the click must front it.
  await expect(seededTurn).toBeVisible({ timeout: 45_000 })
  await expect(canonical.getByText(sidePrompt, { exact: true })).toHaveCount(0)
  // Retain the exact populated side tile, not merely any unused empty tile.
  await expect(sideTab).toHaveCount(1)
  await expect(sideTab).toHaveAttribute('aria-selected', 'false')
  await expect(page.locator('[data-zone-tabstrip="grp-main"] [data-tree-tab^="session-tile:"]')).toHaveCount(1)
  await settle(page, 15_000)

  // Reopen the retained side without resending, then explicitly return to the
  // canonical row. This is another navigation stage, never a failed-click retry.
  await sideTab.click()
  await expect(sideTab).toHaveAttribute('aria-selected', 'true')
  await expect(side.getByText(sidePrompt, { exact: true })).toBeVisible()
  await expect(side.getByText(MOCK_REPLY, { exact: true })).toBeVisible()
  await expect(side.getByText('Hello alpha', { exact: true })).toHaveCount(0)
  await alphaRow.click()
  await expect(seededTurn).toBeVisible({ timeout: 45_000 })
  await expect(canonical.getByText(sidePrompt, { exact: true })).toHaveCount(0)
  await settle(page, 15_000)
  await test.info().attach('final-alpha-before-cleanup', {
    body: await page.screenshot(),
    contentType: 'image/png'
  })
})
