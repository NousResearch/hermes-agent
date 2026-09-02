import fs from 'node:fs'
import path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { MOCK_REPLY, startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

const RECOVERABLE_TURN = 'Recover this side conversation'

async function openBots(page: Awaited<ReturnType<typeof launchDesktop>>['page']): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function settleBotWake(page: Awaited<ReturnType<typeof launchDesktop>>['page']): Promise<void> {
  await page
    .getByText(/Waking up/i)
    .first()
    .waitFor({ state: 'hidden', timeout: 90_000 })
    .catch(() => undefined)
  await page.waitForTimeout(500)
}

const botSessionTabs = (page: Awaited<ReturnType<typeof launchDesktop>>['page']) =>
  page.evaluate(() =>
    [...document.querySelectorAll<HTMLElement>('[data-zone-tabstrip="grp-main"] [data-tree-tab]')]
      .map(element => element.getAttribute('data-tree-tab') ?? '')
      .filter(id => id.startsWith('session-tile:'))
  )

const persistedAlphaPreviews = (page: Awaited<ReturnType<typeof launchDesktop>>['page']) =>
  page.evaluate(async () => {
    const desktop = (
      window as unknown as {
        hermesDesktop: { api: <T>(request: { path: string }) => Promise<T> }
      }
    ).hermesDesktop

    const result = await desktop.api<{ sessions?: Array<{ preview?: null | string }> }>({
      path: '/api/profiles/sessions?profile=alpha&limit=200&offset=0&min_messages=1&archived=include&order=recent&include_hidden=true'
    })

    return (result.sessions || []).map(session => session.preview || '')
  })

async function seedBot(hermesHome: string, mockUrl: string): Promise<void> {
  const dir = path.join(hermesHome, 'profiles', 'alpha')
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  const builder = await RealSessionBuilder.start(dir)

  try {
    await builder.createSession({ title: 'Bot Chat', turns: ['Hello alpha'] })
  } finally {
    await builder.close()
  }
}

async function createAndCloseRecoverableConversation(
  page: Awaited<ReturnType<typeof launchDesktop>>['page']
): Promise<void> {
  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  await alphaRow.click()
  await expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true })).toBeVisible({ timeout: 60_000 })
  await settleBotWake(page)

  // The Bot-workspace + path creates a hidden side session: hidden from global
  // Sessions, but durable and therefore required in this profile history.
  const tabsBefore = await botSessionTabs(page)
  await page.keyboard.press('Control+t')
  await expect.poll(() => botSessionTabs(page), { timeout: 30_000 }).toHaveLength(tabsBefore.length + 1)
  const tabsAfter = await botSessionTabs(page)
  const sideTabId = tabsAfter.find(id => !tabsBefore.includes(id))

  expect(sideTabId).toBeTruthy()

  const mainGroup = page.locator('[data-tree-group="grp-main"]')
  const sideTab = mainGroup.locator(`[data-zone-tabstrip="grp-main"] [data-tree-tab="${sideTabId}"]`)

  await expect(sideTab).toBeVisible({ timeout: 15_000 })
  await expect(sideTab).toHaveAttribute('aria-selected', 'true')

  const composer = mainGroup.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).last()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.fill(RECOVERABLE_TURN)
  await page.keyboard.press('Enter')
  await expect(mainGroup.locator('[data-chat-surface]').filter({ visible: true }).getByText(MOCK_REPLY, { exact: true })).toBeVisible({
    timeout: 60_000
  })

  // Deliberately close the projection. The durable conversation must remain in
  // history before and after the whole Desktop process relaunches.
  await sideTab.hover()
  await sideTab.getByRole('button', { name: 'Close' }).click({ force: true })
  await page.getByRole('button', { name: 'Close tab', exact: true }).click({ timeout: 2_000 }).catch(() => undefined)
  await expect(sideTab).toHaveCount(0, { timeout: 15_000 })
  await expect.poll(() => persistedAlphaPreviews(page), { timeout: 15_000 }).toContain(RECOVERABLE_TURN)
}

async function openRecoverableConversation(page: Awaited<ReturnType<typeof launchDesktop>>['page']): Promise<void> {
  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await alphaRow.click()

  const history = page.getByRole('button', { name: 'Conversation history for Alpha' })
  await expect(history).toBeVisible({ timeout: 30_000 })
  await history.click()

  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText('Alpha conversations')).toBeVisible()
  const recovered = dialog.getByRole('button', { name: `Open ${RECOVERABLE_TURN}`, exact: true })
  await expect(recovered).toBeVisible({ timeout: 60_000 })
  await recovered.click()
  await expect(dialog).toBeHidden({ timeout: 90_000 })
  await expect(page.getByText(RECOVERABLE_TURN, { exact: true }).filter({ visible: true })).toBeVisible({
    timeout: 120_000
  })
}

test('a hidden Bot side conversation remains discoverable after its tab closes and Desktop relaunches', async () => {
  test.setTimeout(480_000)
  const mock = await startMockServer()
  const sandbox = createSandbox('bot-conversation-history')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  await seedBot(sandbox.hermesHome, mock.url)

  let running = await launchDesktop(buildAppEnv(sandbox))

  try {
    await waitForAppReady({ ...running, mock, mockUrl: mock.url, sandbox, cleanup: async () => undefined }, 120_000)
    await openBots(running.page)
    await createAndCloseRecoverableConversation(running.page)
    await openRecoverableConversation(running.page)

    await running.app.close()
    running = await launchDesktop(buildAppEnv(sandbox))
    await waitForAppReady({ ...running, mock, mockUrl: mock.url, sandbox, cleanup: async () => undefined }, 120_000)
    await openBots(running.page)
    await openRecoverableConversation(running.page)
  } finally {
    await running.app.close().catch(() => undefined)
    await mock.close()
    sandbox.cleanup()
  }
})
