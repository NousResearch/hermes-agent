import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | null = null

async function openBots(page: MockBackendFixture['page']): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function createAgent(page: MockBackendFixture['page'], name: string, title: string): Promise<void> {
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Bot' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Bot' })
  await dialog.getByPlaceholder('inbox-triage').fill(name)
  await dialog.getByPlaceholder('Inbox Triage').fill(title)
  await dialog.getByRole('button', { name: 'Create Bot' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: new RegExp(`^${title}\\b`) }).first()).toBeVisible({ timeout: 30_000 })
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('local bot replaces an open group main workspace', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page

  await openBots(page)
  await createAgent(page, 'programmer', 'Programmer')
  await createAgent(page, 'reviewer', 'Reviewer')

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const title of ['Programmer', 'Reviewer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill('Programmer, Reviewer')
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const groupTab = page.getByRole('tab', { name: /Programmer, Reviewer Close/ })
  const groupComposer = page.getByRole('textbox', { name: 'Message Programmer, Reviewer' }).filter({ visible: true })
  await expect(groupTab).toBeVisible({ timeout: 20_000 })
  await expect(groupTab).toHaveAttribute('aria-selected', 'true')
  await expect(groupComposer).toBeVisible()

  const programmer = page.getByRole('button', { name: /^Programmer\b/ }).filter({ visible: true }).first()
  await programmer.click()

  // The bot's canonical chat opens INTO the main workspace pane (post
  // design-system rework); as the lone pane in the zone it renders chromeless
  // — no "Bot Chat" tab exists until a second pane joins the strip. The
  // handoff is observed by the group surfaces leaving and the bot's chat
  // (here a fresh one: its empty-state splash asks for a first message)
  // taking the main workspace. The first open also spawns the bot's own
  // backend, so give the "Loading session" phase a real chance to clear.
  await expect(page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({
    timeout: 120_000
  })
  await expect(groupTab).toHaveCount(0)
  await expect(groupComposer).toHaveCount(0)
  // No "Waking up…" assertion: the mock backend can keep a bot's wake notice
  // around indefinitely (see bot-mode-closed-chat-stays-closed's settle()),
  // so its presence no longer distinguishes a stranded handoff. The splash
  // and composer above are the proof the bot's chat took the workspace.
  await expect(page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()).toBeVisible()
})
