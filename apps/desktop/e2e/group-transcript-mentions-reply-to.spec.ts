import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// #91359 / #89883 — group-room transcript affordances:
//  - a recognized @mention in a sent message renders as an inline reference
//    (`.ref[data-ref=agent|human|broadcast]`); unknown @tokens stay prose;
//  - a bot's message carries "Reply to <bot>", which seeds `@tag ` into the
//    room composer so the next send routes to that member only.

const ROOM = 'Programmer, Reviewer'
const SHOTS = '/tmp/batchbots/features-groups-ui/shots'
let fixture: MockBackendFixture | null = null

type Page = MockBackendFixture['page']

async function openBots(page: Page): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function createAgent(page: Page, name: string, title: string): Promise<void> {
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Bot' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Bot' })
  await dialog.getByPlaceholder('inbox-triage').fill(name)
  await dialog.getByPlaceholder('Inbox Triage').fill(title)
  await dialog.getByRole('button', { name: 'Create Bot' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: new RegExp(`^${title}\\b`) }).first()).toBeVisible({ timeout: 30_000 })
}

async function createRoom(page: Page) {
  await openBots(page)
  await createAgent(page, 'programmer', 'Programmer')
  await createAgent(page, 'reviewer', 'Reviewer')

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const title of ['Programmer', 'Reviewer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill(ROOM)
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const groupComposer = page.getByRole('textbox', { name: `Message ${ROOM}` }).filter({ visible: true })
  await expect(groupComposer).toBeVisible({ timeout: 20_000 })

  return groupComposer
}

test.beforeEach(async () => {
  fixture = await setupMockBackend({ mockServer: {} })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('mentions render as inline references and Reply-to seeds the composer', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)

  await groupComposer.fill('@programmer MENTION_RENDER check this, then @user decides; not ops@example.com and not @nobody')
  await groupComposer.press('Enter')

  // The sent user line: exactly the routed bot and the human token are refs.
  const sent = page.getByText(/MENTION_RENDER check this/).filter({ visible: true }).first()
  await expect(sent).toBeVisible()
  await expect(sent.locator('.ref[data-ref="agent"]')).toHaveText('@programmer')
  await expect(sent.locator('.ref[data-ref="human"]')).toHaveText('@user')
  await expect(sent.locator('.ref')).toHaveCount(2)
  await expect(sent).toContainText('@nobody')

  // The mock reply from programmer arrives; its hover action targets that bot.
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0, { timeout: 90_000 })
  const replyButton = page.getByRole('button', { name: 'Reply to Programmer' }).first()
  await expect(replyButton).toBeAttached()
  await sent.hover()
  await page.screenshot({ path: `${SHOTS}/mention-refs.png` })

  await groupComposer.fill('')
  await replyButton.click({ force: true })
  await expect(groupComposer).toHaveValue('@programmer ')
  await page.screenshot({ path: `${SHOTS}/reply-to-seeded.png` })

  // Idempotent: a second click never doubles the tag.
  await replyButton.click({ force: true })
  await expect(groupComposer).toHaveValue('@programmer ')

  // Storage truth: the stored text is untouched by rendering.
  const stored = await page.evaluate(room => {
    const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

    return (rooms[room]?.log || []).map((entry: any) => entry.text)[0]
  }, ROOM)

  expect(stored).toBe('@programmer MENTION_RENDER check this, then @user decides; not ops@example.com and not @nobody')
})
