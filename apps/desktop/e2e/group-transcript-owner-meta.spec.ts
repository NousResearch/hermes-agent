import os from 'node:os'
import path from 'node:path'

import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// Group Chat identity follows the OWNER-qualified bot meta (#96432, #101382,
// #102294, #94869): bot meta is persisted under `connectionId::profile`
// (botMetaKey), while the transcript speaker row and the Activity feed used
// to probe the bare profile name — a stale v1 relic at best, nothing for a
// remote or renamed bot at worst. Re-titling a member after the room exists
// is the single-gateway form of the bug: the roster repaints, the room does
// not. The room log itself is untouched by this — only presentation moves.

const ROOM = 'Ops Room'
const NEW_TITLE = 'Infra Ops'
const SHOTS = path.join(os.tmpdir(), 'batchbots/group-identity-members/shots')
let fixture: MockBackendFixture | null = null

type Page = MockBackendFixture['page']

const roster = (page: Page) => page.locator('[data-slot="bots-roster"]')
const botRow = (page: Page, name: string) => roster(page).locator(`[data-roster-key="local::${name}"]`)
const speakerLabels = (page: Page) => page.locator('button[title="Show full handle"]')

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
  await expect(botRow(page, name)).toBeVisible({ timeout: 30_000 })
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

  return frontRoom(page)
}

/** A just-created Bot's canonical Bot Chat hydrates late and can front its
 *  "Draft" tab over the room tab; bring the room back before typing into it. */
async function frontRoom(page: Page) {
  const tab = page.getByRole('tab', { name: ROOM }).first()
  await expect(tab).toBeVisible({ timeout: 20_000 })

  if ((await tab.getAttribute('aria-selected')) !== 'true') {
    await tab.click()
  }

  const composer = page.getByRole('textbox', { name: `Message ${ROOM}` }).filter({ visible: true })
  await expect(composer).toBeVisible({ timeout: 20_000 })

  return composer
}

async function storedMeta(page: Page) {
  return page.evaluate(() => {
    const meta = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.bot-meta-v2') || '{}')

    return { bare: meta.programmer?.title ?? null, routed: meta['local::programmer']?.title ?? null }
  })
}

test.beforeEach(async () => {
  fixture = await setupMockBackend({ mockServer: { replyForPrompt: () => MOCK_REPLY } })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a re-titled member is re-labelled in the room transcript and Activity feed', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  await createRoom(page)

  const composer = await frontRoom(page)
  await composer.fill('@programmer hello')
  await composer.press('Enter')
  await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 90_000 })
  await expect(speakerLabels(page).first()).toHaveText('Programmer')
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)

  // Re-title the member through the real Edit dialog. The write lands on the
  // route-qualified key (`local::programmer`); the bare-name relic keeps the
  // old title — exactly the two-slot store reporters found in leveldb.
  await botRow(page, 'programmer').click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Edit…' }).click()
  const edit = page.getByRole('dialog', { name: 'Edit profile' })
  await expect(edit).toBeVisible({ timeout: 30_000 })
  // The Title input's placeholder is the bot's current display name.
  await edit.getByPlaceholder('Programmer').fill(NEW_TITLE)
  await edit.getByRole('button', { name: 'Save' }).click()
  await expect(edit).toBeHidden({ timeout: 30_000 })
  await expect(botRow(page, 'programmer')).toContainText(NEW_TITLE, { timeout: 30_000 })
  await expect.poll(async () => (await storedMeta(page)).routed, { timeout: 15_000 }).toBe(NEW_TITLE)
  expect((await storedMeta(page)).bare).toBe('Programmer')

  // The room must show the same identity the roster does.
  await expect(speakerLabels(page).first()).toHaveText(NEW_TITLE, { timeout: 15_000 })
  await page.getByRole('button', { name: /^Activity/ }).click()
  await expect(page.getByText(`${NEW_TITLE} replied`, { exact: true })).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText('Programmer replied', { exact: true })).toHaveCount(0)
  await page.screenshot({ path: `${SHOTS}/group-transcript-owner-meta-after.png` })

  // Control: the user's own lines never pick up bot meta, and the durable
  // room log still stores the profile id, not the title.
  await expect(page.getByText('You', { exact: true }).first()).toBeVisible()
  const logFrom = await page.evaluate(
    room => (JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')[room]?.log || []).map((entry: any) => entry.from.name),
    ROOM
  )
  expect(logFrom).toEqual(['You', 'programmer'])
})
