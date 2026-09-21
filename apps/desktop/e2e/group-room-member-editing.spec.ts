import os from 'node:os'
import path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// #91329 / #110736 — membership of an EXISTING room is editable from Group
// settings: the picker pre-checks the current seats, Save rewrites both
// membership sources the room reads (each local Bot's groups[] metadata and
// the room's durable member descriptors), and the next unaddressed send seats
// exactly the saved roster — the removed Bot never takes a turn again.

const ROOM = 'Programmer, Reviewer'
const SHOTS = path.join(os.tmpdir(), 'batchbots/features-groups-ui/shots')
let fixture: MockBackendFixture | null = null

type Page = MockBackendFixture['page']

async function durableMembers(page: Page) {
  return page.evaluate(room => {
    const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

    return ((rooms[room]?.members || []) as { name: string }[]).map(member => member.name).sort()
  }, ROOM)
}

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
  await createAgent(page, 'planner', 'Planner')

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

test.beforeEach(async () => {
  fixture = await setupMockBackend({ mockServer: {} })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('Group settings edits the room roster and the next round seats only the saved members', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  await createRoom(page)

  expect(await durableMembers(page)).toEqual(['programmer', 'reviewer'])

  // Group settings → Manage members: the checklist opens pre-checked.
  await page.getByRole('button', { name: `Group settings for ${ROOM}` }).click()
  await page.getByRole('dialog', { name: 'Group settings' }).getByRole('button', { name: /^Manage members \(2\)/ }).click()

  const picker = page.getByRole('dialog', { name: 'Manage members' })
  await expect(picker).toBeVisible()
  const row = (title: string) => picker.getByRole('checkbox').filter({ hasText: title })
  await expect(row('Programmer')).toHaveAttribute('aria-checked', 'true')
  await expect(row('Reviewer')).toHaveAttribute('aria-checked', 'true')
  await expect(row('Planner')).toHaveAttribute('aria-checked', 'false')
  await page.screenshot({ path: `${SHOTS}/member-picker-open.png` })

  // Cancel changes nothing.
  await row('Planner').click()
  await picker.getByRole('button', { name: 'Cancel' }).click()
  await expect(picker).toBeHidden()
  expect(await durableMembers(page)).toEqual(['programmer', 'reviewer'])

  // Header door → swap Reviewer for Planner → Save.
  await page.getByRole('button', { name: 'Manage group members' }).click()
  await expect(row('Planner')).toHaveAttribute('aria-checked', 'false')
  await row('Planner').click()
  await row('Reviewer').click()
  await page.screenshot({ path: `${SHOTS}/member-picker-edited.png` })
  await picker.getByRole('button', { name: 'Save members' }).click()
  await expect(picker).toBeHidden()

  // Storage truth: the durable descriptors carry the new roster …
  await expect.poll(() => durableMembers(page)).toEqual(['planner', 'programmer'])
  // … and so does each local Bot's groups[] metadata (the plugin's meta snapshot).
  await expect.poll(() => page.evaluate(room => {
    const meta = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.bot-meta-v2') || '{}')

    return Object.entries(meta as Record<string, any>)
      .filter(([, value]) => (value?.groups || []).includes(room))
      .map(([key]) => key.split('::').pop())
      .sort()
  }, ROOM), { timeout: 30_000 }).toEqual(['planner', 'programmer'])

  // An unaddressed send seats exactly the saved roster.
  const groupComposer = await frontRoom(page)
  await groupComposer.fill('ROOM_EDIT_ROUND who is here?')
  await groupComposer.press('Enter')
  await expect.poll(() => fixture!.mock.receivedPrompts.filter(p => p.includes('ROOM_EDIT_ROUND')).length, { timeout: 90_000 }).toBeGreaterThanOrEqual(2)
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0, { timeout: 90_000 })

  const speakers = fixture!.mock.receivedPrompts
    .filter(p => p.includes('ROOM_EDIT_ROUND'))
    .map(p => /You are @([a-z0-9_-]+)/.exec(p)?.[1])
  console.log('ROUND SPEAKERS after roster edit:', JSON.stringify(speakers))
  expect(new Set(speakers)).toEqual(new Set(['programmer', 'planner']))
  expect(speakers).not.toContain('reviewer')
  await page.screenshot({ path: `${SHOTS}/member-picker-after-round.png` })
})
