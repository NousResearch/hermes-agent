import {
  PROVIDER_FAILURE_TRIGGER,
  TOOL_THEN_FAILURE_TEXT,
  TOOL_THEN_FAILURE_TRIGGER
} from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// #92760 "thinks then never speaks": when a member's backend fails the turn
// (bad credentials, provider refusal), the gateway keeps the failed turn under
// `session.resume.inflight` as `{ status: 'error' }` so a reconnecting client
// can rebuild the error bubble. The room engine read that truthy object as
// "still working" and kept extending the member's deadline toward the
// 20-minute cap — the user saw a bot thinking forever with no error anywhere.
// Real Electron + real gateway; the mock provider answers the member with a
// non-retryable 401.

const ROOM = 'Programmer, Reviewer'
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

async function createRoom(page: MockBackendFixture['page']) {
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

  const groupTab = page.getByRole('tab', { name: new RegExp(`${ROOM} Close`) })
  const groupComposer = page.getByRole('textbox', { name: `Message ${ROOM}` }).filter({ visible: true })
  await expect(groupTab).toBeVisible({ timeout: 20_000 })
  await expect(groupTab).toHaveAttribute('aria-selected', 'true')
  await expect(groupComposer).toBeVisible()

  return groupComposer
}

test.beforeEach(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a member whose backend fails the turn is reported at once, not read as busy (#92760)', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)

  await groupComposer.fill(`@programmer ${PROVIDER_FAILURE_TRIGGER}`)
  await groupComposer.press('Enter')
  await expect.poll(() => fixture!.mock.receivedPrompts.some(p => p.includes(PROVIDER_FAILURE_TRIGGER)), { timeout: 60_000 }).toBe(true)

  // The gateway has failed the turn; the room must say so within the base
  // turn timeout instead of extending the deadline on the retained snapshot.
  const activity = page.getByRole('button', { name: /^Activity/ })
  const groupTab = page.getByRole('tab', { name: new RegExp(`${ROOM} Close`) })

  await expect(async () => {
    // A just-created bot's background intro turn can front that bot's chat
    // tab and yank the center away from the room; re-select the room first.
    if ((await groupTab.getAttribute('aria-selected')) !== 'true') {
      await groupTab.click()
    }

    await expect(activity).toContainText('Programmer hit an error', { timeout: 5_000 })
  }).toPass({ timeout: 120_000 })
  // #117366: the row names the cause, not just the fact — the raw error's
  // first line rides along so a stopped backend and a provider refusal differ.
  await expect(activity).toContainText(/Programmer hit an error — \S+/)
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0, { timeout: 30_000 })

  const room = await page.evaluate(name => {
    const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

    return { stranded: Object.keys(rooms[name]?.stranded || {}), running: Boolean(rooms[name]?.running) }
  }, ROOM)

  expect(room.stranded).toEqual([])
  console.log('RETAINED FAILURE: activity =', await activity.textContent(), 'room =', JSON.stringify(room))
  await page.screenshot({ path: test.info().outputPath('retained-failure-after.png') })
})

// The member spoke and called a tool before the provider failed. The session
// ends on the tool row (no failed-turn boundary behind it) and only the
// retained 401 says the turn died: the text written before the tool call must
// not be posted into the room as the member's reply.
test('a member that fails after pre-tool text is reported, and that text is not its reply', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)

  await groupComposer.fill(`@programmer ${TOOL_THEN_FAILURE_TRIGGER}`)
  await groupComposer.press('Enter')

  const activity = page.getByRole('button', { name: /^Activity/ })
  const groupTab = page.getByRole('tab', { name: new RegExp(`${ROOM} Close`) })

  await expect(async () => {
    if ((await groupTab.getAttribute('aria-selected')) !== 'true') {
      await groupTab.click()
    }

    await expect(activity).toContainText('Programmer hit an error', { timeout: 5_000 })
  }).toPass({ timeout: 150_000 })

  const room = await page.evaluate(name => {
    const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

    return { log: (rooms[name]?.log || []).map((entry: { text?: string }) => entry.text || ''), stranded: Object.keys(rooms[name]?.stranded || {}) }
  }, ROOM)

  expect(room.log.some((text: string) => text.includes(TOOL_THEN_FAILURE_TEXT))).toBe(false)
  expect(room.stranded).toEqual([])
  console.log('TOOL THEN FAILURE: activity =', await activity.textContent(), 'log =', JSON.stringify(room.log))
  await page.screenshot({ path: test.info().outputPath('tool-then-failure-after.png') })
})
