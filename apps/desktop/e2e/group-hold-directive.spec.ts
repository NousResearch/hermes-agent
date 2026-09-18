import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// Sticky Stop holds (#93129) are set and released by the hold classifier in
// group-rounds.ts. Two community reports against it:
//   #103893 — a German room's filler word ("@impl go, das ist halt ein Test")
//             held the mentioned bot on every resend;
//   #97740  — a room stopped with the Stop button stayed silent for "@all <task>"
//             because only the literal "@all resume" released the holds.
// Both are asserted against the real Electron app + real gateway + mock
// inference: the member's prompt must reach the provider, and the persisted
// room record must carry no hold.

const ROOM = 'Programmer, Reviewer'
const FIRST_REPLY = 'FIRST_REPLY: initial work completed'
let fixture: MockBackendFixture | null = null

async function roomHolds(page: MockBackendFixture['page']) {
  return page.evaluate(name => {
    const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

    return Object.keys(rooms[name]?.holds || {})
  }, ROOM)
}

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

/** A bot created moments ago runs its intro turn in the background; when it
 *  lands, the roster fronts that bot's chat tab and yanks the center away from
 *  the room. Re-select the room before reading room content. */
async function showRoom(page: MockBackendFixture['page']) {
  const groupTab = page.getByRole('tab', { name: new RegExp(`${ROOM} Close`) })

  if ((await groupTab.getAttribute('aria-selected')) !== 'true') {
    await groupTab.click()
  }
}

test.beforeEach(async () => {
  fixture = await setupMockBackend({ mockServer: {
    holdFirstCompletionContaining: 'LANE_H_FIRST',
    replyForPrompt: prompt => prompt.includes('LANE_H_FIRST') ? FIRST_REPLY : MOCK_REPLY
  } })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a distant German filler word does not hold the mentioned bot (#103893)', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)

  await groupComposer.fill('@programmer go, das ist halt ein Test LANE_H_GERMAN')
  await groupComposer.press('Enter')

  // The bot was addressed: its turn prompt must reach the provider ...
  await expect.poll(() => fixture!.mock.receivedPrompts.some(p => p.includes('LANE_H_GERMAN')), { timeout: 60_000 }).toBe(true)
  // ... and the persisted room record must not carry a hold for it.
  expect(await roomHolds(page)).toEqual([])
  await expect(async () => {
    await showRoom(page)
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 5_000 })
  }).toPass({ timeout: 60_000 })
  await expect(page.locator('[data-slot="group-hold-status"]')).toHaveCount(0)
  console.log('GERMAN FILLER: prompt delivered, holds =', JSON.stringify(await roomHolds(page)))
  await page.screenshot({ path: test.info().outputPath('german-filler-after.png') })
})

test('@all with a task re-engages a room the user stopped (#97740)', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)

  await groupComposer.fill('@programmer LANE_H_FIRST')
  await groupComposer.press('Enter')
  await fixture!.mock.waitForHeldCompletion()
  await expect(async () => {
    await showRoom(page)
    await page.getByRole('button', { name: 'Stop', exact: true }).click({ timeout: 5_000 })
  }).toPass({ timeout: 60_000 })
  fixture!.mock.releaseHeldStream()
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)
  // Stop holds every member durably.
  await expect.poll(() => roomHolds(page)).toHaveLength(2)
  // The durable room status (#99094) names the held members.
  await expect(page.locator('[data-slot="group-hold-status"]')).toBeVisible()
  console.log('STOPPED: holds =', JSON.stringify(await roomHolds(page)))
  await page.screenshot({ path: test.info().outputPath('all-task-stopped.png') })

  // No "resume" anywhere: addressing the whole room is the release.
  await groupComposer.fill('@all LANE_H_ALLTASK tell me a joke each')
  await groupComposer.press('Enter')
  await expect.poll(() => fixture!.mock.receivedPrompts.filter(p => p.includes('LANE_H_ALLTASK')).length, { timeout: 90_000 }).toBeGreaterThanOrEqual(2)
  expect(await roomHolds(page)).toEqual([])
  await expect(async () => {
    await showRoom(page)
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 5_000 })
  }).toPass({ timeout: 60_000 })
  await expect(page.locator('[data-slot="group-hold-status"]')).toHaveCount(0)
  console.log('@all TASK: prompts delivered to', fixture!.mock.receivedPrompts.filter(p => p.includes('LANE_H_ALLTASK')).length, 'members; holds =', JSON.stringify(await roomHolds(page)))
  await page.screenshot({ path: test.info().outputPath('all-task-after.png') })
})
