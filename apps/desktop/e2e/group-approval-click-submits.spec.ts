import { APPROVAL_COMMAND_TRIGGER, MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// #91706: a command-approval card in a Bot Mode group room could highlight
// the clicked choice but never send it — the footer "Respond" button was
// off-screen or covered, so the member's hidden session stayed blocked until
// the approval expired. Approvals are a closed choice set: the click IS the
// answer. Driven against the real Electron app, a real gateway with
// `approvals: mode: "manual"`, and mock inference that runs a gated
// `rm -rf` through the real terminal tool.

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
  fixture = await setupMockBackend({ extraConfig: 'approvals:\n  mode: "manual"\n' })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('clicking an approval choice in a group room submits it (#91706)', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  // Count approval.respond frames on the real gateway socket.
  await page.evaluate(() => {
    const send = WebSocket.prototype.send

    ;(window as any).__approvalResponds = [] as string[]

    WebSocket.prototype.send = function (data) {
      const frame = JSON.parse(String(data))

      if (frame.method === 'approval.respond') {
        ;(window as any).__approvalResponds.push(JSON.stringify(frame.params))
      }

      return send.call(this, data)
    }
  })
  const groupComposer = await createRoom(page)

  await groupComposer.fill(`@programmer ${APPROVAL_COMMAND_TRIGGER}`)
  await groupComposer.press('Enter')

  // The member's gated terminal command surfaces as an approval card in the
  // ROOM. Assert inside the room's own card: the same approval also reaches
  // the Desktop's session-level approval surface, and a build that switches
  // tabs on it would otherwise fail here on visibility instead of on the
  // contract below (the click must be the submit).
  const groupTab = page.getByRole('tab', { name: new RegExp(`${ROOM} Close`) })
  const card = page.getByText(/wants to run a command/).locator('xpath=..')
  const once = card.getByRole('button', { name: 'once', exact: true })

  await expect(async () => {
    if ((await groupTab.getAttribute('aria-selected')) !== 'true') {
      await groupTab.click()
    }

    await expect(card).toBeVisible({ timeout: 5_000 })
  }).toPass({ timeout: 90_000 })
  await expect(once).toBeVisible()
  console.log('APPROVAL CARD: visible; responds so far =', await page.evaluate(() => (window as any).__approvalResponds.length))
  await page.screenshot({ path: test.info().outputPath('approval-card.png') })

  await once.click()

  // One approval.respond leaves the Desktop for the click itself — no second
  // "Respond" click required (there is none to click for approvals).
  await expect.poll(() => page.evaluate(() => (window as any).__approvalResponds.length), { timeout: 15_000 }).toBe(1)
  await expect(card.getByRole('button', { name: 'Respond', exact: true })).toHaveCount(0)
  expect(await page.evaluate(() => (window as any).__approvalResponds[0])).toContain('once')

  // The blocked member resumes: the command runs and its reply lands in the room.
  await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 90_000 })
  await expect(once).toHaveCount(0)
  console.log('APPROVAL: submitted on click; responds =', await page.evaluate(() => (window as any).__approvalResponds))
  await page.screenshot({ path: test.info().outputPath('approval-after.png') })
})
