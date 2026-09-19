import fs from 'node:fs'
import path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// A group chat's roster row gets the same organisation affordances a bot row
// has: right-click → Pin to top floats the room above unpinned rows and
// persists on the room record; right-click → Move to section files the room
// into a user-made section, with membership riding the durable group-chats
// record rather than any bot's profile meta (hermes-agent#89813, #105544).

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

// BOT_GROUP_ROW_SCREENSHOT_DIR=<dir> saves full-window captures at the key
// states — PR evidence; never part of the assertions.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.BOT_GROUP_ROW_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })
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

const ROOM = 'Standup'

const roster = (page: Page) => page.locator('[data-slot="bots-roster"]')
const groupRow = (page: Page) => roster(page).getByRole('button', { name: new RegExp(`^${ROOM}, `) })
const sectionLabel = (page: Page, name: string) => page.locator('span.truncate', { hasText: new RegExp(`^${name}$`, 'i') })

const sectionBlock = (page: Page, name: string) =>
  roster(page).locator('[data-slot="bots-section"]').filter({ has: sectionLabel(page, name) })

/** Roster rows top to bottom: bot rows by roster key, the room by its label. */
async function rowOrder(page: Page): Promise<string[]> {
  return roster(page)
    .locator(`[data-roster-key], button[aria-label^="${ROOM},"]`)
    .evaluateAll(rows => rows.map(row => (row as HTMLElement).dataset.rosterKey || 'group'))
}

async function storedRoom(page: Page): Promise<{ pinned?: boolean; sectionId?: null | string }> {
  return page.evaluate(
    room => JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')[room] || {},
    ROOM
  )
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-group-row')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  for (const name of ['alpha', 'beta']) {
    await seedBot(sandbox.hermesHome, mock.url, name)
  }

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

test('a group chat row pins to the top and files into a user section like a bot row', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
  await expect(roster(page).locator('[data-roster-key="local::alpha"]')).toBeVisible({ timeout: 30_000 })
  await expect(roster(page).locator('[data-roster-key="local::beta"]')).toBeVisible({ timeout: 30_000 })

  // Make a room from the two bots.
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()
  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const name of ['Alpha @alpha', 'Beta @beta']) {
    await dialog.getByRole('checkbox', { name }).click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill(ROOM)
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })

  // Back to the roster: the fresh, quiet room sinks under the bots that have
  // history, so a pin has something to change.
  await tab.click()
  await expect(groupRow(page)).toBeVisible({ timeout: 30_000 })
  await expect.poll(() => rowOrder(page).then(rows => rows.at(-1))).toBe('group')
  await capture(page, '1-room-unpinned-last')

  await groupRow(page).click({ button: 'right' })
  await expect(page.getByRole('menuitem', { name: 'Pin to top' })).toBeVisible()
  await capture(page, '2-group-row-menu')
  await page.getByRole('menuitem', { name: 'Pin to top' }).click()

  await expect.poll(() => rowOrder(page).then(rows => rows[0])).toBe('group')
  await expect(groupRow(page).locator('.codicon-pinned')).toHaveCount(1)
  // Storage truth: the flag is on the room record, where a reload hydrates it.
  await expect.poll(async () => (await storedRoom(page)).pinned).toBe(true)
  await capture(page, '3-room-pinned-first')

  // Unpin through the same item — the room sinks again.
  await groupRow(page).click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Unpin' }).click()
  await expect.poll(() => rowOrder(page).then(rows => rows.at(-1))).toBe('group')
  await expect.poll(async () => (await storedRoom(page)).pinned).toBe(false)

  // File the room into a brand-new section from its own menu.
  await groupRow(page).click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Move to section' }).hover()
  await expect(page.getByRole('menuitem', { name: 'New section…' })).toBeVisible()
  await capture(page, '4-group-row-move-to-section')
  await page.getByRole('menuitem', { name: 'New section…' }).click()
  const nameField = page.getByRole('textbox', { name: 'Section name' })
  await nameField.fill('Rooms')
  await page.getByRole('button', { name: 'Create' }).click()

  await expect(sectionBlock(page, 'Rooms').getByRole('button', { name: new RegExp(`^${ROOM}, `) })).toBeVisible()
  await expect(sectionBlock(page, 'Unassigned').getByRole('button', { name: new RegExp(`^${ROOM}, `) })).toHaveCount(0)
  await expect.poll(async () => (await storedRoom(page)).sectionId).toMatch(/^sec-/)
  await capture(page, '5-room-filed-into-rooms')

  // Remove from section returns it to the unassigned bucket.
  await groupRow(page).click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Move to section' }).hover()
  await page.getByRole('menuitem', { name: 'Remove from section' }).click()
  await expect(sectionBlock(page, 'Unassigned').getByRole('button', { name: new RegExp(`^${ROOM}, `) })).toBeVisible()
  await expect.poll(async () => (await storedRoom(page)).sectionId).toBeNull()
  await capture(page, '6-room-unfiled')
})
