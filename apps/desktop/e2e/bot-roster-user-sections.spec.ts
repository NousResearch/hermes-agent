import fs from 'node:fs'
import path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// User-made sections in the Bots roster: a bot is filed by dragging it onto a
// section or through its row menu, the section is renamed through the same
// dialog shape sessions use, and deleting a section returns its bots to
// Unassigned (with an Undo toast, no confirmation). With no sections created
// the roster is the plain list it always was.

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

// BOT_SECTIONS_SCREENSHOT_DIR=<dir> saves full-window captures at the key
// states — handy for design review; never part of the assertions.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.BOT_SECTIONS_SCREENSHOT_DIR

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

const roster = (page: Page) => page.locator('[data-slot="bots-roster"]')
const botRow = (page: Page, name: string) => roster(page).locator(`[data-roster-key="local::${name}"]`)

/** A section's label span — the one node whose text is exactly the name. */
const sectionLabel = (page: Page, name: string) =>
  page.locator('span.truncate', { hasText: new RegExp(`^${name}$`, 'i') })

/** The heading's fold button (label + count) — the ⋯ menu trigger is a sibling with no text. */
const sectionHeading = (page: Page, name: string) =>
  roster(page).locator('[data-slot="bots-section"] button[aria-expanded]').filter({ has: sectionLabel(page, name) })

const sectionBlock = (page: Page, name: string) =>
  roster(page).locator('[data-slot="bots-section"]').filter({ has: sectionLabel(page, name) })

/** Section name → roster keys of the rows under it (the plain list has no sections). */
async function layout(page: Page): Promise<Array<[string, string[]]>> {
  return roster(page).locator('[data-slot="bots-section"]').evaluateAll(blocks =>
    blocks.map(block => [
      block.querySelector('button[aria-expanded] span.truncate')?.textContent?.trim() ?? '',
      [...block.querySelectorAll<HTMLElement>('[data-roster-key]')].map(row => row.dataset.rosterKey ?? '')
    ])
  )
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-sections')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  for (const name of ['alpha', 'beta', 'gamma']) {
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

test('file bots into user sections by menu and drag; rename; delete returns them to Unassigned', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
  await expect(botRow(page, 'alpha')).toBeVisible({ timeout: 30_000 })
  await expect(botRow(page, 'beta')).toBeVisible({ timeout: 30_000 })

  // No sections yet: the plain list, no section chrome at all.
  await expect(roster(page).locator('[data-slot="bots-section"]')).toHaveCount(0)
  await capture(page, '1-plain-roster')

  // Right-click alpha → Move to section → New section… → name it → alpha is filed.
  await botRow(page, 'alpha').click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Move to section' }).hover()
  await expect(page.getByRole('menuitem', { name: 'New section…' })).toBeVisible()
  await capture(page, '2-row-menu-move-to-section')
  await page.getByRole('menuitem', { name: 'New section…' }).click()
  const nameField = page.getByRole('textbox', { name: 'Section name' })
  await expect(nameField).toBeVisible()
  await nameField.fill('Clients')
  await capture(page, '3-new-section-dialog')
  await page.getByRole('button', { name: 'Create' }).click()

  await expect(sectionHeading(page, 'Clients')).toBeVisible()
  await expect(sectionBlock(page, 'Clients').locator('[data-roster-key="local::alpha"]')).toBeVisible()
  // The remainder is Unassigned, drawn last.
  await expect
    .poll(async () => (await layout(page)).map(([name, keys]) => [name, keys.length]))
    .toEqual([
      ['Clients', 1],
      ['Unassigned', 3]
    ])
  await capture(page, '4-alpha-filed')

  // Drag beta over the Clients block: the target highlights while over it.
  // Escape cancels — nothing moves, nothing stays highlighted or faded.
  const target = sectionBlock(page, 'Clients')
  const from = (await botRow(page, 'beta').boundingBox())!
  const to = (await sectionHeading(page, 'Clients').boundingBox())!

  const dragBetaOverClients = async () => {
    await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2)
    await page.mouse.down()
    await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2 - 10, { steps: 4 })
    await page.mouse.move(to.x + to.width / 2, to.y + to.height / 2, { steps: 12 })
    await expect(target).toHaveAttribute('data-drop-over', 'true')
  }

  await dragBetaOverClients()
  await page.keyboard.press('Escape')
  await page.mouse.up()
  await expect(target).not.toHaveAttribute('data-drop-over', 'true')
  await expect(botRow(page, 'beta')).toHaveCSS('opacity', '1')
  expect((await layout(page)).map(([name, keys]) => [name, keys.length])).toEqual([
    ['Clients', 1],
    ['Unassigned', 3]
  ])

  // Drop it for real: the bot is filed.
  await dragBetaOverClients()
  await capture(page, '5-drag-over-clients')
  await page.mouse.up()

  await expect(target.locator('[data-roster-key="local::beta"]')).toBeVisible()
  await expect(target).not.toHaveAttribute('data-drop-over', 'true')
  // The moved row remounts under its new section; it must not stay faded.
  await expect(botRow(page, 'beta')).toHaveCSS('opacity', '1')
  await expect
    .poll(async () => (await layout(page)).map(([name, keys]) => [name, keys.length]))
    .toEqual([
      ['Clients', 2],
      ['Unassigned', 2]
    ])
  await capture(page, '6-beta-dropped')

  // Rename through the heading's context menu — the same Dialog + Input
  // + Save shape as a session rename.
  await sectionHeading(page, 'Clients').click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Rename…' }).click()
  await expect(nameField).toHaveValue('Clients')
  await nameField.fill('Customers')
  await page.getByRole('button', { name: 'Save' }).click()
  await expect(sectionHeading(page, 'Customers')).toBeVisible()
  await expect(sectionHeading(page, 'Clients')).toHaveCount(0)
  await capture(page, '7-renamed')

  // A second, empty section from the + menu shows its drop hint; collapsing
  // a section folds its rows like the gateway headings do.
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New section' }).click()
  await nameField.fill('Team')
  await page.getByRole('button', { name: 'Create' }).click()
  await expect(sectionBlock(page, 'Team').getByText('Drag bots here')).toBeVisible()
  await sectionHeading(page, 'Customers').click()
  await expect(sectionBlock(page, 'Customers').locator('[data-roster-key]')).toHaveCount(0)
  await capture(page, '8-empty-section-and-collapsed')
  await sectionHeading(page, 'Customers').click()
  await expect(sectionBlock(page, 'Customers').locator('[data-roster-key]')).toHaveCount(2)

  // Delete Customers: no confirmation, its two bots return to Unassigned,
  // and the toast offers Undo.
  await sectionHeading(page, 'Customers').click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Delete' }).click()
  await expect(sectionHeading(page, 'Customers')).toHaveCount(0)
  const toast = page.getByRole('status').filter({ hasText: 'Deleted “Customers”' })
  await expect(toast).toBeVisible()
  await expect
    .poll(async () => (await layout(page)).map(([name, keys]) => [name, keys.length]))
    .toEqual([
      ['Team', 0],
      ['Unassigned', 4]
    ])
  await capture(page, '9-deleted-with-undo-toast')

  await toast.getByRole('button', { name: 'Undo' }).click()
  await expect(sectionHeading(page, 'Customers')).toBeVisible()
  await expect
    .poll(async () => (await layout(page)).map(([name, keys]) => [name, keys.length]))
    .toEqual([
      ['Customers', 2],
      ['Team', 0],
      ['Unassigned', 2]
    ])

  // Membership rides the bot's profile ui_meta, so it follows profile sync.
  const alphaProfile = path.join(fixture!.sandbox.hermesHome, 'profiles', 'alpha', 'profile.yaml')
  await expect.poll(() => (fs.existsSync(alphaProfile) ? fs.readFileSync(alphaProfile, 'utf8') : '')).toMatch(/sectionId:\s*sec-/)

  // Delete both sections: the roster is the plain list again.
  for (const name of ['Customers', 'Team']) {
    await sectionHeading(page, name).click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Delete' }).click()
  }

  await expect(roster(page).locator('[data-slot="bots-section"]')).toHaveCount(0)
  await expect(botRow(page, 'alpha')).toBeVisible()
})
