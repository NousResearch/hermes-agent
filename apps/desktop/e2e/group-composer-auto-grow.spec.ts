import fs from 'node:fs'
import path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// The group-room composer starts as one compact row, grows with wrapped text
// and newlines so a long brief can be read back while drafting, and stops at
// min(50vh, 24rem) — past that the textarea scrolls internally so the room
// transcript keeps its space (hermes-agent#95300).

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

// GROUP_COMPOSER_SCREENSHOT_DIR=<dir> saves full-window captures at the key
// states — PR evidence; never part of the assertions.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.GROUP_COMPOSER_SCREENSHOT_DIR

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

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-group-composer')
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

test('the group composer grows with a long prompt and stops at the viewport cap', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
  await expect(page.locator('[data-roster-key="local::alpha"]')).toBeVisible({ timeout: 30_000 })

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()
  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const name of ['Alpha @alpha', 'Beta @beta']) {
    await dialog.getByRole('checkbox', { name }).click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill(ROOM)
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })

  const composer = page.getByRole('textbox', { name: `Message ${ROOM}` }).filter({ visible: true })
  await expect(composer).toBeVisible({ timeout: 30_000 })

  const metrics = () =>
    composer.evaluate(el => {
      const box = el.getBoundingClientRect()

      return { height: Math.round(box.height), scrollHeight: el.scrollHeight, clientHeight: el.clientHeight }
    })

  const empty = await metrics()
  await capture(page, '1-composer-empty')

  // Eight explicit lines: the box must grow well past the single row.
  const brief = Array.from({ length: 8 }, (_, i) => `Brief line ${i + 1}: agree on the release checklist item`).join('\n')
  await composer.fill(brief)
  const grown = await metrics()
  await capture(page, '2-composer-eight-lines')
  expect(grown.height).toBeGreaterThan(empty.height * 3)
  // Content fits: nothing to scroll yet.
  expect(grown.scrollHeight).toBeLessThanOrEqual(grown.clientHeight + 1)

  // Far more than fits: growth stops at min(50vh, 24rem) and the text
  // scrolls inside the box instead of taking over the room.
  await composer.fill(Array.from({ length: 80 }, (_, i) => `Line ${i + 1}`).join('\n'))
  const capped = await metrics()
  await capture(page, '3-composer-capped-scrolls')
  const viewport = page.viewportSize() ?? (await page.evaluate(() => ({ height: window.innerHeight, width: 0 })))
  const cap = Math.min(viewport.height / 2, 24 * 16)
  expect(capped.height).toBeLessThanOrEqual(cap + 1)
  expect(capped.height).toBeGreaterThanOrEqual(grown.height)
  expect(capped.scrollHeight).toBeGreaterThan(capped.clientHeight + 40)

  // Enter still sends (not a newline): the composer empties back to one row.
  await composer.fill('@alpha hello there')
  await composer.press('Enter')
  await expect.poll(async () => (await metrics()).height).toBe(empty.height)
})
