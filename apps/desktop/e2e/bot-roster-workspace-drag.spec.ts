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
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// Dragging a bot from the Bots roster onto the workspace commits like a
// SESSION drag — the shared pointer-drag machinery (`startBotChatDrag` over
// `drag-session.ts`): ghost chip, split bands, strip caret, composer link,
// Esc. One press drives BOTH drags the row runs (the workspace pointer drag
// and the native section drag on the avatar), each declining outside its own
// region. The tile must land in the BOTS workspace with the bot's name on
// the tab.

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

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

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function botRow(page: Page) {
  const row = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  await expect(row).toBeVisible({ timeout: 30_000 })

  return row
}

/** Press the row's center, drag to (x, y), release — one real pointer drag. */
async function dragRowTo(page: Page, row: { boundingBox(): Promise<null | { height: number; width: number; x: number; y: number }> }, x: number, y: number): Promise<void> {
  const box = await row.boundingBox()

  if (!box) {
    throw new Error('bot row has no bounding box')
  }

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(x, y, { steps: 14 })
  await page.mouse.up()
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-roster-drag')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  await seedBot(sandbox.hermesHome, mock.url, 'alpha')

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

test('dragging a roster bot to the workspace edge splits the bot chat into it', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)
  const row = await botRow(page)

  // A plain click still opens the chat — the pointer drag rides the same
  // press, so a sub-threshold press must keep minting the canonical chat.
  await row.click()
  await expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true }).first()).toBeVisible({
    timeout: 45_000
  })

  // A second tab gives the main zone a strip (a one-tab strip is hidden) —
  // also the surface the strip-caret drop targets.
  await page.keyboard.press('Control+t')
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })

  const paneBox = await page
    .locator('[data-session-anchor="workspace"]')
    .filter({ visible: true })
    .first()
    .boundingBox()

  if (!paneBox) {
    throw new Error('workspace pane has no bounding box')
  }

  // Drop on the zone's LEFT EDGE: the split band.
  await dragRowTo(page, row, paneBox.x + 30, paneBox.y + paneBox.height / 2)

  const captions = () =>
    page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>('[data-zone-tabstrip="grp-main"] [data-tree-tab]')]
        .map(element => (element.textContent ?? '').trim())
        .filter(Boolean)
    )

  await expect.poll(captions, { timeout: 20_000 }).some(caption => /alpha/i.test(caption))
})

test('dragging a roster bot over a chat composer drops an @session link chip', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)
  const row = await botRow(page)

  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('mention check')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })

  const composerBox = await composer.boundingBox()

  if (!composerBox) {
    throw new Error('composer has no bounding box')
  }

  await dragRowTo(page, row, composerBox.x + composerBox.width / 2, composerBox.y + composerBox.height / 2)

  // The composer gains an @session chip for the bot's chat (inserted ref).
  await expect(page.locator('[data-inline-ref], [data-session-ref]').filter({ visible: true }).first()).toBeVisible({
    timeout: 20_000
  })
})
