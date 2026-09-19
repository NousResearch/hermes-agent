import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { expect, test } from './test'

// The primary profile (`default`) renamed to a Bot Mode title ("Bobby") is
// addressed by the roster, autocomplete and mention routing as @bobby — but the
// group-turn prompt introduced it to itself as @hermes, so a member told
// "You are @hermes" read `@bobby …` as a message for someone else and passed.
// The mock inference server scripts a member's line by the `You are @<handle>`
// opener, so the assertion is exactly that mismatch: with a script for
// `bobby`, the renamed primary must answer (the persisted room log carries an
// entry authored by `default`), not settle silently.

let fixture: MockBackendFixture | null = null

type Page = MockBackendFixture['page']

interface RoomLogEntry {
  from?: { kind?: string; name?: string }
  text?: string
}

const GROUP = 'Bobby, Code Farmer'

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

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

async function roomLog(page: Page, group: string): Promise<RoomLogEntry[]> {
  return page.evaluate(name => {
    const raw = window.localStorage.getItem('hermes.plugin.hermes-bots.group-chats')
    const rooms = raw ? (JSON.parse(raw) as Record<string, { log?: RoomLogEntry[] }>) : {}

    return rooms[name]?.log ?? []
  }, group)
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('renamed-primary')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  // The primary profile carries a Bot Mode title — what `hermes profile rename`
  // / the Edit Profile dialog persist into profile.yaml's ui_meta.
  writeFileSync(
    join(sandbox.hermesHome, 'profile.yaml'),
    'ui_meta:\n  hermes-bots:\n    title: Bobby\n    shape: circle\n    color: "#4f8"\n',
    'utf8'
  )

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

// eslint-disable-next-line no-empty-pattern
test.afterEach(async ({}, info) => {
  if (!fixture) {
    return
  }

  await info.attach('room-log', {
    body: JSON.stringify(await roomLog(fixture.page, GROUP), null, 2),
    contentType: 'application/json'
  })
  await info.attach('native-window', { body: await fixture.page.screenshot(), contentType: 'image/png' })
  await info.attach('desktop-log', {
    body: readFileSync(join(fixture.sandbox.hermesHome, 'logs/desktop.log')),
    contentType: 'text/plain'
  })
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a renamed primary bot is addressed by its own @tag in the group-turn prompt', async () => {
  test.setTimeout(420_000)
  const page = fixture!.page

  await openBots(page)
  await expect(page.getByRole('button', { name: /^Bobby\b/ }).first()).toBeVisible({ timeout: 30_000 })
  await createAgent(page, 'code-farmer', 'Code Farmer')

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const title of ['Bobby', 'Code Farmer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill(GROUP)
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const composer = page.getByRole('textbox', { name: `Message ${GROUP}` }).filter({ visible: true })
  await expect(composer).toBeVisible({ timeout: 20_000 })

  // Only the renamed primary is addressed, by its friendly tag. Its scripted
  // line is keyed to `bobby`: the member only speaks when its turn prompt
  // introduces it as @bobby (the control, code-farmer, is scripted too so a
  // wrong-handle prompt cannot be mistaken for an idle room).
  await composer.fill('@bobby Please reply with one line only. E2E_SAY(bobby)[B] E2E_SAY(code-farmer)[C]')
  await composer.press('Enter')

  await expect
    .poll(async () => (await roomLog(page, GROUP)).some(e => e.from?.name === 'default' && (e.text || '').trim() === 'B'), {
      timeout: 180_000,
      message: 'the renamed primary (default / Bobby) was introduced to itself as @hermes and never answered @bobby'
    })
    .toBe(true)

  await expect(page.getByText('B', { exact: true }).filter({ visible: true }).first()).toBeVisible()
})
