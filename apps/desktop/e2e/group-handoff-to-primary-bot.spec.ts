import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// #100406: in a Bot Mode group room a teammate's `@hermes` handoff must give
// the primary profile (internal name `default`) its turn, exactly like
// `@hermes → @code-farmer` already does. The live roster stamps the primary
// row's handle as the bare profile id ("default"), and the mention parser let
// that stamped handle shadow the `@hermes` alias — so the room settled with
// Hermes never driven. The mock inference server scripts each member's line
// from the user's send (`E2E_SAY(<handle>)[…]`), so the assertion is on the
// persisted room log: an entry authored by `default` saying "B".

let fixture: MockBackendFixture | null = null

type Page = MockBackendFixture['page']

interface RoomLogEntry {
  from?: { kind?: string; name?: string }
  text?: string
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

/** The plugin's persisted room log (`hermes.plugin.hermes-bots.group-chats`). */
async function roomLog(page: Page, group: string): Promise<RoomLogEntry[]> {
  return page.evaluate(name => {
    const raw = window.localStorage.getItem('hermes.plugin.hermes-bots.group-chats')
    const rooms = raw ? (JSON.parse(raw) as Record<string, { log?: RoomLogEntry[] }>) : {}

    return rooms[name]?.log ?? []
  }, group)
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a teammate handing off with @hermes drives the primary profile', async () => {
  test.setTimeout(420_000)
  const page = fixture!.page
  const group = 'Hermes, Code Farmer'

  await openBots(page)
  await createAgent(page, 'code-farmer', 'Code Farmer')

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const title of ['Hermes', 'Code Farmer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill(group)
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const composer = page.getByRole('textbox', { name: `Message ${group}` }).filter({ visible: true })
  await expect(composer).toBeVisible({ timeout: 20_000 })

  // Only Code Farmer is addressed by the user. Its scripted reply hands off
  // to @hermes; Hermes' scripted reply is "B". Neither script token carries
  // a literal `@`, so the user send itself never mentions Hermes.
  await composer.fill(
    '@code-farmer Please reply with one line only. ' +
      'E2E_SAY(code-farmer)[{at}hermes Please reply with the letter B.] E2E_SAY(hermes)[B]'
  )
  await composer.press('Enter')

  // Code Farmer's handoff line lands first (the reverse direction is not in
  // question); then the room must NOT settle without Hermes' turn.
  await expect
    .poll(async () => (await roomLog(page, group)).some(e => e.from?.name === 'code-farmer' && /@hermes/.test(e.text || '')), {
      timeout: 180_000
    })
    .toBe(true)

  await expect
    .poll(async () => (await roomLog(page, group)).some(e => e.from?.name === 'default' && (e.text || '').trim() === 'B'), {
      timeout: 180_000,
      message: 'the primary profile (default / @hermes) never took its turn after being @mentioned by a teammate'
    })
    .toBe(true)

  // And the transcript shows the handoff answered.
  await expect(page.getByText('B', { exact: true }).filter({ visible: true }).first()).toBeVisible()
})
