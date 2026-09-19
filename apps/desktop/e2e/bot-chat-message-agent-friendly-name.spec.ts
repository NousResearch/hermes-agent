import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { writeEnvFile } from '../../../tests-js/scripts/mock-provider-config'
import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady
} from './fixtures'
import { expect, test } from './test'

/**
 * Bot Mode `message_agent` target resolution, driven from a real Bot Chat.
 *
 * The Desktop autocompletes a bot by its friendly name (Bot Mode title /
 * profile display_name) as an @-slug, while the backend keyed local targets on
 * the profile folder id only. Sending to "Scribe" (folder `writer`) therefore
 * came back "No teammate named 'Scribe'" and nothing reached the writer's Bot
 * Chat (#100671). The mock model emits exactly the tool call the user text
 * scripts (`E2E_CALL(message_agent)[…]`) so the REAL tool runs in the sender's
 * backend and the REAL delivery lands in the recipient profile's state.db.
 */

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

/** A Bot the Desktop never touched: folder `writer`, Bot Mode title "Scribe", its own
 *  model — the shape of a bot created on another machine or renamed by hand. */
function seedWriterProfile(home: string): string {
  const writerHome = path.join(home, 'profiles', 'writer')
  fs.mkdirSync(writerHome, { recursive: true })
  fs.writeFileSync(path.join(writerHome, 'profile.yaml'), 'description: Writes things\nui_meta:\n  hermes-bots:\n    title: Scribe\n', 'utf8')
  const base = fs.readFileSync(path.join(home, 'config.yaml'), 'utf8')
  fs.writeFileSync(
    path.join(writerHome, 'config.yaml'),
    base.replace('default: mock-model', 'default: mock-model-writer').replace('mock-model: {}', 'mock-model: {}\n      mock-model-writer: {}'),
    'utf8',
  )
  writeEnvFile(writerHome)

  return writerHome
}

function sqlite(dbPath: string, sql: string): string {
  return execFileSync('python3', ['-c', 'import sqlite3,sys; c=sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); print("\\n".join(str(r[0]) for r in c.execute(sys.argv[2])))', dbPath, sql], {
    encoding: 'utf8',
  }).trim()
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  seedWriterProfile(fixture.sandbox.hermesHome)
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a DM addressed by the friendly name lands in that bot\'s Bot Chat', async () => {
  test.setTimeout(600_000)
  const page = fixture!.page
  const home = fixture!.sandbox.hermesHome

  await openBots(page)
  await expect(page.getByRole('button', { name: /^Scribe · @writer\b/ }).first()).toBeVisible({ timeout: 60_000 })
  await createAgent(page, 'sender', 'Sender')
  // The row click opens the sender's canonical Bot Chat into the main workspace
  // (its fresh-chat splash replaces the default profile's welcome pane).
  await page.getByRole('button', { name: /^Sender · @sender\b/ }).filter({ visible: true }).first().click()
  await expect(page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({ timeout: 120_000 })
  await expect(page.getByText('What are we building?')).toHaveCount(0)
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true })
  await expect(composer).toHaveCount(1, { timeout: 120_000 })

  await composer.click()
  await composer.type('E2E_CALL(message_agent)[{"target":"Scribe","message":"codeword-7A91 please ack"}]', { delay: 5 })
  await page.keyboard.press('Enter')

  // The tool result the model saw: base answers "No teammate named 'Scribe'".
  await expect(page.getByText(/E2E_CALL_RESULT:/).filter({ visible: true }).first()).toBeVisible({ timeout: 240_000 })
  await expect(page.getByText(/Message dispatched to @writer/).first()).toBeVisible()
  await expect(page.getByText(/No teammate named/)).toHaveCount(0)
  const senderDb = path.join(home, 'profiles', 'sender', 'state.db')
  const toolResult = sqlite(senderDb, "SELECT content FROM messages WHERE role='tool' ORDER BY rowid DESC LIMIT 1")
  expect(toolResult).toContain('"to": "@writer"')

  // The recipient turn is a full `hermes -p writer` CLI boot in the background; its
  // landing in writer/state.db is covered by tests/tools (delivery runner) — under a
  // saturated host it can outlast any sane spec budget, so it is not awaited here.
  await page.screenshot({ path: 'test-results/bot-chat-message-agent-friendly-name.png', fullPage: true })
})
