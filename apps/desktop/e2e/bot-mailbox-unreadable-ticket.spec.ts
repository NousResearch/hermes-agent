import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'

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

// A teammate's `message_agent` DM into a Bot Chat the Desktop holds open travels through the
// live-owner mailbox (`<profile>/runtime/bot_live_delivery/<id>.json`): the sender admits a
// ticket, the gateway's idle poller claims it and runs the turn. Both sides bulk-scan the
// directory, so ONE unreadable ticket (ACL drift, elevated-owner file, AV lock) used to make
// every sender report `ambiguous` and crash the receiver's poll each cycle — the healthy DM
// never rendered in the open chat. Regression: the DM must still arrive.

type Page = MockBackendFixture['page']

let fixture: MockBackendFixture | null = null

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..')
const PYTHON = process.env.HERMES_E2E_PYTHON ?? path.join(REPO_ROOT, '.venv', 'bin', 'python')

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
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

/** Plant one chmod-000 ticket, then admit a real DM through the production mailbox API. */
function admitDmBesideUnreadableTicket(profileHome: string, message: string): string {
  const script = `
import json, sys
from pathlib import Path
from tools import bot_live_delivery as mailbox
home = Path(sys.argv[1]); message = sys.argv[2]
owner = mailbox.find_canonical_live_owner(home)
if owner is None:
    print(json.dumps({"error": "no live owner advertised for " + str(home)})); raise SystemExit(0)
root = home / "runtime" / mailbox.DELIVERY_DIR_NAME
root.mkdir(parents=True, exist_ok=True)
bad = root / ("e" * 32 + ".json")
bad.write_text('{"status": "queued"}', encoding="utf-8")
bad.chmod(0)
try:
    record = mailbox.deliver_to_live_owner(home, owner, message, author={"id": "bot:beta", "name": "beta", "is_bot": True})
    print(json.dumps({"status": record["status"], "delivery_id": record["delivery_id"]}))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
`

  return execFileSync(PYTHON, ['-c', script, profileHome, message], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: REPO_ROOT, HERMES_HOME: profileHome },
    encoding: 'utf8'
  }).trim()
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-mailbox-unreadable')
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

test('a teammate DM still reaches the open Bot Chat when an unreadable ticket sits in the mailbox', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const profileHome = path.join(fixture!.sandbox.hermesHome, 'profiles', 'alpha')

  await openBots(page)
  const alphaRow = page.getByRole('button', { name: /^alpha\b/i }).filter({ visible: true }).first()
  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await alphaRow.click()
  await expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true }).first()).toBeVisible({
    timeout: 60_000
  })

  // One real Desktop turn first: the lease (and its live-consumer advertisement) is acquired on submit.
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 15_000 })
  await composer.click()
  await composer.fill('warm up alpha')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })

  const dm = 'Message from 🤖 beta (@beta): PROBE-MAILBOX-DM please ack'
  const admission = JSON.parse(admitDmBesideUnreadableTicket(profileHome, dm)) as { status?: string; error?: string }
  await test.info().attach('admission', { body: JSON.stringify(admission), contentType: 'application/json' })
  expect(admission.error, 'sender admission must not degrade to ambiguous').toBeUndefined()
  expect(admission.status).toBe('queued')

  // The gateway poller claims the ticket at the idle boundary and runs it as a real inbound turn;
  // the transcript renders the teammate message as a collapsed inter-agent directive card, so
  // assert on presence in the transcript DOM rather than on visibility of the folded text.
  await expect(page.getByText(/PROBE-MAILBOX-DM/).first()).toBeAttached({ timeout: 90_000 })
  await test.info().attach('bot-chat-after-dm', { body: await page.screenshot(), contentType: 'image/png' })
})
