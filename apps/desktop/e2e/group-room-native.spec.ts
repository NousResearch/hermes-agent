/**
 * Live evidence rig for Bot Mode group rooms: drives a gateway-owned
 * (canonical) room and a legacy Desktop-driven room through working, settled
 * and failure states and writes full-window screenshots plus DOM dumps.
 *
 * Verdicts are PROBE_RESULT console lines and JSON files next to the PNGs,
 * not pass/fail: SHOT_DIR (default `<tmpdir>/group-room-native`) receives
 * everything. Mock scripting (tests-js/scripts/mock-server.ts):
 * E2E_SAY(<handle>)[line] replies, E2E_TOOL(<handle>)[terminal:cmd] runs a tool,
 * E2E_FAIL(<handle>) answers HTTP 401; unscripted members `(pass)`.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { type MockServerOptions, startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, waitForAppReady } from './fixtures'
import { allowErrorBanners, expect, test } from './test'

type Page = MockBackendFixture['page']

const SHOT_DIR = process.env.SHOT_DIR || path.join(os.tmpdir(), 'group-room-native')
const NL = String.fromCharCode(10)
const MULTIPLEX = ['gateway:', '  multiplex_profiles: true', ''].join(NL)
let fixture: MockBackendFixture | null = null

fs.mkdirSync(SHOT_DIR, { recursive: true })

function probe(label: string, data: unknown): void {
  console.log(`PROBE_RESULT ${label} ${JSON.stringify(data)}`)
}

/**
 * setupMockBackend whose .env carries the provider key. A routed profile reads keys only
 * from its own .env, and the stock fixture's .env lacks the provider's
 * `key_env` (OPENAI_API_KEY), so the gateway refuses every turn
 * ("No API key for custom provider 'Mock'") and the app parks on onboarding.
 */
async function setupKeyedBackend(mockServer: MockServerOptions, multiplex: boolean): Promise<MockBackendFixture> {
  const mock = await startMockServer(mockServer)
  const sandbox = createSandbox('mock')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, multiplex ? MULTIPLEX : undefined)
  writeEnvFile(sandbox.hermesHome, 'e2e-mock-key', mock.url)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  return {
    app, page, mock, mockUrl: mock.url, sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
}

/** Every shot in the maintainer's theme: dark, unless THEME=light. */
async function applyTheme(): Promise<void> {
  const scheme: 'dark' | 'light' = process.env.THEME === 'light' ? 'light' : 'dark'
  await fixture!.app.evaluate(({ nativeTheme }, mode) => { nativeTheme.themeSource = mode }, scheme)
  await fixture!.page.emulateMedia({ colorScheme: scheme })
  await fixture!.page.waitForTimeout(300)
}

async function sizeWindow(): Promise<void> {
  const bounds = await fixture!.app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map(win => {
    win.unmaximize()
    win.setBounds({ x: 0, y: 0, width: 1400, height: 900 })

    return win.getBounds()
  }))

  await fixture!.page.waitForTimeout(500)
  await applyTheme()
  probe('window', { bounds, viewport: await fixture!.page.evaluate(() => [window.innerWidth, window.innerHeight]) })
}

async function dumpState(page: Page, composerLabel: string) {
  return page.evaluate(label => {
    const box = document.querySelector(`[aria-label="${label}"]`)
    let region: Element | null = box

    while (region && !region.querySelector('header')) {
      region = region.parentElement
    }

    const visibleLog = [...document.querySelectorAll('[role=log]')].find(el => (el as HTMLElement).offsetParent !== null)

    return {
      alerts: [...document.querySelectorAll('[role=alert]')].map(el => (el.textContent || '').trim()).filter(Boolean),
      header: (region?.querySelector('header') as HTMLElement | null)?.innerText ?? null,
      log: (document.querySelector('[role=log]') as HTMLElement | null)?.innerText ?? null,
      visibleLog: (visibleLog as HTMLElement | undefined)?.innerText ?? null,
      region: ((region as HTMLElement | null)?.innerText ?? '').slice(0, 6000),
      tabs: [...document.querySelectorAll('[role=tab]')].map(el => `${(el.textContent || '').trim()}${el.getAttribute('aria-selected') === 'true' ? ' [selected]' : ''}`)
    }
  }, composerLabel)
}

async function capture(name: string, composerLabel: string, extra: Record<string, unknown> = {}): Promise<void> {
  const page = fixture!.page
  await page.evaluate(() => {
    for (const viewport of document.querySelectorAll('[role=log], [data-radix-scroll-area-viewport]')) {
      viewport.scrollTop = viewport.scrollHeight
    }
  })
  await page.screenshot({ path: path.join(SHOT_DIR, `${name}.png`) })
  const dump = { ...(await dumpState(page, composerLabel)), ...extra, prompts: fixture!.mock.receivedPrompts.length, held: fixture!.mock.heldCompletionCount() }
  fs.writeFileSync(path.join(SHOT_DIR, `${name}.json`), JSON.stringify(dump, null, 2))
  probe(name, { header: dump.header, log: dump.log, alerts: dump.alerts })
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
  await expect(dialog).toBeHidden({ timeout: 60_000 })
  await expect(page.getByRole('button', { name: new RegExp(`^${title}\\b`) }).first()).toBeVisible({ timeout: 60_000 })
}

async function createRoom(page: Page, bots: [string, string][], roomName: string): Promise<void> {
  await openBots(page)

  for (const [name, title] of bots) {
    await createAgent(page, name, title)
  }

  // A late bot birth can foreground its own tab; return to Bots before the dialog.
  await page.waitForTimeout(3000)
  await openBots(page)
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const [, title] of bots) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill(roomName)
  await dialog.getByRole('button', { name: `Create Group (${bots.length})` }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })
}

/** Send once (retrying only until the composer takes the text), then report whether inference saw it. */
async function send(page: Page, roomName: string, composerLabel: string, text: string): Promise<boolean> {
  const composer = page.getByRole('textbox', { name: composerLabel, exact: true }).filter({ visible: true })

  await expect(async () => {
    const tab = page.getByRole('tab', { name: new RegExp(`^${roomName}`) }).first()

    if (await tab.count()) {
      await tab.click()
    }

    await expect(composer).toBeEnabled({ timeout: 5000 })
    await composer.fill(text)
    await composer.press('Enter')
    await expect.poll(() => composer.evaluate(el => (el as HTMLTextAreaElement).value ?? el.textContent ?? ''), { timeout: 10_000 }).toBe('')
  }).toPass({ timeout: 150_000 })

  const delivered = await expect.poll(() => fixture!.mock.receivedPrompts.some(p => p.includes(text.slice(0, 24))), { timeout: 60_000 })
    .toBe(true).then(() => true, () => false)

  probe('send', { text: text.slice(0, 40), delivered })

  return delivered
}

/** The room's visible text stops changing and no working line remains. */
async function waitSettled(page: Page, composerLabel: string, timeoutMs = 90_000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  let last = ''
  let stableSince = Date.now()

  while (Date.now() < deadline) {
    const state = await dumpState(page, composerLabel)
    const now = `${state.region}|${fixture!.mock.receivedPrompts.length}`

    if (now !== last) {
      last = now
      stableSince = Date.now()
    } else if (Date.now() - stableSince > 8000 && !/working|thinking|running/i.test(state.visibleLog ?? state.region)) {
      return true
    }

    await page.waitForTimeout(1000)
  }

  return false
}

/** Reply in thread + its chip, the @ picker, the details panel and the Bots-list row. */
async function shootExtras(page: Page, room: string, label: string): Promise<void> {
  const composer = page.getByRole('textbox', { name: label, exact: true }).filter({ visible: true })
  const message = page.locator('[data-room-message="member"]').filter({ visible: true }).last()
  await message.hover()
  const reply = message.getByRole('button', { name: /^Reply/ }).first()

  if (await reply.count()) {
    await reply.click()
    await page.waitForTimeout(500)
    await capture('07-reply-chip', label, { replyChip: await page.locator('[data-room-reply]').count() })
    await page.locator('[data-room-reply]').getByRole('button').first().click().catch(() => undefined)
    await composer.fill('')
  } else {
    probe('07-reply-chip', { skipped: 'no Reply action on the last member message' })
  }

  await composer.click()
  await composer.pressSequentially('@')
  await page.waitForTimeout(800)
  await capture('08-mention-picker', label, { listbox: await page.getByRole('listbox').count() })
  await page.keyboard.press('Escape')
  await composer.fill('')

  const title = page.getByRole('button', { name: /details/i }).filter({ visible: true }).first()

  if (await title.count()) {
    await title.click()
    await page.waitForTimeout(800)
    await capture('09-details-panel', label, { details: true })
    await title.click().catch(() => undefined)
  } else {
    probe('09-details-panel', { skipped: 'no details trigger' })
  }

}

/** The Bots list while the room works: its row with member avatars and the working glyph. */
async function shootBotsRow(page: Page, room: string, label: string): Promise<void> {
  await openBots(page)
  const refresh = page.getByRole('button', { name: /Refresh gateway groups/i }).first()

  if (await refresh.count()) {await refresh.click().catch(() => undefined)}
  await page.waitForTimeout(1500)
  await capture('10-bots-row', label, { rowState: await page.locator('[data-room-state]').count(), row: await page.getByRole('button', { name: room, exact: true }).count() })
  const tab = page.getByRole('tab', { name: new RegExp(`^${room}`) }).first()

  if (await tab.count()) {await tab.click()}
}

test.afterEach(async () => {
  // Backend receipts (install stamp, member turns) die with the sandbox; keep a copy.
  const logs = fixture && path.join(fixture.sandbox.hermesHome, 'logs')

  if (logs && fs.existsSync(logs)) {
    fs.cpSync(logs, path.join(SHOT_DIR, `logs-${test.info().title.split(':')[0].replace(/[^a-z0-9]+/gi, '-')}`), { recursive: true })
  }

  await fixture?.cleanup()
  fixture = null
})

test('gateway-owned room: working, settled, failure', async () => {
  test.setTimeout(600_000)
  allowErrorBanners()
  const sentinel = path.join(os.tmpdir(), `grn-release-${process.pid}-${Date.now()}`)
  // The scripts ride hidden keys (mock `groupScripts`) so the transcript reads like a real chat.
  const kickoff = 'Can the team get the release branch green? @builder scaffold it, @scout check the build logs.'
  const second = 'Thanks all. One more pass on the changelog before we ship?'

  fixture = await setupKeyedBackend({
    holdFirstCompletionContaining: 'SCOUT_42',
    groupScripts: {
      'release branch green': [
        'E2E_SAY(builder)[Scaffold is up on `release/0.23`. {at}scout the CI run is yours.]',
        `E2E_TOOL(scout)[terminal:echo SCOUT_$((40+2)); until test -f ${sentinel}; do sleep 0.5; done]`,
        'E2E_SAY(scout)[Build logs are clean: **312 passed**, 0 failed. Ready to tag.]'
      ].join(' '),
      'changelog before we ship': 'E2E_FAIL(critic) E2E_SAY(builder)[Changelog reads well. Nothing more from me.]'
    }
  }, true)
  await waitForAppReady(fixture, 120_000)
  await sizeWindow()
  const page = fixture.page
  const room = 'Builder, Critic, Scout'
  const label = 'Group message'

  await createRoom(page, [['builder', 'Builder'], ['critic', 'Critic'], ['scout', 'Scout']], room)
  await expect(page.getByRole('textbox', { name: label, exact: true }).filter({ visible: true })).toBeVisible({ timeout: 60_000 })
  await sizeWindow()

  // SCOUT_42 only exists in Scout's tool OUTPUT, so the hold lands on Scout's post-tool completion.
  await send(page, room, label, kickoff)

  // Tool still running (sentinel absent): the 2 s poll should show tool + elapsed.
  await expect.poll(() => fixture!.mock.receivedPrompts.filter(p => p.includes('You are @scout')).length, { timeout: 90_000 }).toBeGreaterThan(0)
  await page.waitForTimeout(6000)
  await capture('01-working', label, { phase: 'scout terminal tool running' })
  await shootBotsRow(page, room, label)

  fs.writeFileSync(sentinel, 'go')
  const held = await expect.poll(() => fixture!.mock.heldCompletionCount(), { timeout: 60_000 }).toBe(1).then(() => true, () => false)
  await page.waitForTimeout(5000)
  await capture('01b-working-held', label, { phase: 'scout post-tool completion held', heldReached: held })

  fixture.mock.releaseHeldStream()
  const settled = await waitSettled(page, label)
  await capture('02-settled', label, { settled })

  await shootExtras(page, room, label)

  await send(page, room, label, second)
  const failSettled = await waitSettled(page, label, 180_000)
  await capture('03-failure', label, { settled: failSettled })

  const menu = page.getByRole('button', { name: /more|actions|options/i }).filter({ visible: true }).first()

  if (await menu.count()) {
    await menu.click().catch(() => undefined)
    await page.waitForTimeout(1000)
    await capture('04-header-menu', label, { menu: true })
    await page.keyboard.press('Escape')
  } else {
    probe('04-header-menu', { skipped: 'no visible button named more/actions/options' })
  }

  fs.rmSync(sentinel, { force: true })
})

/** Each chip's member and status (`data-member-status`), and who the visible user messages are by. */
const roomFacts = () => fixture!.page.evaluate(() => ({
  chips: [...document.querySelectorAll('[data-member-status]')].filter(el => (el as HTMLElement).offsetParent !== null)
    .map(el => `${(el as HTMLElement).innerText.replace(/\s+/g, ' ').trim()} [${el.getAttribute('data-member-status')}]`),
  cycle: (document.querySelector('[data-room-cycle]') as HTMLElement | null)?.innerText ?? null,
  userAuthors: [...document.querySelectorAll('[data-room-message="user"]')].map(el => (el as HTMLElement).innerText.split('\n')[0])
}))

test('gateway-owned room: a failure in one thread stays while another thread works', async () => {
  test.setTimeout(600_000)
  allowErrorBanners()
  const sentinel = path.join(os.tmpdir(), `grn-nightly-${process.pid}-${Date.now()}`)
  const threadA = 'Quick check before standup: @critic review the plan, @builder confirm the branch.'
  const threadB = 'Separate topic: @scout how long does the nightly build take?'

  fixture = await setupKeyedBackend({
    groupScripts: {
      'before standup': 'E2E_FAIL(critic) E2E_SAY(builder)[Branch `release/0.23` is up to date with main.]',
      'nightly build take': [
        `E2E_TOOL(scout)[terminal:echo NIGHTLY; until test -f ${sentinel}; do sleep 0.5; done]`,
        'E2E_SAY(scout)[About 14 minutes end to end.]'
      ].join(' ')
    }
  }, true)
  await waitForAppReady(fixture, 120_000)
  await sizeWindow()
  const page = fixture.page
  const room = 'Builder, Critic, Scout'
  const label = 'Group message'

  await createRoom(page, [['builder', 'Builder'], ['critic', 'Critic'], ['scout', 'Scout']], room)
  await expect(page.getByRole('textbox', { name: label, exact: true }).filter({ visible: true })).toBeVisible({ timeout: 60_000 })
  await sizeWindow()

  await send(page, room, label, threadA)
  const aSettled = await waitSettled(page, label, 180_000)
  await capture('11-thread-a-failed', label, { settled: aSettled, ...(await roomFacts()) })

  // A plain Send starts thread B; Scout's tool holds it open while we look at Critic's chip.
  await send(page, room, label, threadB)
  await expect.poll(() => fixture!.mock.receivedPrompts.filter(p => p.includes('nightly build')).length, { timeout: 90_000 }).toBeGreaterThan(0)
  await expect.poll(async () => (await roomFacts()).chips.some(chip => /^scout/i.test(chip) && chip.endsWith('[working]')), { timeout: 60_000 }).toBe(true)
  await page.waitForTimeout(3000)
  const during = await roomFacts()
  await capture('12-thread-b-working-a-failed', label, { phase: 'thread B: scout in terminal; thread A: critic failed', ...during })
  probe('two-threads', during)

  fs.writeFileSync(sentinel, 'go')
  const bSettled = await waitSettled(page, label, 180_000)
  await capture('13-two-threads-settled', label, { settled: bSettled, ...(await roomFacts()) })
  fs.rmSync(sentinel, { force: true })
})

test('legacy Desktop-driven room: working and settled', async () => {
  test.setTimeout(400_000)
  allowErrorBanners()
  // group-turn-integrity's shape (no multiplex), plus the provider key: on trees where the
  // gateway always owns rooms, members otherwise fail with "No API key for custom provider".
  fixture = await setupKeyedBackend({
    holdFirstCompletionContaining: 'LEGACY_42',
    groupScripts: { 'review the retry patch': 'E2E_TOOL(programmer)[terminal:echo LEGACY_$((6*7))] E2E_SAY(programmer)[Patch is ready for review: retries now back off **exponentially**.]' }
  }, false)
  await waitForAppReady(fixture, 120_000)
  await sizeWindow()
  const page = fixture.page
  // Current gateways own every this-device room; answer the capability probe the way a
  // gateway without a room driver does so the Desktop-driven view opens (turns stay real).
  await page.evaluate(() => {
    const original = WebSocket.prototype.send

    WebSocket.prototype.send = function(data) {
      let frame: any = null

      try { frame = JSON.parse(String(data)) } catch { frame = null }

      if (frame?.method === 'groups.capabilities') {
        queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { driver: false, persistent_process: false } }) })))
      } else {
        original.call(this, data)
      }
    }
  })
  const room = 'Programmer, Reviewer'
  const legacyLabel = `Message ${room}`

  await createRoom(page, [['programmer', 'Programmer'], ['reviewer', 'Reviewer']], room)
  // Trees whose gateway always answers groups.capabilities with a driver build a
  // canonical room here even without multiplexing; record which one appeared.
  const legacy = page.getByRole('textbox', { name: legacyLabel, exact: true }).filter({ visible: true })
  const canonical = page.getByRole('textbox', { name: 'Group message', exact: true }).filter({ visible: true })
  await expect(legacy.or(canonical).first()).toBeVisible({ timeout: 60_000 })
  const roomKind = (await legacy.count()) ? 'legacy' : 'canonical'
  const label = roomKind === 'legacy' ? legacyLabel : 'Group message'
  probe('legacy-room-kind', { roomKind, label })
  await sizeWindow()

  await send(page, room, label, '@programmer can you review the retry patch before lunch?')

  const held = await expect.poll(() => fixture!.mock.heldCompletionCount(), { timeout: 90_000 }).toBe(1).then(() => true, () => false)
  await page.waitForTimeout(5000)
  await capture('06-legacy-working', label, { heldReached: held, roomKind })

  fixture.mock.releaseHeldStream()
  const settled = await waitSettled(page, label)
  await capture('05-legacy-settled', label, { settled, roomKind })
})
