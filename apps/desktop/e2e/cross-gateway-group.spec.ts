/**
 * E2E: every new Bot Mode group chat is created on and run by a gateway, including one that
 * seats a bot from a NAMED profile on a second computer's gateway.
 *
 * "This device" (owner A) is the Electron-managed local backend with `default` + `scribe`.
 * "Homelab" (lender B) is a REAL `hermes gateway run` on its own HOME lending its named
 * profile `coder` (catalog at `<room_link_url>/p/coder`), plus the `hermes serve` viewer
 * Desktop connects to. A plain `hermes serve` Homelab (no lender) must make the dialog say
 * what is missing instead of creating anything.
 *
 * Model turns are REAL when NOUS_API_KEY is in the runner's environment (each probe home gets
 * only that one line in its .env; the model is CROSS_GATEWAY_MODEL, default a cheap Nous
 * model) and fall back to the mock inference server otherwise. Real models do not follow
 * scripts, so replies are asserted by author and count. CROSS_GATEWAY_SHOTS=<dir> saves
 * screenshots; PROBE_RESULT lines carry the evidence. Needs `pnpm run build` (dist/) first.
 */

import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, type Sandbox, waitForAppReady } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const REMOTE_LABEL = 'Homelab'
const REMOTE_ID = 'homelab'
const REMOTE_TOKEN = 'e2e-cross-gateway-homelab-token'
const LEG = process.env.CROSS_GATEWAY_LEG || 'head'
const NL = String.fromCharCode(10)
const MULTIPLEX = ['gateway:', '  multiplex_profiles: true', ''].join(NL)
const REAL_KEY = process.env.NOUS_API_KEY || ''
const REAL_MODEL = process.env.CROSS_GATEWAY_MODEL || 'anthropic/claude-haiku-5.5'
const LENT_PROFILE = 'coder'

/** The model block every probe home uses: the Nous inference API (key from NOUS_API_KEY) or nothing (mock). */
const MODEL = REAL_KEY
  ? ['model:', `  default: ${REAL_MODEL}`, '  provider: custom', '  base_url: https://inference-api.nousresearch.com/v1',
    '  key_env: NOUS_API_KEY', '  context_length: 200000', ''].join(NL)
  : ''

/** Point a probe home at the turn's model: its config, and an .env holding only what it needs. */
function provision(home: string, mockUrl: string, extra = ''): void {
  writeMockProviderConfig(home, mockUrl, undefined, extra + MODEL)

  if (REAL_KEY) {
    fs.writeFileSync(path.join(home, '.env'), `NOUS_API_KEY=${REAL_KEY}${NL}`, { mode: 0o600 })
  } else {
    writeEnvFile(home, undefined, mockUrl)
  }
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

/** A real second gateway: its own HOME (the host singleton is per HOME) and
 *  HERMES_HOME, this tree's code, the runtime venv's interpreter. */
async function startRemoteGateway(root: string, mockUrl: string, config = MULTIPLEX) {
  const fakeHome = path.join(root, 'homelab-root')
  const home = path.join(fakeHome, '.hermes')
  fs.mkdirSync(home, { recursive: true })
  provision(home, mockUrl, config)
  const port = await freePort()
  const url = `http://127.0.0.1:${port}`
  const python = process.env.CROSS_GATEWAY_PYTHON || path.join(REPO_ROOT, '.venv', 'bin', 'python')

  const child: ChildProcess = spawn(python, ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', String(port), '--skip-build'], {
    cwd: REPO_ROOT,
    detached: true,
    env: { ...process.env, HOME: fakeHome, HERMES_HOME: home, PYTHONPATH: REPO_ROOT, HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  let log = ''
  child.stdout?.on('data', (chunk: Buffer) => { log += chunk.toString() })
  child.stderr?.on('data', (chunk: Buffer) => { log += chunk.toString() })
  const deadline = Date.now() + 90_000

  while (Date.now() < deadline) {
    if (child.exitCode !== null) {throw new Error(`remote hermes serve exited early (${child.exitCode}):\n${log}`)}

    try {
      if ((await fetch(`${url}/api/status`, { headers: { 'X-Hermes-Session-Token': REMOTE_TOKEN } })).ok) {break}
    } catch {
      // not up yet
    }

    await new Promise(resolve => setTimeout(resolve, 500))
  }

  console.log(`PROBE_RESULT remote serve pid=${child.pid} url=${url} home=${home}`)

  return {
    home,
    url,
    log: () => log,
    close: async () => {
      if (child.pid && child.exitCode === null) {
        try { process.kill(-child.pid, 'SIGTERM') } catch { child.kill('SIGTERM') }
      }

      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
}

function writeConnectionsRegistry(sandbox: Sandbox, remoteUrl: string): void {
  fs.writeFileSync(path.join(sandbox.userDataDir, 'connections.json'), JSON.stringify({
    version: 2, primary: 'local', launchMode: 'primary', lastUsed: 'local',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: REMOTE_ID, kind: 'remote', label: REMOTE_LABEL, url: remoteUrl, authMode: 'token', token: { encoding: 'plain', value: REMOTE_TOKEN } }
    ]
  }, null, 2), { encoding: 'utf8', mode: 0o600 })
}

async function shot(page: Page, name: string): Promise<void> {
  const dir = process.env.CROSS_GATEWAY_SHOTS

  if (dir) {
    fs.mkdirSync(dir, { recursive: true })
    const file = path.join(dir, `${LEG}-${name}.png`)
    await page.screenshot({ path: file })
    console.log(`PROBE_RESULT screenshot ${file}`)
  }
}

async function openBots(page: Page): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

/** Every picker row as { text, checkbox index }. */
async function pickerRows(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'New group chat' })

  return dialog.locator('label:has([role="checkbox"])').evaluateAll(nodes => nodes.map(node => (node as HTMLElement).innerText.replace(/\s+/g, ' ').trim()))
}

async function openGroupDialog(page: Page) {
  await openBots(page)
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  const menuItem = page.getByRole('menuitem', { name: 'New group chat' })
  await expect(menuItem).toBeEnabled({ timeout: 30_000 })
  await menuItem.click()
  const dialog = page.getByRole('dialog', { name: 'New group chat' })
  await expect(dialog).toBeVisible()
  // Let groups.capabilities settle so the picker shows its final roster.
  await page.waitForTimeout(3_000)

  return dialog
}

/** Tick the picker rows whose text matches each predicate, in order; returns their texts. */
async function pick(page: Page, ...wanted: Array<(text: string) => boolean>): Promise<string[]> {
  const dialog = page.getByRole('dialog', { name: 'New group chat' })
  const rows = await pickerRows(page)
  console.log('PROBE_RESULT picker rows', JSON.stringify(rows))
  const checkboxes = dialog.locator('label:has([role="checkbox"]) [role="checkbox"]')
  const chosen: string[] = []

  for (const want of wanted) {
    const index = rows.findIndex((text, i) => want(text) && !chosen.includes(rows[i]))
    expect(index, `picker row for ${want}`).toBeGreaterThanOrEqual(0)
    await checkboxes.nth(index).click()
    chosen.push(rows[index])
  }

  return chosen
}

const handleOf = (row: string) => /@([a-z0-9][a-z0-9._-]*)/i.exec(row)?.[1] ?? ''

const isRemote = (text: string) => text.includes(REMOTE_LABEL)

const isLocalDefault = (text: string) => !isRemote(text) && !/scribe/i.test(text)

const isLocalScribe = (text: string) => !isRemote(text) && /scribe/i.test(text)

const isRemoteCoder = (text: string) => isRemote(text) && new RegExp(LENT_PROFILE, 'i').test(text)

interface RoomEvent { kind: string; actor: Record<string, unknown>; payload: Record<string, unknown>; created_at: number }

/** The owner's authoritative room event log, read straight from its state.db (works with Electron closed). */
function roomEvents(home: string, name: string): RoomEvent[] {
  const query = spawnSync('python3', ['-c', [
    'import json, sqlite3, sys',
    'c = sqlite3.connect("file:" + sys.argv[1] + "/state.db?mode=ro", uri=True)',
    'rows = c.execute("select e.kind, e.actor_json, e.payload_json, e.created_at from hosted_room_events e join hosted_rooms r using(room_id) where r.name = ? order by e.seq", (sys.argv[2],))',
    'print(json.dumps([{"kind": k, "actor": json.loads(a), "payload": json.loads(p), "created_at": t} for k, a, p, t in rows]))'
  ].join(NL), home, name], { encoding: 'utf8' })

  if (query.status !== 0) {throw new Error(query.stderr || `room log read exited ${query.status}`)}

  return JSON.parse(query.stdout)
}

const memberMessages = (events: RoomEvent[], afterSeq = 0) =>
  events.slice(afterSeq).filter(event => event.kind === 'message.member')

const summary = (events: RoomEvent[]) => events.map(event => ({
  profile: event.actor.profile, round: event.payload.round_index, peer: event.actor.connection_id ?? null, at: event.created_at,
  text: String(event.payload.text).slice(0, 60)
}))

/** A real discussion may run more than one round: each member speaks at most once per round,
 *  and every member answers the first round. */
function expectOnePerMemberRound(replies: RoomEvent[], profiles: string[]): void {
  const keys = replies.map(event => `${event.actor.profile}#${event.payload.round_index}`)
  expect(new Set(keys).size, `duplicate member reply in a round: ${keys}`).toBe(keys.length)
  expect(profiles.filter(profile => !keys.includes(`${profile}#0`))).toEqual([])
}

/** Assistant rows containing `snippet`, per state.db under `home` (root and each profile). */
function assistantRows(home: string, snippet: string): Record<string, number> {
  const query = spawnSync('python3', ['-c', [
    'import glob, json, os, sqlite3, sys',
    'out = {}',
    'for db in glob.glob(sys.argv[1] + "/state.db") + glob.glob(sys.argv[1] + "/profiles/*/state.db"):',
    '    c = sqlite3.connect("file:" + db + "?mode=ro", uri=True)',
    '    out[os.path.relpath(db, sys.argv[1])] = c.execute("select count(*) from messages where role = \'assistant\' and content like ?", ("%" + sys.argv[2] + "%",)).fetchone()[0]',
    'print(json.dumps(out))'
  ].join(NL), home, snippet], { encoding: 'utf8' })

  if (query.status !== 0) {throw new Error(query.stderr || `assistant row read exited ${query.status}`)}

  return JSON.parse(query.stdout)
}

/** A distinctive, markdown-free stretch of a reply to find it on screen and in transcripts. */
function snippetOf(text: string): string {
  const line = text.replace(/[*_#`>[\]]/g, '').split(NL).map(part => part.trim()).find(part => part.length >= 12) ?? text.trim()

  return line.slice(0, 40).trim()
}

/** How many times `snippet` appears in the visible page text. */
async function shownCount(page: Page, snippet: string): Promise<number> {
  return page.evaluate(text => document.body.innerText.split(text).length - 1, snippet)
}

const ASK = 'Each of you: reply with ONE short plain-text sentence (no markdown) naming your favourite colour.'
test.describe('cross-gateway group chat without a lender', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let remote: Awaited<ReturnType<typeof startRemoteGateway>>
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    test.setTimeout(300_000)
    mock = await startMockServer()
    sandbox = createSandbox('xgw')
    writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, MULTIPLEX)
    writeEnvFile(sandbox.hermesHome, undefined, mock.url)
    // Homelab is a plain `hermes serve`: no gateway is lending its bots.
    remote = await startRemoteGateway(sandbox.root, mock.url)
    writeConnectionsRegistry(sandbox, remote.url)
    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
    await page.waitForTimeout(2_000)
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await remote?.close()
    await mock?.close()
    sandbox?.cleanup()
  })

  test('a mixed selection says what is missing, offers Retry, and creates nothing', async () => {
    test.setTimeout(240_000)
    await openBots(page)
    await expect(page.getByText(REMOTE_LABEL).first()).toBeVisible({ timeout: 90_000 })
    const dialog = await openGroupDialog(page)
    await pick(page, isLocalDefault, isRemote)
    const note = dialog.locator('[data-group-engine="unavailable"]')
    await expect(note).toBeVisible({ timeout: 60_000 })
    const words = await note.innerText()
    console.log('PROBE_RESULT leg5 unavailable note', JSON.stringify(words))
    expect(words.trim().length).toBeGreaterThan(20)
    await expect(dialog.locator('[data-group-engine="desktop"]')).toHaveCount(0)
    await expect(dialog.getByRole('button', { name: 'Retry' })).toBeVisible()
    await dialog.getByRole('textbox', { name: 'Group name' }).fill('No lender')
    const create = dialog.getByRole('button', { name: /^Create group/ })
    await expect(create).toBeDisabled()
    await shot(page, 'leg5-unavailable')
    await dialog.getByRole('button', { name: 'Retry' }).click()
    await page.waitForTimeout(3_000)
    await expect(note).toBeVisible()
    await expect(create).toBeDisabled()
    await page.keyboard.press('Escape')
    const stored = await page.evaluate(() => Object.keys(JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')))
    const hosted = spawnSync('python3', ['-c', 'import sqlite3,sys; c=sqlite3.connect("file:"+sys.argv[1]+"/state.db?mode=ro",uri=True); print(c.execute("select count(*) from hosted_rooms").fetchone()[0])', sandbox.hermesHome], { encoding: 'utf8' }).stdout.trim()
    console.log('PROBE_RESULT leg5 rooms after', JSON.stringify({ localStorage: stored, ownerHostedRooms: hosted }))
    expect(stored).not.toContain('No lender')
    expect(hosted).toBe('0')
  })
})

test.describe('every new group chat runs on a gateway (real model turns)', () => {
  test.describe.configure({ mode: 'serial' })
  test.skip(!REAL_KEY, 'NOUS_API_KEY not set: the mock legs below cover what they can')

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let lender: Awaited<ReturnType<typeof startPeerGateway>>
  let app: ElectronApplication | undefined
  let page: Page
  let env: Record<string, string>
  const ROOM = 'Two machines'
  const roundOne: Record<string, string> = {}

  const launch = async (first = false) => {
    const launched = await launchDesktop(env)
    app = launched.app
    page = launched.page
    await waitForAppReady(launched as MockBackendFixture, 120_000)

    // A reopen restores Bot Mode, which has no chat statusbar.
    if (first) {await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })}

    await page.waitForTimeout(2_000)
  }

  const openRoom = async (name: string) => {
    await expect(page.getByRole('button', { name: 'New bot or group chat' }).or(page.getByRole('button', { name: 'Bots', exact: true })).first()).toBeVisible({ timeout: 60_000 })

    if (!await page.getByRole('button', { name: 'New bot or group chat' }).count()) {await openBots(page)}

    if (!await page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true }).count()) {
      await page.getByText(name, { exact: true }).first().click()
    }

    await expect(page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })).toBeVisible({ timeout: 30_000 })
  }

  /** Every member reply after `afterSeq`, once each member has settled its turn. */
  const settledReplies = async (name: string, afterSeq: number, members: number) => {
    await expect.poll(() => memberMessages(roomEvents(sandbox.hermesHome, name), afterSeq).length, { timeout: 240_000, intervals: [2_000] })
      .toBeGreaterThanOrEqual(members)
    // A real discussion may run another round; let it settle before counting.
    let last = -1

    await expect.poll(() => {
      const count = roomEvents(sandbox.hermesHome, name).length
      const quiet = count === last
      last = count

      return quiet
    }, { timeout: 120_000, intervals: [8_000] }).toBe(true)

    return memberMessages(roomEvents(sandbox.hermesHome, name), afterSeq)
  }

  const shownInRoom = (snippet: string) => page.locator('[role="log"]').filter({ visible: true }).evaluate(
    (node, text) => (node as HTMLElement).innerText.split(text).length - 1, snippet)

  test.beforeAll(async () => {
    test.setTimeout(300_000)
    mock = await startMockServer()
    sandbox = createSandbox('xgw-real')
    provision(sandbox.hermesHome, mock.url, MULTIPLEX)
    provision(path.join(sandbox.hermesHome, 'profiles', 'scribe'), mock.url)
    lender = await startPeerGateway(sandbox.root, mock.url, LENT_PROFILE)
    writeConnectionsRegistry(sandbox, lender.url)
    env = buildAppEnv(sandbox)
    console.log(`PROBE_RESULT model ${REAL_MODEL} via Nous inference (real turns); owner home=${sandbox.hermesHome} lender home=${lender.home}`)
    await launch(true)
  })

  test.afterAll(async () => {
    const dir = process.env.CROSS_GATEWAY_SHOTS

    if (dir && lender) {fs.writeFileSync(path.join(dir, `${LEG}-lender-viewer.log`), lender.log())}

    await app?.close().catch(() => undefined)
    await lender?.close()
    await mock?.close()
    sandbox?.cleanup()
  })

  test('leg 1: this device + a named profile on another gateway, both answer once', async () => {
    test.setTimeout(420_000)
    await openBots(page)
    await expect(page.getByText(REMOTE_LABEL).first()).toBeVisible({ timeout: 90_000 })
    const dialog = await openGroupDialog(page)
    const chosen = await pick(page, isLocalDefault, isRemoteCoder)
    const note = dialog.locator('[data-group-engine="gateway-peer"]')
    await expect(note).toHaveText('Runs on this device, including when this Desktop is closed. Only text is sent to bots on other computers.', { timeout: 60_000 })
    await expect(dialog.locator('[data-group-engine="desktop"]')).toHaveCount(0)
    console.log('PROBE_RESULT leg1 dialog', JSON.stringify({ chosen, note: await note.innerText() }))
    await dialog.getByRole('textbox', { name: 'Group name' }).fill(ROOM)
    await shot(page, 'leg1-dialog-gateway-peer')
    await dialog.getByRole('button', { name: 'Create group (2)' }).click()
    await expect(dialog).toBeHidden({ timeout: 30_000 })
    const composer = page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })
    await expect(composer).toBeVisible({ timeout: 30_000 })
    const record = await page.evaluate(name => JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')[name], ROOM)
    console.log('PROBE_RESULT leg1 room record', JSON.stringify({ engine: record?.engine, members: record?.members?.map((m: any) => `${m.connectionId}/${m.targetProfile || m.name}`) }))
    expect(record?.engine?.kind).toBe('gateway')
    await composer.fill(`@all ${ASK}`)
    await composer.press('Enter')
    const replies = await settledReplies(ROOM, 0, 2)
    console.log('PROBE_RESULT leg1 owner hosted_room_events member messages', JSON.stringify(summary(replies)))
    expectOnePerMemberRound(replies, ['default', LENT_PROFILE])
    const byProfile = (profile: string) => replies.filter(event => event.actor.profile === profile && event.payload.round_index === 0)
    roundOne.default = String(byProfile('default')[0].payload.text)
    roundOne.coder = String(byProfile(LENT_PROFILE)[0].payload.text)
    const coderSnippet = snippetOf(roundOne.coder)
    const localSnippet = snippetOf(roundOne.default)

    const where = {
      coder: { lender: assistantRows(lender.home, coderSnippet), owner: assistantRows(sandbox.hermesHome, coderSnippet) },
      default: { lender: assistantRows(lender.home, localSnippet), owner: assistantRows(sandbox.hermesHome, localSnippet) }
    }

    console.log('PROBE_RESULT leg1 assistant rows per state.db', JSON.stringify(where))
    expect(where.coder.lender[`profiles/${LENT_PROFILE}/state.db`]).toBeGreaterThan(0)
    expect(where.coder.lender['state.db'] ?? 0).toBe(0)
    expect(Object.values(where.coder.owner).reduce((a, b) => a + b, 0)).toBe(0)
    expect(where.default.owner['state.db']).toBeGreaterThan(0)
    expect(Object.values(where.default.lender).reduce((a, b) => a + b, 0)).toBe(0)

    for (const snippet of [localSnippet, coderSnippet]) {
      await expect.poll(() => shownInRoom(snippet), { timeout: 60_000 }).toBe(1)
    }

    await expect(page.getByText(/hit an error/)).toHaveCount(0)
    await shot(page, 'leg1-both-replies')
  })

  test('leg 2: a turn sent just before quitting finishes on the gateways; reopen shows each reply once', async () => {
    test.setTimeout(480_000)
    const before = roomEvents(sandbox.hermesHome, ROOM).length
    const composer = page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })
    await composer.fill(`@all Each of you: in two or three plain sentences (no markdown), describe a city you would like to visit and why.`)
    await composer.press('Enter')
    // The send is durable on the owner gateway before Desktop goes away.
    await expect.poll(() => roomEvents(sandbox.hermesHome, ROOM).slice(before).some(event => event.kind === 'message.user'), { timeout: 30_000, intervals: [250] }).toBe(true)
    const repliedBeforeQuit = memberMessages(roomEvents(sandbox.hermesHome, ROOM), before).length
    await app!.close()
    app = undefined
    const closedAt = Date.now() / 1000
    console.log(`PROBE_RESULT leg2 desktop closed at ${closedAt.toFixed(3)} with ${repliedBeforeQuit} replies already in`)
    const replies = await settledReplies(ROOM, before, 2)
    const closed = replies.map(event => ({ profile: event.actor.profile, at: event.created_at, whileClosed: event.created_at > closedAt }))
    console.log('PROBE_RESULT leg2 replies vs exit time', JSON.stringify(closed))
    expectOnePerMemberRound(replies, ['default', LENT_PROFILE])
    expect(closed.some(entry => entry.whileClosed)).toBe(true)
    await launch()
    await openRoom(ROOM)

    for (const text of [...replies.map(event => String(event.payload.text)), roundOne.default, roundOne.coder]) {
      await expect.poll(() => shownInRoom(snippetOf(text)), { timeout: 60_000 }).toBe(1)
    }

    await expect(page.locator('li[data-phase="working"], li[data-phase="queued"]')).toHaveCount(0, { timeout: 30_000 })
    console.log('PROBE_RESULT leg2 after reopen each of 4 replies shown exactly once')
    await shot(page, 'leg2-after-reopen')
  })

  test('leg 3: two bots on this device make a gateway-owned room', async () => {
    test.setTimeout(360_000)
    const name = 'This device only'
    const dialog = await openGroupDialog(page)
    await pick(page, isLocalDefault, isLocalScribe)
    await expect(dialog.locator('[data-group-engine]')).toHaveCount(0)
    await dialog.getByRole('textbox', { name: 'Group name' }).fill(name)
    await dialog.getByRole('button', { name: 'Create group (2)' }).click()
    await expect(dialog).toBeHidden({ timeout: 30_000 })
    const composer = page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })
    await expect(composer).toBeVisible({ timeout: 30_000 })
    const engine = await page.evaluate(room => JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')[room]?.engine, name)
    console.log('PROBE_RESULT leg3 engine', JSON.stringify(engine))
    expect(engine?.kind).toBe('gateway')
    await composer.fill(`@all ${ASK}`)
    await composer.press('Enter')
    const replies = await settledReplies(name, 0, 2)
    console.log('PROBE_RESULT leg3 replies', JSON.stringify(summary(replies)))
    expectOnePerMemberRound(replies, ['default', 'scribe'])
    expect(replies.every(event => event.actor.connection_id === undefined)).toBe(true)
    await page.waitForTimeout(1_500)
    await shot(page, 'leg3-same-computer-room')
  })

  test('leg 6: Create & join from a bot\'s Manage groups makes a gateway-run room with a bot on another computer', async () => {
    test.setTimeout(420_000)
    const name = 'Joined from Manage groups'
    await openBots(page)
    await expect(page.getByText(REMOTE_LABEL).first()).toBeVisible({ timeout: 90_000 })
    const rows = page.locator('[data-roster-key]')
    const texts = await rows.evaluateAll(nodes => nodes.map(node => (node as HTMLElement).innerText.replace(/\s+/g, ' ').trim()))
    // Roster rows show no connection label: this device's default Bot is the one named Hermes.
    const index = texts.findIndex(text => /^Hermes\b/.test(text))
    console.log('PROBE_RESULT leg6 roster rows', JSON.stringify(texts), 'chosen', index)
    expect(index).toBeGreaterThanOrEqual(0)
    await rows.nth(index).click({ button: 'right' })
    await page.getByRole('menuitem', { name: /^(Manage groups|Groups:)/ }).click()
    const manage = page.getByRole('dialog', { name: 'Manage groups' })
    await expect(manage).toBeVisible()
    await manage.getByRole('textbox').fill(name)
    await manage.getByRole('button', { name: 'Create & join' }).click()
    // Create & join opens New group chat with this bot and the name filled in.
    const dialog = page.getByRole('dialog', { name: 'New group chat' })
    await expect(dialog).toBeVisible({ timeout: 30_000 })
    await expect(dialog.getByRole('textbox', { name: 'Group name' })).toHaveValue(name)
    await page.waitForTimeout(3_000)
    const seeded = dialog.locator('label:has([role="checkbox"][aria-checked="true"])')
    await expect(seeded).toHaveCount(1)
    expect(isLocalDefault((await seeded.innerText()).replace(/\s+/g, ' '))).toBe(true)
    const chosen = await pick(page, isRemoteCoder)
    const note = dialog.locator('[data-group-engine="gateway-peer"]')
    await expect(note).toHaveText('Runs on this device, including when this Desktop is closed. Only text is sent to bots on other computers.', { timeout: 60_000 })
    console.log('PROBE_RESULT leg6 dialog', JSON.stringify({ chosen, note: await note.innerText() }))
    await shot(page, 'leg6-create-and-join-dialog')
    await dialog.getByRole('button', { name: 'Create group (2)' }).click()
    await expect(dialog).toBeHidden({ timeout: 30_000 })
    const composer = page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })
    await expect(composer).toBeVisible({ timeout: 30_000 })
    const rooms = await page.evaluate(() => JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}'))
    const engines = Object.fromEntries(Object.entries(rooms).map(([room, record]: [string, any]) => [room, record?.engine]))
    console.log('PROBE_RESULT leg6 room engines', JSON.stringify(engines))
    expect(engines[name]?.kind).toBe('gateway')
    expect(Object.values(engines)).not.toContain('desktop')
    await composer.fill(`@all ${ASK}`)
    await composer.press('Enter')
    const replies = await settledReplies(name, 0, 2)
    console.log('PROBE_RESULT leg6 owner hosted_room_events member messages', JSON.stringify(summary(replies)))
    expectOnePerMemberRound(replies, ['default', LENT_PROFILE])
    await page.waitForTimeout(1_500)
    await shot(page, 'leg6-create-and-join-room')
  })

  test('leg 4: an old Desktop-driven room still opens in the classic workspace and sends', async () => {
    test.setTimeout(360_000)
    const name = 'Old classic room'

    // An older build's cross-machine room: Desktop-driven, members on both computers, some history.
    const members = [
      { name: 'default', handle: 'default-this-device', display_name: 'Hermes', connectionId: 'local', connectionKind: 'local',
        connectionLabel: 'This device', targetProfile: 'default', route: { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' } },
      { name: LENT_PROFILE, handle: LENT_PROFILE, display_name: 'Coder', connectionId: REMOTE_ID, connectionKind: 'remote', connectionLabel: REMOTE_LABEL,
        remoteSource: true, sourceScoped: true, targetProfile: LENT_PROFILE,
        route: { connectionId: REMOTE_ID, mode: 'remote', profile: LENT_PROFILE, targetProfile: LENT_PROFILE } }
    ]

    await page.evaluate(({ room, seated }) => {
      const key = 'hermes.plugin.hermes-bots.group-chats'
      const rooms = JSON.parse(localStorage.getItem(key) || '{}')
      const at = Date.now() - 3_600_000

      rooms[room] = {
        engine: 'desktop', roomId: null, members: seated, watermarks: {}, sessions: {}, holds: {},
        log: [
          { id: 'old-1', thread: 'old-t1', at, from: { kind: 'user', name: 'You' }, text: 'Earlier question from an older build' },
          { id: 'old-2', thread: 'old-t1', at: at + 1_000, from: { kind: 'member', name: seated[0].name }, text: 'Earlier answer one' },
          { id: 'old-3', thread: 'old-t1', at: at + 2_000, from: { kind: 'member', name: seated[1].name }, text: 'Earlier answer two' }
        ]
      }
      localStorage.setItem(key, JSON.stringify(rooms))
    }, { room: name, seated: members })
    await page.reload()
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await page.waitForTimeout(3_000)

    if (!await page.getByRole('button', { name: 'New bot or group chat' }).count()) {await openBots(page)}

    await page.getByText(name, { exact: true }).first().click()
    const composer = page.getByRole('textbox', { name: `Message ${name}` }).filter({ visible: true })
    await expect(composer).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })).toHaveCount(0)
    await expect(page.getByText('Earlier answer two', { exact: true }).filter({ visible: true })).toHaveCount(1)
    await composer.fill(`@all ${ASK}`)
    await composer.press('Enter')

    const log = () => page.evaluate(room => (JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')[room]?.log || [])
      .map((entry: any) => ({ from: entry.from?.name, kind: entry.from?.kind, text: String(entry.text).slice(0, 60) })), name)

    await expect.poll(async () => (await log()).filter((entry: any) => entry.kind === 'member').length, { timeout: 240_000, intervals: [2_000] })
      .toBeGreaterThanOrEqual(3)
    await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0, { timeout: 180_000 })
    const entries = await log()
    console.log('PROBE_RESULT leg4 classic room log', JSON.stringify(entries))
    const engine = await page.evaluate(room => JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')[room]?.engine, name)
    console.log('PROBE_RESULT leg4 engine after send', JSON.stringify(engine))
    expect(engine).toBe('desktop')
    expect(roomEvents(sandbox.hermesHome, name)).toEqual([])
    await expect(page.getByText(/hit an error/)).toHaveCount(0)
    await shot(page, 'leg4-old-classic-room')
  })
})

/** Run a persistent lender behind the same remote connection used by the courier test. */
async function startPeerGateway(root: string, mockUrl: string, lent = '') {
  const apiPort = await freePort()
  const fakeHome = path.join(root, 'homelab-root')
  const home = path.join(fakeHome, '.hermes')
  fs.mkdirSync(home, { recursive: true })

  const config = [MULTIPLEX.trimEnd(), `  room_link_url: http://127.0.0.1:${apiPort}`,
    '  platforms:', '    api_server:', '      enabled: true', `      port: ${apiPort}`, '      host: 127.0.0.1', 'approvals:', '  mode: manual', ''].join(NL)

  provision(home, mockUrl, config)

  // A named profile the lender serves and lends; it inherits the root room_link_url (-> /p/<name>).
  if (lent) {
    const profileHome = path.join(home, 'profiles', lent)
    fs.mkdirSync(profileHome, { recursive: true })
    provision(profileHome, mockUrl, ['approvals:', '  mode: manual', ''].join(NL))
  }

  const python = process.env.CROSS_GATEWAY_PYTHON || path.join(REPO_ROOT, '.venv', 'bin', 'python')

  const child = spawn(python, ['-m', 'hermes_cli.main', 'gateway', 'run'], {
    cwd: REPO_ROOT,
    env: { ...process.env, HOME: fakeHome, HERMES_HOME: home, PYTHONPATH: REPO_ROOT,
      API_SERVER_KEY: REMOTE_TOKEN, API_SERVER_ENABLED: 'true', API_SERVER_PORT: String(apiPort) },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  let log = ''
  child.stdout.on('data', (chunk: Buffer) => { log += chunk.toString() })
  child.stderr.on('data', (chunk: Buffer) => { log += chunk.toString() })
  let viewer: Awaited<ReturnType<typeof startRemoteGateway>> | undefined

  const close = async () => {
    await viewer?.close()

    if (child.exitCode === null && child.signalCode === null) {
      const exited = new Promise<void>(resolve => child.once('exit', () => resolve()))
      child.kill('SIGTERM')
      const killTimer = setTimeout(() => child.kill('SIGKILL'), 5_000)
      await exited.finally(() => clearTimeout(killTimer))
    }
  }

  try {
    await expect.poll(async () => {
      if (child.exitCode !== null || child.signalCode !== null) {throw new Error(`peer gateway exited: ${log}`)}

      try { return (await fetch(`http://127.0.0.1:${apiPort}/v1/models`, { headers: { Authorization: `Bearer ${REMOTE_TOKEN}` } })).ok } catch { return false }
    }, { timeout: 90_000 }).toBe(true)
    // Keep the lender's gateway config when adding the authenticated WebSocket viewer.
    viewer = await startRemoteGateway(root, mockUrl, config)
    console.log(`PROBE_RESULT lender gateway pid=${child.pid} api=http://127.0.0.1:${apiPort} lent=${lent || 'default'}`)

    return { ...viewer, close }
  } catch (error) {
    console.log(`PROBE_RESULT lender gateway failed to start:${NL}${log.slice(-4000)}`)
    await close()
    throw error
  }
}

/** Read the authoritative event log while no Electron process is running. */
function peerRoomLog(home: string, name: string): Array<{ kind: string; payload: Record<string, unknown> }> {
  const query = spawnSync('python3', ['-c', [
    'import json, sqlite3, sys',
    'c = sqlite3.connect("file:" + sys.argv[1] + "/state.db?mode=ro", uri=True)',
    'rows = c.execute("select e.kind, e.payload_json from hosted_room_events e join hosted_rooms r using(room_id) where r.name = ? order by e.seq", (sys.argv[2],))',
    'print(json.dumps([{"kind": k, "payload": json.loads(p)} for k, p in rows]))'
  ].join('\n'), home, name], { encoding: 'utf8' })

  if (query.status !== 0) {throw new Error(query.stderr || `room log read exited ${query.status}`)}

  return JSON.parse(query.stdout)
}

test('mock: gateway peers finish a held turn after Electron quits and reopen once', async () => {
  test.skip(Boolean(REAL_KEY), 'the real-model describe covers the closed-Desktop leg')
  test.setTimeout(420_000)
  const marker = 'PEER_DESKTOP_CLOSED'
  const name = 'Persistent two machines'
  const localReply = 'LOCAL_PERSISTENT_REPLY'
  const peerReply = 'HOMELAB_PERSISTENT_REPLY'
  const localModel = await startMockServer({ holdFirstCompletionContaining: marker, replyForPrompt: () => localReply })
  const peerModel = await startMockServer({ holdFirstCompletionContaining: marker, replyForPrompt: () => peerReply })
  const sandbox = createSandbox('peer-gateway')
  let remote: Awaited<ReturnType<typeof startPeerGateway>> | undefined
  let app: ElectronApplication | undefined

  try {
    writeMockProviderConfig(sandbox.hermesHome, localModel.url, undefined, MULTIPLEX)
    writeEnvFile(sandbox.hermesHome, undefined, localModel.url)
    remote = await startPeerGateway(sandbox.root, peerModel.url)
    writeConnectionsRegistry(sandbox, remote.url)
    const env = buildAppEnv(sandbox)
    let launched = await launchDesktop(env)
    app = launched.app
    let page = launched.page
    await waitForAppReady(launched as MockBackendFixture, 120_000)
    await openBots(page)
    await expect(page.getByText(REMOTE_LABEL).first()).toBeVisible({ timeout: 90_000 })
    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    await page.getByRole('menuitem', { name: 'New group chat' }).click()
    const dialog = page.getByRole('dialog', { name: 'New group chat' })
    const checks = dialog.locator('label:has([role="checkbox"]) [role="checkbox"]')
    await expect(checks).toHaveCount(2)
    await checks.nth(0).click()
    await checks.nth(1).click()
    await expect(dialog.locator('[data-group-engine="gateway-peer"]')).toHaveText(
      'Runs on this device, including when this Desktop is closed. Only text is sent to bots on other computers.')
    await dialog.getByRole('textbox', { name: 'Group name' }).fill(name)
    await dialog.getByRole('button', { name: 'Create group (2)' }).click()
    await expect(dialog).toBeHidden()
    const composer = page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })
    await composer.fill(`@all ${marker}`)
    await expect(page.getByRole('button', { name: 'Send' }).filter({ visible: true })).toBeEnabled({ timeout: 30_000 })
    await composer.press('Enter')
    // Discussion admits one member at a time: quit with the first held and the second outstanding.
    await expect.poll(() => localModel.heldCompletionCount() + peerModel.heldCompletionCount(), { timeout: 120_000 }).toBe(1)
    expect(peerRoomLog(sandbox.hermesHome, name).filter(event => event.kind === 'message.member')).toEqual([])
    await app.close()
    app = undefined
    // Releasing also opens the gate for the provider whose turn has not started yet.
    localModel.releaseHeldStream()
    peerModel.releaseHeldStream()
    const replies = () => peerRoomLog(sandbox.hermesHome, name).filter(event => event.kind === 'message.member').map(event => event.payload.text).sort()
    await expect.poll(replies, { timeout: 120_000, intervals: [1_000] }).toEqual([peerReply, localReply])
    launched = await launchDesktop(env)
    app = launched.app
    page = launched.page
    await waitForAppReady(launched as MockBackendFixture, 120_000)
    await openBots(page)

    if (!await page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true }).count()) {
      await page.getByText(name, { exact: true }).first().click()
    }

    for (const reply of [localReply, peerReply]) {
      await expect(page.getByText(reply, { exact: true }).filter({ visible: true })).toHaveCount(1, { timeout: 30_000 })
    }

    await expect(page.locator('li[data-phase="working"], li[data-phase="queued"]')).toHaveCount(0, { timeout: 30_000 })
    expect(replies()).toEqual([peerReply, localReply])
  } finally {
    localModel.releaseHeldStream()
    peerModel.releaseHeldStream()
    await app?.close()
    await remote?.close()
    await localModel.close()
    await peerModel.close()
    sandbox.cleanup()
  }
})
