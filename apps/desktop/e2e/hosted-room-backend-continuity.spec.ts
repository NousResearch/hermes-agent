import { execSync } from 'node:child_process'
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
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { expect, test } from './test'

// Hosted Group Chat rooms live in the gateway, and the Desktop's own `hermes serve` backend runs
// a room worker. A member turn used to keep the profile's `bot_room` active-session slot for the
// life of the backend process, so a second room worker on the same home (the messaging gateway
// beside Desktop serve) was refused every turn once the driver lease flipped (#106847). Asserted
// against the REAL Electron app's spawned backend over its JSON-RPC socket plus storage truth
// (`profiles/<member>/runtime/active_sessions.json`).

type Page = MockBackendFixture['page']


let fixture: MockBackendFixture | null = null
let rpc: Rpc | null = null

function seedProfile(hermesHome: string, mockUrl: string, name: string): void {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)
}

/** Locate the Electron-spawned `hermes serve` backend for this sandbox: its port and session token. */
function findBackend(hermesHome: string): { port: number; token: string; pid: number } {
  const pids = execSync(`pgrep -f "serve --host 127.0.0.1 --port 0" || true`, { encoding: 'utf8' })
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean)
  for (const pidText of pids) {
    const pid = Number(pidText)
    let environ = ''
    try {
      environ = fs.readFileSync(`/proc/${pid}/environ`, 'latin1')
    } catch {
      continue
    }
    const vars = new Map(environ.split('\0').map(kv => [kv.slice(0, kv.indexOf('=')), kv.slice(kv.indexOf('=') + 1)]))
    if (vars.get('HERMES_HOME') !== hermesHome) {
      continue
    }
    const token = vars.get('HERMES_DASHBOARD_SESSION_TOKEN') ?? ''
    // The backend inherits Electron's remote-debugging socket fd too; the serve port is the listener
    // that only the hermes process holds.
    const listen = execSync(`ss -ltnp | grep "pid=${pid}," | grep -v electron || true`, { encoding: 'utf8' })
    const port = Number(/127\.0\.0\.1:(\d+)/.exec(listen)?.[1] ?? 0)
    if (token && port) {
      return { port, token, pid }
    }
  }
  throw new Error(`no hermes serve backend found for ${hermesHome} (pids: ${pids.join(',')})`)
}

/** Minimal JSON-RPC client over the backend's /api/ws (Node's global WebSocket; no Origin header is sent,
 *  which the loopback backend accepts — the session token is the auth boundary). */
class Rpc {
  private ws: WebSocket
  private id = 0
  private ready: Promise<void>

  constructor(port: number, token: string) {
    this.ws = new WebSocket(`ws://127.0.0.1:${port}/api/ws?token=${token}`)
    this.ready = new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve(), { once: true })
      this.ws.addEventListener('error', () => reject(new Error(`ws upgrade to 127.0.0.1:${port} failed`)), { once: true })
    })
  }

  async call(method: string, params: Record<string, unknown> = {}): Promise<any> {
    await this.ready
    const id = `e2e-${++this.id}`
    return new Promise((resolve, reject) => {
      const onMessage = (event: MessageEvent) => {
        const frame = JSON.parse(String(event.data))
        if (frame.id === id) {
          this.ws.removeEventListener('message', onMessage)
          resolve(frame)
        }
      }
      this.ws.addEventListener('message', onMessage)
      this.ws.addEventListener('error', () => reject(new Error(`ws error during ${method}`)), { once: true })
      this.ws.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
    })
  }

  close(): void {
    this.ws.close()
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

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('hosted-rooms-backend')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  seedProfile(sandbox.hermesHome, mock.url, 'sentinel')
  seedProfile(sandbox.hermesHome, mock.url, 'friday')

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  fixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      rpc?.close()
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 120_000)
  const backend = findBackend(sandbox.hermesHome)
  rpc = new Rpc(backend.port, backend.token)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test("a hosted room turn driven by the Desktop backend releases the member's bot_room slot (#106847)", async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const created = await rpc!.call('groups.create', {
    room_id: 'r-slot',
    name: 'Slot probe',
    members: [
      { member_id: 'm-sentinel', profile: 'sentinel', handle: 'sentinel' },
      { member_id: 'm-friday', profile: 'friday', handle: 'friday' }
    ]
  })
  expect(created.error, JSON.stringify(created.error)).toBeUndefined()
  const sent = await rpc!.call('groups.send', {
    room_id: 'r-slot',
    event_id: 'u1',
    payload: { text: '@sentinel say hello', thread_id: 't1' }
  })
  expect(sent.error, JSON.stringify(sent.error)).toBeUndefined()

  const kinds = async () => {
    const log = await rpc!.call('groups.log', { room_id: 'r-slot', since_seq: 0, limit: 50 })
    return log.result.events.map((event: any) => event.kind) as string[]
  }
  await expect.poll(kinds, { timeout: 180_000, intervals: [1_000] }).toContain('message.member')
  await expect
    .poll(kinds, { timeout: 60_000, intervals: [1_000] })
    .toContain('room.activity')

  const slotsPath = path.join(fixture!.sandbox.hermesHome, 'profiles', 'sentinel', 'runtime', 'active_sessions.json')
  const botRoomSlots = () => {
    if (!fs.existsSync(slotsPath)) {
      return []
    }
    const entries = JSON.parse(fs.readFileSync(slotsPath, 'utf8')).entries ?? []
    return entries.filter((entry: any) => entry.surface === 'bot_room')
  }
  // The slot is claimed for the turn and must be gone once the turn settled — on the pre-fix build it
  // stays (updated_at === started_at) until the backend process exits.
  await expect.poll(botRoomSlots, { timeout: 20_000, intervals: [500] }).toEqual([])

  // Rendered surface control: the Desktop Bots pane is healthy alongside the backend-driven room.
  await openBots(page)
  await expect(page.getByRole('button', { name: /^sentinel\b/i }).filter({ visible: true }).first()).toBeVisible({
    timeout: 30_000
  })
  await page.screenshot({ path: 'test-results/hosted-rooms-backend-bots-pane.png' })
})
