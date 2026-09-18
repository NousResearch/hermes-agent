import { execFileSync, execSync } from 'node:child_process'
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
// a room worker. The room store moved from the root state.db to shared-state.db without carrying
// the rows across, so every room from before that upgrade answered "hosted room not found"
// (#109775). Asserted against the REAL Electron app's spawned backend over its JSON-RPC socket,
// with the pre-upgrade layout seeded on disk by the store's own writers.

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..')
const PYTHON = process.env.HERMES_DESKTOP_PYTHON ?? path.join(REPO_ROOT, 'venv', 'bin', 'python')

let fixture: MockBackendFixture | null = null
let rpc: Rpc | null = null

function seedProfile(hermesHome: string, mockUrl: string, name: string): void {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)
}

/** Write a room + one user event into the PRE-isolation layout (root state.db) with the store's own writers. */
function seedLegacyRoom(hermesHome: string): void {
  const script = `
import sys
from pathlib import Path
from gateway import hosted_rooms
legacy = Path(sys.argv[1]) / "state.db"
hosted_rooms.create_room(legacy, room_id="oldroom", name="Pre-split room",
    members=[{"member_id": "m-sentinel", "profile": "sentinel", "handle": "sentinel"},
             {"member_id": "m-friday", "profile": "friday", "handle": "friday"}],
    authority_gateway_id="legacy", now=10)
hosted_rooms.append_event(legacy, room_id="oldroom", event_id="user:u1", kind="message.user",
    actor={"kind": "user", "id": "desktop"}, payload={"text": "hello from before the upgrade", "thread_id": "t1"},
    authority_gateway_id="legacy", authority_epoch=1, now=11)
print([r["room_id"] for r in hosted_rooms.list_rooms(legacy)])
`
  const out = execFileSync(PYTHON, ['-c', script, hermesHome], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
    encoding: 'utf8'
  })
  if (!out.includes('oldroom')) {
    throw new Error(`legacy seed failed: ${out}`)
  }
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

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('hosted-rooms-backend')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  seedProfile(sandbox.hermesHome, mock.url, 'sentinel')
  seedProfile(sandbox.hermesHome, mock.url, 'friday')
  seedLegacyRoom(sandbox.hermesHome)
  expect(fs.existsSync(path.join(sandbox.hermesHome, 'shared-state.db'))).toBe(false)

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

test('rooms from before the shared-state.db split are still reachable from the Desktop backend (#109775)', async () => {
  test.setTimeout(300_000)
  const caps = await rpc!.call('groups.capabilities')
  expect(caps.result?.driver).toBe(true)

  const listed = await rpc!.call('groups.list')
  expect(listed.result?.rooms.map((room: any) => room.room_id)).toContain('oldroom')

  const state = await rpc!.call('groups.state', { room_id: 'oldroom' })
  expect(state.error, JSON.stringify(state.error)).toBeUndefined()
  expect(state.result.room.latest_seq).toBe(1)

  const log = await rpc!.call('groups.log', { room_id: 'oldroom', since_seq: 0, limit: 20 })
  expect(log.result.events.map((event: any) => event.kind)).toEqual(['message.user'])
  // Storage truth: the store now lives in shared-state.db and the legacy file was left intact.
  expect(fs.existsSync(path.join(fixture!.sandbox.hermesHome, 'shared-state.db'))).toBe(true)
  expect(fs.existsSync(path.join(fixture!.sandbox.hermesHome, 'state.db'))).toBe(true)
})
