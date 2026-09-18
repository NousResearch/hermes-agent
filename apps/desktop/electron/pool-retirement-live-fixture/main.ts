import assert from 'node:assert/strict'
import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'

import { app, BrowserWindow, ipcMain } from 'electron'

import { stopBackendChild, waitForBackendExit } from '../backend-child'
import { createPoolRetirer } from '../pool-retire'
import { createPoolRetirementClient } from '../pool-retire-http'
import { LocalBackendSpawnCoordinator, type LocalBackendSpawnRequest, registerLocalBackendExitFinalizer } from '../pool-spawn-coordinator'

interface Resident {
  process: ChildProcessWithoutNullStreams
  home: string
  token: string
  port: number | null
  activeTurn: boolean
  lastActiveAt: number
  output: string
}
interface Receipt {
  event: string
  key?: string
  pid?: number
  active: number
  live: number
  [detail: string]: unknown
}

const [root, repo, python, fixture] = process.argv.slice(2)
assert.ok(root && repo && python && fixture)
assert.equal(process.type, 'browser')
assert.ok(process.versions.electron)
app.setPath('userData', join(root, 'user-data'))
app.setPath('sessionData', join(root, 'session-data'))
app.commandLine.appendSwitch('disable-background-networking')
// Cleanup owns app exit even after destroying the last hidden fixture window.
app.on('window-all-closed', () => {})
const coordinator = new LocalBackendSpawnCoordinator(3)
const pool = new Map<string, Resident>()
const children = new Map<string, Resident>()
const live = new Set<number>()
const receipts: Receipt[] = []
const retired: string[] = []
const opened: string[] = []
const errors: string[] = []
const tickets: LocalBackendSpawnRequest[] = []
let maxLiveServeChildren = 0
let busyCronSurvived = false
let finishing = false
let retirer: ReturnType<typeof createPoolRetirer<Resident>> | undefined
let window: BrowserWindow | undefined
let descriptorCalls = 0
let rendererParking: { parked: Record<string, string[]>; redials: number; open: number } | undefined

const stopDeps = { forceKillProcessTree: () => { throw new Error('POSIX fixture only') } }

const record = (event: string, key?: string, details: Record<string, unknown> = {}) => {
  receipts.push({ event, key, active: coordinator.activeCount, live: live.size, ...details })
}

const alive = (pid: number) => {
  try { process.kill(pid, 0);

 return true } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ESRCH') {return false}
    throw error
  }
}

async function until(description: string, predicate: () => boolean | Promise<boolean>, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs

  while (!await predicate()) {
    if (Date.now() > deadline) {
      throw new Error(`Timed out: ${description}\n${JSON.stringify(receipts)}\n${[...children].map(([key, entry]) => `${key}: ${entry.output}`).join('\n')}`)
    }

    await delay(30)
  }
}

async function http(entry: Resident, route: string, body?: Record<string, string>, token = entry.token) {
  const response = await fetch(`http://127.0.0.1:${entry.port}${route}`, {
    method: body ? 'POST' : 'GET', headers: {
      'Content-Type': 'application/json', ...(token ? { 'X-Hermes-Session-Token': token } : {}),
    }, ...(body ? { body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(5000),
  })

  return { status: response.status, body: await response.json() as Record<string, unknown> }
}

const client = createPoolRetirementClient(async (url, token, options) => {
  const response = await fetch(url, { method: options.method,
    headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' },
    body: JSON.stringify(options.body), signal: AbortSignal.timeout(options.timeoutMs) })

  const reply = await response.json() as Record<string, unknown>
  record('http', undefined, { action: options.body.action, status: response.status, ok: reply.ok, idle: reply.idle })

  if (!response.ok) {throw new Error(`Retirement HTTP ${response.status}: ${JSON.stringify(reply)}`)}

  return reply
})

async function spawnResident(key: string, release: () => void): Promise<Resident> {
  const home = join(root, key, '.hermes')
  mkdirSync(home, { recursive: true })
  writeFileSync(join(home, 'config.yaml'), 'cron:\n  script_timeout: 160\n')
  const token = randomUUID()

  const child = spawn(python, ['-u', join(fixture, 'serve.py'), repo, key], {
    cwd: join(root, key), detached: true, stdio: 'pipe',
    env: { ...process.env, HOME: join(root, key), USERPROFILE: join(root, key), HERMES_HOME: home,
      HERMES_DESKTOP: '1', HERMES_SERVE_HEADLESS: '1', HERMES_DASHBOARD_SESSION_TOKEN: token,
      PYTHONUNBUFFERED: '1', PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1',
      // Environment contains only the launch allowlist and fixture values.
    },
  })

  const entry: Resident = { process: child, home,
    token, port: null, activeTurn: false,
    // All residents are renderer-occupied/recent; backend-only work is the authority.
    lastActiveAt: Date.now(), output: '' }

  children.set(key, entry)
  pool.set(key, entry)
  child.on('error', error => { entry.output += String(error) })
  child.stdout.on('data', chunk => { entry.output += String(chunk) })
  child.stderr.on('data', chunk => { entry.output += String(chunk) })
  child.once('exit', (code, signal) => {
    live.delete(child.pid!)
    record('exit', key, { pid: child.pid, code, signal })
  })
  registerLocalBackendExitFinalizer(pool, key, entry, () => {
    record('release', key, { pid: child.pid, osAlive: alive(child.pid!) })
    release()
  })
  await new Promise<void>((resolveSpawn, reject) => {
    child.once('spawn', resolveSpawn)
    child.once('error', error => { release(); reject(error) })
  })
  live.add(child.pid!)
  maxLiveServeChildren = Math.max(maxLiveServeChildren, live.size)
  record('spawn', key, { pid: child.pid })
  assert.ok(live.size <= coordinator.limit, 'Live OS children exceeded the pool cap')
  await until(`${key} HERMES_BACKEND_READY`, () => {
    if (child.exitCode !== null || child.signalCode !== null) {throw new Error(`${key} exited before ready:\n${entry.output}`)}
    const match = entry.output.match(/HERMES_BACKEND_READY[^\n]*port=(\d+)/)

    if (match) {entry.port = Number(match[1])}

    return Boolean(match)
  }, 60_000)
  record('ready', key, { pid: child.pid, port: entry.port })

  return entry
}

async function run() {
  const cron = await spawnResident('cron-busy', await coordinator.acquire('cron-busy'))
  const idleA = await spawnResident('idle-a', await coordinator.acquire('idle-a'))
  await spawnResident('idle-b', await coordinator.acquire('idle-b'))
  assert.equal(coordinator.activeCount, 3)
  const cronPid = JSON.parse(readFileSync(join(cron.home, 'cron-started'), 'utf8')).pid as number
  assert.ok(alive(cronPid), 'A real no-agent cron script must be running')
  const heartbeat = readFileSync(join(cron.home, 'cron-heartbeat'), 'utf8')
  assert.equal((await http(cron, '/api/health/idle')).body.idle, false)
  assert.equal(await client.prepare('cron-busy', cron), null, 'Backend-only cron vetoes retirement')
  const unauthenticated = await http(idleA, '/api/health/retirement', { action: 'prepare' }, '')
  assert.equal(unauthenticated.status, 401)
  assert.equal(unauthenticated.body.token, undefined)
  const firstPermit = await client.prepare('idle-a', idleA)
  assert.ok(firstPermit, 'Live idle backend must implement POST prepare')
  assert.equal(await client.commit('idle-a', idleA, 'not-the-permit'), false)
  await client.cancel('idle-a', idleA, firstPermit)
  const nextPermit = await client.prepare('idle-a', idleA)
  assert.ok(nextPermit && nextPermit !== firstPermit, 'Cancel must reopen admission and revoke the old token')
  await client.cancel('idle-a', idleA, nextPermit)

  ipcMain.handle('retirement-fixture:descriptor', (_event, key: string) => {
    descriptorCalls += 1
    record('descriptor', key)
    const entry = children.get(key)
    assert.ok(entry?.port, `No real backend for ${key}`)

    return { mode: 'local', authMode: 'token', profile: key, token: entry.token,
      baseUrl: `http://127.0.0.1:${entry.port}`,
      wsUrl: `ws://127.0.0.1:${entry.port}/api/ws?token=${encodeURIComponent(entry.token)}` }
  })
  window = new BrowserWindow({ show: false, webPreferences: {
    preload: join(root, 'preload.cjs'), contextIsolation: true, nodeIntegration: false,
    sandbox: true, backgroundThrottling: false,
  } })
  await window.loadFile(join(root, 'renderer.html'))
  assert.equal(await window.webContents.executeJavaScript('window.poolRetirementRenderer.open()'), 4,
    'Real renderer opens legacy and registry WebSockets for both idle children')
  const dialsBeforeRetirement = descriptorCalls

  retirer = createPoolRetirer<Resident>({ pool, coordinator,
    prepare: async (key, entry) => {
      const token = await client.prepare(key, entry)
      record(token ? 'prepared' : 'busy', key)

      return token
    },
    commit: async (key, entry, token) => {
      const committed = await client.commit(key, entry, token)

      if (committed) {
        assert.equal(await client.prepare(key, entry), token, 'A lost commit reply must recover the committed permit')
        assert.equal((await http(entry, '/api/health/retirement', { action: 'cancel', token })).body.ok, false,
          'Committed retirement cannot reopen admission')
        record('committed', key)
      }

      return committed
    },
    cancel: client.cancel,
    onRetiring: key => {
      retired.push(key)
      record('park', key)
      window!.webContents.send('retirement-fixture:retiring', key)
    },
    stopBackend: async key => {
      const entry = pool.get(key)!
      record('signal', key)
      const acquiredBefore = opened.length
      stopBackendChild(entry.process, stopDeps)
      const exit = waitForBackendExit(entry.process, stopDeps, 15_000)
      await until(`${key} real SIGTERM shutdown barrier`, () => entry.output.includes('RETIREMENT_SHUTDOWN_STARTED'))
      assert.ok(alive(entry.process.pid!))
      assert.equal(entry.process.exitCode, null)
      assert.equal(entry.process.signalCode, null)
      assert.equal(coordinator.activeCount, 3, 'SIGTERM is not free capacity')
      assert.equal(opened.length, acquiredBefore, 'A successor must remain queued throughout real shutdown')
      record('shutdown-held', key)
      writeFileSync(join(entry.home, 'allow-exit'), '')
      await exit
      assert.equal(alive(entry.process.pid!), false)
    },
    log: message => { errors.push(message) },
  })
  const direct = coordinator.request('foreground-d', { timeoutMs: 45_000, priority: 'foreground' })
  const promoted = coordinator.request('promoted-e', { timeoutMs: 45_000, priority: 'background' })
  tickets.push(direct, promoted)
  assert.equal(direct.queued, true)
  assert.equal(promoted.queued, true)
  assert.equal(promoted.promote('foreground'), true)
  assert.deepEqual([...coordinator.foregroundWaiters], ['foreground-d', 'promoted-e'])
  record('waiters', undefined, { keys: [...coordinator.foregroundWaiters] })
  await Promise.all(([['foreground-d', direct], ['promoted-e', promoted]] as const).map(async ([name, ticket]) => {
    const release = await ticket.acquired
    opened.push(name)
    record('acquired', name)
    await spawnResident(name, release)
  }))
  assert.deepEqual(errors, [])
  assert.deepEqual(retired, ['idle-a', 'idle-b'])
  assert.equal(coordinator.queuedCount, 0)
  assert.equal(coordinator.activeCount, 3)

  for (const [oldKey, newKey] of [['idle-a', 'foreground-d'], ['idle-b', 'promoted-e']]) {
    const index = (event: string, key: string) => receipts.findIndex(receipt => receipt.event === event && receipt.key === key)
    assert.ok(index('committed', oldKey) < index('park', oldKey))
    assert.ok(index('park', oldKey) < index('signal', oldKey))
    assert.ok(index('signal', oldKey) < index('shutdown-held', oldKey))
    assert.ok(index('shutdown-held', oldKey) < index('exit', oldKey))
    assert.ok(index('exit', oldKey) < index('release', oldKey))
    assert.ok(index('release', oldKey) < index('acquired', newKey))
    assert.ok(index('acquired', newKey) < index('spawn', newKey))
  }

  assert.ok(receipts.filter(receipt => receipt.event === 'release').every(receipt => receipt.osAlive === false))
  assert.ok(alive(cron.process.pid!) && alive(cronPid))
  assert.equal((await http(cron, '/api/health/idle')).body.idle, false)
  await until('cron made progress after reclamation', () => readFileSync(join(cron.home, 'cron-heartbeat'), 'utf8') !== heartbeat)
  assert.equal(retired.includes('cron-busy'), false)
  busyCronSurvived = true
  const parked = await window.webContents.executeJavaScript('window.poolRetirementRenderer.wake()')
  rendererParking = { ...parked, redials: descriptorCalls - dialsBeforeRetirement }
  assert.deepEqual(rendererParking?.parked['idle-a'], ['idle-a', 'conn:local::idle-a'])
  assert.deepEqual(rendererParking?.parked['idle-b'], ['idle-b', 'conn:local::idle-b'])
  assert.equal(rendererParking?.open, 0)
  assert.equal(rendererParking?.redials, 0, 'Socket close and wake sweeps must not redial parked scopes')
  record('renderer-parked', undefined, rendererParking)
  writeFileSync(join(cron.home, 'finish-cron'), '')
  await until('real cron completed and released backend activity', async () =>
    existsSync(join(cron.home, 'cron-finished')) && !alive(cronPid) && (await http(cron, '/api/health/idle')).body.idle === true)
  record('cron-completed', 'cron-busy', { pid: cronPid })
}

async function finish(error?: unknown) {
  if (finishing) {return}
  finishing = true
  retirer?.dispose()

  for (const ticket of tickets) {ticket.cancel()}
  let cleanupError: unknown

  try {
    if (window && !window.isDestroyed()) {window.destroy()}

    for (const entry of children.values()) {
      writeFileSync(join(entry.home, 'finish-cron'), '')
      writeFileSync(join(entry.home, 'allow-exit'), '')
    }

    const cron = children.get('cron-busy')

    if (cron && existsSync(join(cron.home, 'cron-started'))) {
      const pid = JSON.parse(readFileSync(join(cron.home, 'cron-started'), 'utf8')).pid as number
      await until('cron script cleanup', () => !alive(pid), 5000).catch(() => { process.kill(pid, 'SIGKILL') })
    }

    await Promise.all([...children.values()].map(async entry => {
      stopBackendChild(entry.process, stopDeps)
      await waitForBackendExit(entry.process, stopDeps, 5000)
    }))
    assert.equal(live.size, 0)
    assert.equal(coordinator.activeCount, 0)
  } catch (caught) { cleanupError = caught }

  const failure = error || cleanupError
  writeFileSync(join(root, 'result.json'), JSON.stringify({
    ok: !failure, error: failure ? String((failure as Error).stack || failure) : undefined,
    cleanupError: cleanupError ? String(cleanupError) : undefined, cleanedUp: !cleanupError,
    electronVersion: process.versions.electron, processType: process.type,
    capacity: coordinator.limit, maxLiveServeChildren, retired, opened, busyCronSurvived,
    rendererParking, receipts,
    ...(failure ? { backendOutput: Object.fromEntries([...children].map(([key, entry]) => [key, entry.output])) } : {}),
  }, null, 2))
  app.exit(failure ? 1 : 0)
}

process.once('SIGTERM', () => { void finish(new Error('Native fixture interrupted')) })
app.whenReady().then(run).then(() => finish(), finish)
