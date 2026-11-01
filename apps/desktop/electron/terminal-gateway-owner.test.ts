import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

// terminal-ipc.ts loads Electron and the native PTY at module scope; both are
// stubbed, and the ssh argv builder is replaced so a remote spawn is visible.
const handles = vi.hoisted(() => ({ map: new Map<string, (...args: never[]) => unknown>() }))
const spawnMock = vi.hoisted(() => ({ fn: vi.fn() }))

vi.mock('electron', () => ({
  app: { getPath: () => '/tmp', getVersion: () => '0.0.0-test' },
  ipcMain: {
    handle: (channel: string, listener: (...args: never[]) => unknown) => {
      handles.map.set(channel, listener)
    }
  }
}))
vi.mock('node-pty', () => ({ default: { spawn: (...args: unknown[]) => spawnMock.fn(...args) } }))
vi.mock('./ssh-connection', async importOriginal => ({
  ...(await importOriginal<object>()),
  buildInteractiveSshArgs: (ssh: { host: string }) => [`ssh-to-${ssh.host}`]
}))

import type { ConnectionRegistry } from './connection-registry'
import { terminalGatewayTarget } from './terminal-gateway-target'
import { registerTerminalIpc } from './terminal-ipc'

const registry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device' },
    { id: 'nas', kind: 'ssh', label: 'NAS' },
    { id: 'cloudbox', kind: 'remote', label: 'Cloud box' }
  ],
  primary: 'local',
  version: 2
} as unknown as ConnectionRegistry

const nasTunnel = { registryConnectionId: 'nas', ssh: { host: 'nas.lan' } }

type Start = (
  event: { sender: ReturnType<typeof makeSender> },
  payload: Record<string, unknown>
) => Promise<{ connectionId?: null | string; id: string; shell: string; unavailable?: string }>

function makeSender(id: number) {
  return { id, isDestroyed: () => false, once: vi.fn(), send: vi.fn() }
}

function fakePty() {
  const listeners: { exit: null | ((event: { exitCode: number | null; signal: null | string }) => void) } = {
    exit: null
  }

  return {
    listeners,
    pty: {
      // A real PTY reports its exit once, however often kill() runs.
      kill: vi.fn(() => {
        const exit = listeners.exit
        listeners.exit = null
        exit?.({ exitCode: null, signal: '1' })
      }),
      onData: vi.fn(),
      onExit: vi.fn((callback: typeof listeners.exit) => {
        listeners.exit = callback
      }),
      pid: 4242,
      resize: vi.fn(),
      write: vi.fn()
    }
  }
}

/** Register the IPC with the window route pointing at `windowTarget`. */
function register(windowTarget: unknown) {
  const pool = new Map([['conn:nas::default', nasTunnel]])

  const api = registerTerminalIpc({
    activeSshTerminalTarget: () => windowTarget,
    ensureBackend: async () => undefined,
    ensureGatewayBackend: async () => undefined,
    findOnPath: () => null,
    gatewayTarget: owner => terminalGatewayTarget(owner, registry, pool),
    getSshConnectionState: scope => pool.get(scope),
    isWindows: false,
    rememberLog: () => {},
    sshBinary: () => '/usr/bin/ssh'
  })

  return { api, start: handles.map.get('hermes:terminal:start') as Start }
}

async function spawnFor(start: Start, payload: Record<string, unknown>) {
  const fake = fakePty()
  spawnMock.fn.mockReset().mockReturnValueOnce(fake.pty)
  const sender = makeSender(7)
  const session = await start({ sender }, payload)

  return { argv: spawnMock.fn.mock.calls[0] as [string, string[]] | undefined, fake, sender, session }
}

test('a tab runs on the gateway it recorded, not the one the window shows', async () => {
  // Window on This device: the NAS tab still opens its shell on the NAS.
  const onLocal = register(null)
  const nas = await spawnFor(onLocal.start, { connectionId: 'nas', profile: 'default' })

  assert.deepEqual(nas.argv?.slice(0, 2), ['/usr/bin/ssh', ['ssh-to-nas.lan']])
  assert.equal(nas.session.connectionId, 'nas')

  // Window on the NAS: a This device tab stays a local shell.
  const onNas = register({ scope: 'conn:nas::default', ssh: nasTunnel.ssh })
  const local = await spawnFor(onNas.start, { connectionId: 'local' })

  assert.notEqual(local.argv?.[0], '/usr/bin/ssh')
  assert.equal(local.session.connectionId, 'local')
})

test('a removed or shell-less gateway is reported, never answered with a local shell', async () => {
  const { start } = register(null)

  for (const [connectionId, unavailable] of [
    ['gone', 'missing'],
    ['cloudbox', 'no-shell']
  ]) {
    const result = await spawnFor(start, { connectionId })

    assert.equal(result.session.unavailable, unavailable)
    assert.equal(result.argv, undefined, `${connectionId} must not spawn anything`)
  }
})

test('an SSH teardown leaves the tab disconnected instead of closing it', async () => {
  const { api, start } = register(null)
  const nas = await spawnFor(start, { connectionId: 'nas', profile: 'default' })
  const attach = handles.map.get('hermes:terminal:attach') as (event: unknown, id: string) => boolean

  assert.equal(attach({ sender: nas.sender }, nas.session.id), true)
  api.disposeTerminalSessionsForSshScope('conn:nas::default')

  const exits = nas.sender.send.mock.calls.filter(([channel]) => String(channel).endsWith(':exit'))

  // The kill's own exit (signal 1) would close the tab; the first verdict wins.
  assert.deepEqual(
    exits.map(([, payload]) => payload),
    [{ code: null, signal: 'disconnected' }]
  )
})

test('an SSH teardown before the tab attaches still delivers disconnected on attach', async () => {
  const { api, start } = register(null)
  const nas = await spawnFor(start, { connectionId: 'nas', profile: 'default' })
  const attach = handles.map.get('hermes:terminal:attach') as (event: unknown, id: string) => boolean

  api.disposeTerminalSessionsForSshScope('conn:nas::default')

  // The shell stops at once; the session lives on until the exit is delivered.
  assert.equal(nas.fake.pty.kill.mock.calls.length >= 1, true)
  assert.equal(attach({ sender: nas.sender }, nas.session.id), true)

  const exits = nas.sender.send.mock.calls.filter(([channel]) => String(channel).endsWith(':exit'))

  assert.deepEqual(
    exits.map(([, payload]) => payload),
    [{ code: null, signal: 'disconnected' }]
  )
  // Delivered once, then the session is gone.
  assert.equal(attach({ sender: nas.sender }, nas.session.id), false)
})

test('app quit kills shells without telling the tabs they exited', async () => {
  const { api, start } = register(null)
  const local = await spawnFor(start, { connectionId: 'local' })
  const attach = handles.map.get('hermes:terminal:attach') as (event: unknown, id: string) => boolean

  assert.equal(attach({ sender: local.sender }, local.session.id), true)
  api.disposeAllTerminalSessions()

  // An exit here would close the tab and drop it from the restored layout.
  assert.equal(local.fake.pty.kill.mock.calls.length >= 1, true)
  assert.equal(local.sender.send.mock.calls.filter(([channel]) => String(channel).endsWith(':exit')).length, 0)
})
