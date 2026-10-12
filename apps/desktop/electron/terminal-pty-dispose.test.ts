import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

// terminal-ipc.ts imports the Electron and native PTY modules at module scope;
// neither loads under plain node, so both are stubbed and the spawned PTY is
// a fake that captures its onData/onExit listeners.
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

import { registerTerminalIpc } from './terminal-ipc'

interface FakePty {
  listeners: { exit: null | ((event: { exitCode: number | null; signal: string | null }) => void) }
  pty: {
    kill: ReturnType<typeof vi.fn>
    onData: ReturnType<typeof vi.fn>
    onExit: ReturnType<typeof vi.fn>
    pid: number
    resize: ReturnType<typeof vi.fn>
    write: ReturnType<typeof vi.fn>
  }
}

function makeFakePty(): FakePty {
  const listeners: FakePty['listeners'] = { exit: null }

  return {
    listeners,
    pty: {
      kill: vi.fn(),
      onData: vi.fn(),
      onExit: vi.fn((callback: FakePty['listeners']['exit']) => {
        listeners.exit = callback
      }),
      pid: 4242,
      resize: vi.fn(),
      write: vi.fn()
    }
  }
}

function makeSender(id: number) {
  return {
    id,
    isDestroyed: () => false,
    once: vi.fn(),
    send: vi.fn()
  }
}

async function startSession() {
  registerTerminalIpc({
    isWindows: false,
    findOnPath: () => null,
    rememberLog: () => {},
    activeSshTerminalTarget: () => null,
    sshBinary: () => '/usr/bin/ssh',
    ensureBackend: async () => undefined,
    getSshConnectionState: () => undefined
  })

  const sender = makeSender(7)
  const fake = makeFakePty()

  spawnMock.fn.mockReturnValueOnce(fake.pty)

  const start = handles.map.get('hermes:terminal:start') as (
    event: { sender: ReturnType<typeof makeSender> },
    payload: Record<string, unknown>
  ) => Promise<{ cwd: string | null; id: string; shell: string }>

  const session = await start({ sender }, {})

  assert.ok(session?.id)

  return { fake, sender, session }
}

function fireExit(fake: FakePty, event: { exitCode: number | null; signal: string | null }) {
  assert.ok(fake.listeners.exit, 'the session must subscribe to PTY exit')

  fake.listeners.exit(event)
}

test('an exited shell releases its PTY handle without waiting for tab dispose', async () => {
  const { fake, sender, session } = await startSession()

  // The child is gone; the /dev/ptmx master fd must be released right here.
  // Waiting for the tab to close (or the renderer to attach) leaks PTYs until
  // macOS cannot allocate one (#128942).
  fireExit(fake, { exitCode: 0, signal: null })
  assert.equal(fake.pty.kill.mock.calls.length >= 1, true)

  // Killing the handle must not break the buffered-exit delivery: a renderer
  // attaching after the fast exit still gets the output gate's exit payload…
  const attach = handles.map.get('hermes:terminal:attach') as (
    event: { sender: ReturnType<typeof makeSender> },
    id: string
  ) => boolean

  assert.equal(attach({ sender }, session.id), true)

  const exitSend = sender.send.mock.calls.find(([channel]) => String(channel).endsWith(':exit'))

  assert.deepEqual(exitSend?.[1], { code: 0, signal: null })

  // …and once the exit is flushed the session is fully gone.
  const write = handles.map.get('hermes:terminal:write') as (event: unknown, id: string, data: string) => boolean

  assert.equal(write({}, session.id, 'echo hi'), false)
})

test('explicit dispose kills the PTY and drops the session', async () => {
  const { fake, session } = await startSession()

  const dispose = handles.map.get('hermes:terminal:dispose') as (event: unknown, id: string) => boolean

  assert.equal(dispose({}, session.id), true)
  assert.equal(fake.pty.kill.mock.calls.length >= 1, true)

  const write = handles.map.get('hermes:terminal:write') as (event: unknown, id: string, data: string) => boolean

  assert.equal(write({}, session.id, 'echo hi'), false)
  assert.equal(dispose({}, session.id), false)
})
