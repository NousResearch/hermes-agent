import { EventEmitter } from 'node:events'
import fs from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

import { shortSocketTmpDir } from './local-gateway.test-helpers'

// Every ticket-mint failure used to reject with the bare `Gateway ticket bootstrap failed`, which is
// also the string the classification reads (`isStaleLocalGatewayError` → "forget the cached owner and
// re-ensure"). A dead owner, an owner that answered nothing before the deadline, and a helper that
// died on startup were therefore one indistinguishable message in `desktop.log` and in bug reports.
// These are the invariants of the fix: the cause reaches the message, the classification prefix stays.

afterEach(() => {
  vi.useRealTimers()
  vi.doUnmock('node:net')
  vi.doUnmock('node:child_process')
  vi.resetModules()
})

type DialedSocket = EventEmitter & { destroy: ReturnType<typeof vi.fn>; write: ReturnType<typeof vi.fn> }

function dialedSocket(): DialedSocket {
  const socket = new EventEmitter() as DialedSocket

  socket.write = vi.fn(() => true)
  socket.destroy = vi.fn()

  return socket
}

/** A control socket that exists and is private, but whose client dial is the test's own socket. */
async function controlSocketHome(prefix: string, socket: DialedSocket): Promise<{ home: string; server: net.Server; dialed: () => boolean }> {
  const home = await shortSocketTmpDir(prefix)
  const socketPath = path.join(home, 'gateway.sock')
  const server = net.createServer(() => undefined)
  let dials = 0

  await new Promise<void>(resolve => server.listen(socketPath, resolve))
  await fs.chmod(socketPath, 0o600)
  // `local-gateway.ts` only dials through `net`; the real server above is built from the real module.
  vi.doMock('node:net', () => ({ default: { createConnection: () => { dials += 1;

 return socket } } }))

  return { home, server, dialed: () => dials > 0 }
}

/** Yield to the real event loop until the mint has dialed: its own fs checks run first. */
async function untilDialed(dialed: () => boolean): Promise<void> {
  for (let turn = 0; turn < 1_000 && !dialed(); turn += 1) {
    await new Promise(resolve => setImmediate(resolve))
  }

  if (!dialed()) { throw new Error('the mint never dialed the control socket') }
}

async function closeHome(home: string, server: net.Server): Promise<void> {
  await new Promise<void>(resolve => server.close(() => resolve()))
  await fs.rm(home, { recursive: true, force: true })
}

const endpoint = (home: string) => ({ profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' })

test.skipIf(process.platform === 'win32')('a refused control-socket dial reports its cause and errno, and still redials', async () => {
  const socket = dialedSocket()
  const { home, server, dialed } = await controlSocketHome('desktop-ticket-cause-', socket)
  const { isStaleLocalGatewayError, mintLocalGatewayTicket } = await import('./local-gateway')
  const refusal = Object.assign(new Error(`connect ECONNREFUSED ${path.join(home, 'gateway.sock')}`), { code: 'ECONNREFUSED' })

  try {
    const mint = mintLocalGatewayTicket(endpoint(home)).catch(error => error)

    await untilDialed(dialed)
    socket.emit('error', refusal)
    socket.emit('close')
    const failure = await mint

    expect(failure.message).toBe(`Gateway ticket bootstrap failed: connect ECONNREFUSED ${path.join(home, 'gateway.sock')}`)
    expect(failure.code).toBe('ECONNREFUSED')
    // The prefix (now with a cause appended) is still the classification contract.
    expect(isStaleLocalGatewayError(failure)).toBe(true)
  } finally {
    await closeHome(home, server)
  }
})

test.skipIf(process.platform === 'win32')('an owner that never answers names the deadline instead of a bare failure', async () => {
  // Only the mint's deadline is faked; `setImmediate` stays real so the fs checks can finish.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  const socket = dialedSocket()

  socket.destroy = vi.fn((error?: Error) => { queueMicrotask(() => { socket.emit('error', error); socket.emit('close') }) })
  const { home, server, dialed } = await controlSocketHome('desktop-ticket-deadline-', socket)
  const { mintLocalGatewayTicket } = await import('./local-gateway')

  try {
    const mint = mintLocalGatewayTicket(endpoint(home)).catch(error => error)

    await untilDialed(dialed)
    await vi.advanceTimersByTimeAsync(5_000)

    expect((await mint).message).toBe('Gateway ticket bootstrap failed: no reply within 5000ms')
  } finally {
    await closeHome(home, server)
  }
})

test('a ticket bridge that dies on startup carries its exit code and the helper’s last stderr line', async () => {
  const child = Object.assign(new EventEmitter(), {
    stdin: Object.assign(new EventEmitter(), { write: vi.fn(() => true) }),
    stdout: Object.assign(new EventEmitter(), { setEncoding: vi.fn() }),
    stderr: Object.assign(new EventEmitter(), { setEncoding: vi.fn() }),
    kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { closeGatewayTicketBridges, mintGatewayTicketWithPython } = await import('./local-gateway-python')

  try {
    const mint = mintGatewayTicketWithPython({ command: 'python3', env: {} }, '/', { profile_id: '/home/x/.hermes', instance_id: 'owner', runtime_protocol: 1 }, 'native-http')
      .catch(error => error)

    // The bridge used to `stderr.resume()` (read and discard) exactly this.
    child.stderr.emit('data', 'Traceback (most recent call last):\n  File "<string>", line 3, in <module>\nModuleNotFoundError: No module named \'hermes_cli\'\n')
    child.emit('close', 1)

    // A crash ends with the exception; a usage error only opens with it, so both ends are kept.
    expect((await mint).message).toBe('Gateway ticket bootstrap failed: the ticket bridge exited (code 1): Traceback (most recent call last): … ' +
      "ModuleNotFoundError: No module named 'hermes_cli'")
  } finally {
    closeGatewayTicketBridges()
  }
})
