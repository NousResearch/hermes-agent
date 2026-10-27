import { spawn } from 'node:child_process'
import { once } from 'node:events'
import net from 'node:net'
import type { AddressInfo } from 'node:net'

import { describe, expect, it, vi } from 'vitest'

import { teardownSshState, teardownSshTransport } from './connection-apply'
import { recoverRemotePoolBackend, sshRecoveryToken } from './remote-backend-recovery'
import { connect, fingerprintToken, LOCKFILE_SCHEMA_VERSION, PROTOCOL_VERSION, spawnLogPath } from './remote-lifecycle'
import { ensureHealthyPooledRemoteBackendForDispatch } from './remote-liveness'

function deferred() {
  let resolve!: () => void

  const promise = new Promise<void>(done => {
    resolve = done
  })

  return { promise, resolve }
}

async function openForward(remotePort: number) {
  const sockets = new Set<net.Socket>()

  const server = net.createServer(socket => {
    const upstream = net.connect(remotePort, '127.0.0.1')

    for (const stream of [socket, upstream]) {
      sockets.add(stream)
      stream.on('close', () => sockets.delete(stream))
      stream.on('error', () => {
        socket.destroy()
        upstream.destroy()
      })
    }

    socket.pipe(upstream).pipe(socket)
  })

  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const localPort = (server.address() as AddressInfo).port

  return {
    baseUrl: `http://127.0.0.1:${localPort}`,
    localPort,
    async close() {
      if (!server.listening) {
        return
      }

      const closed = once(server, 'close')
      server.close()

      for (const socket of sockets) {
        socket.destroy()
      }

      await closed
    }
  }
}

describe('remote backend recovery', () => {
  it('repairs a timed-out forward without terminating the backend or its ongoing work', async () => {
    // The backend keeps doing work while one health response is delayed. Killing
    // this process to repair its tunnel would lose both its PID and progress.
    const child = spawn(
      process.execPath,
      [
        '-e',
        `const http = require('node:http');
let progress = 0;
let firstHealth = true;
setInterval(() => progress++, 10);
const server = http.createServer((request, response) => {
  if (request.url === '/api/health' && firstHealth) {
    firstHealth = false;
    process.send({ healthPending: true });
    return;
  }
  response.setHeader('content-type', 'application/json');
  response.end(JSON.stringify({ pid: process.pid, progress }));
});
server.listen(0, '127.0.0.1', () => process.send({ port: server.address().port }));`
      ],
      { stdio: ['ignore', 'ignore', 'inherit', 'ipc'] }
    )

    const forwards: Awaited<ReturnType<typeof openForward>>[] = []

    const stopChild = async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
        return
      }

      const exited = once(child, 'exit')
      child.kill('SIGTERM')
      await exited
    }

    try {
      const [{ port }] = (await once(child, 'message')) as [{ port: number }]
      const first = await openForward(port)
      forwards.push(first)
      const before = await fetch(`${first.baseUrl}/progress`).then(response => response.json())
      const key = 'conn:vps::default'
      const stale = { baseUrl: first.baseUrl, mode: 'remote' }
      const original = Promise.resolve(stale)
      const pool = new Map([[key, { process: null, connectionPromise: original }]])
      const cancelForward = vi.fn(() => first.close())
      const close = vi.fn(async () => {})
      const state = { ssh: { cancelForward, close }, localPort: first.localPort, remotePort: port }
      const healthPending = once(child, 'message')
      const abort = new AbortController()

      const recovery = ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: original,
        currentConnectionPromise: () => pool.get(key)?.connectionPromise ?? null,
        probe: async (connection, path) => {
          await fetch(`${connection.baseUrl}${path}`, { signal: abort.signal })
        },
        retire: () =>
          recoverRemotePoolBackend(key, {
            pool,
            cancelAndWait: async (_key, disconnect) => disconnect(),
            disconnectTransport: () => teardownSshTransport(state)
          }),
        reconnect: async () => {
          const forward = await openForward(port)
          forwards.push(forward)
          const replacement = { baseUrl: forward.baseUrl, mode: 'remote' }
          pool.set(key, { process: null, connectionPromise: Promise.resolve(replacement) })

          return replacement
        }
      })

      await healthPending
      abort.abort(new Error('Timed out connecting to Hermes backend after 10000ms'))
      const replacement = await recovery
      expect(replacement.baseUrl).not.toBe(stale.baseUrl)
      expect(cancelForward).toHaveBeenCalledWith(first.localPort, port)
      expect(close).toHaveBeenCalledOnce()
      expect(child.exitCode).toBeNull()
      expect(child.signalCode).toBeNull()

      await expect
        .poll(async () => {
          const current = await fetch(`${replacement.baseUrl}/progress`).then(response => response.json())
          expect(current.pid).toBe(before.pid)

          return current.progress
        })
        .toBeGreaterThan(before.progress)

      // Deliberate disposal retains the stronger contract: stop the remote
      // process while its transport is available, then close the transport.
      const events: string[] = []
      await teardownSshState(
        {
          ...state,
          ssh: {
            open: async () => {
              events.push('open')
            },
            cancelForward: async () => {
              events.push('forward')
            },
            close: async () => {
              events.push('transport')
            }
          }
        },
        {
          cleanupRemote: async () => {
            events.push('remote')
            await stopChild()
          }
        }
      )
      expect(events).toEqual(['open', 'remote', 'forward', 'transport'])
      expect(child.signalCode).toBe('SIGTERM')
    } finally {
      await Promise.all(forwards.map(forward => forward.close()))
      await stopChild()
    }
  })

  it('reuses the same owned backend after another profile replaces the registry token', async () => {
    const ownershipId = '0123456789abcdef0123456789abcdef'
    const spawnNonce = '0123456789abcdef'
    const token = 'default-profile-token'
    const siblingToken = 'csm-profile-token'
    const state = { fingerprint: 'vps-default-config', token }

    const lock = {
      schemaVersion: LOCKFILE_SCHEMA_VERSION,
      protocolVersion: PROTOCOL_VERSION,
      ownershipId,
      spawnNonce,
      pid: 333,
      port: 40000,
      profile: '',
      hermesPath: '~/.local/bin/hermes',
      hermesHome: '~/.hermes',
      logPath: spawnLogPath(ownershipId, spawnNonce),
      tokenFingerprint: fingerprintToken(token),
      startedAt: '2026-07-14T00:00:00.000Z',
      creationTime: 'linux:123456'
    }

    const ssh = {
      exec: vi.fn(async (command: string) => {
        if (command.includes('.hermes-update-in-progress')) {
          return 'CLEAR'
        }

        if (command.includes('uname')) {
          return 'Linux\nx86_64'
        }

        if (command.includes('[ -x')) {
          return 'OK'
        }

        if (/cat .*lock\.json/.test(command)) {
          return JSON.stringify(lock)
        }

        if (command.includes('kill -0')) {
          return 'ALIVE'
        }

        if (command.includes('print("OWNED"')) {
          return 'OWNED\n'
        }

        return ''
      })
    }

    const proof = vi.fn(async (_baseUrl, servedToken, nonce) => {
      expect(servedToken).toBe(token)
      expect(nonce).toBe(spawnNonce)

      return 'authenticated-ok'
    })

    const result = await connect({
      ssh,
      ownershipId,
      profile: '',
      reuseToken: sshRecoveryToken(state, state.fingerprint, siblingToken),
      forward: async () => {},
      cancelForward: async () => {},
      pickLocalPort: async () => 50001,
      waitForHermes: async () => {},
      probeReuseProof: proof,
      adoptServedToken: async (_baseUrl, servedToken) => servedToken
    })

    expect(result.reused).toBe(true)
    expect(result.pid).toBe(lock.pid)
    expect(result.token).toBe(token)
    expect(proof).toHaveBeenCalledOnce()
    expect(sshRecoveryToken(state, 'different-host-config', siblingToken)).toBe(siblingToken)
    expect(sshRecoveryToken(undefined, state.fingerprint, siblingToken)).toBe(siblingToken)
  })

  it('evicts the remote descriptor immediately and drains bootstrap before disconnecting', async () => {
    const key = 'conn:vps::default'
    const pool = new Map([[key, { process: null }]])
    const drained = deferred()
    const events: string[] = []

    const recovery = recoverRemotePoolBackend(key, {
      pool,
      cancelAndWait: async (scope, disconnect) => {
        expect(scope).toBe(key)
        events.push('cancel')
        await drained.promise
        events.push('drained')
        await disconnect()
      },
      disconnectTransport: async scope => {
        expect(scope).toBe(key)
        events.push('disconnect')
      }
    })

    expect(pool.has(key)).toBe(false)
    expect(events).toEqual(['cancel'])
    drained.resolve()
    await recovery
    expect(events).toEqual(['cancel', 'drained', 'disconnect'])
  })

  it('does not remove or disconnect a local child', async () => {
    const key = 'local::default'
    const entry = { process: {} }
    const pool = new Map([[key, entry]])
    const cancelAndWait = vi.fn()
    const disconnectTransport = vi.fn()

    await recoverRemotePoolBackend(key, { pool, cancelAndWait, disconnectTransport })
    expect(pool.get(key)).toBe(entry)
    expect(cancelAndWait).not.toHaveBeenCalled()
    expect(disconnectTransport).not.toHaveBeenCalled()
  })

  it('does not disconnect a replacement when an old dispatch probe fails late', async () => {
    const key = 'conn:vps::default'
    const original = Promise.resolve({ baseUrl: 'http://127.0.0.1:4001', mode: 'remote' })
    const pool = new Map([[key, { process: null, connectionPromise: original }]])
    const probeStarted = deferred()
    const probeFinished = deferred()
    const disconnectTransport = vi.fn()
    const replacement = { baseUrl: 'http://127.0.0.1:4002', mode: 'remote' }

    const pending = ensureHealthyPooledRemoteBackendForDispatch({
      connectionPromise: original,
      currentConnectionPromise: () => pool.get(key)?.connectionPromise ?? null,
      probe: async () => {
        probeStarted.resolve()
        await probeFinished.promise
        throw new Error('old tunnel timed out')
      },
      retire: () =>
        recoverRemotePoolBackend(key, {
          pool,
          cancelAndWait: async (_key, disconnect) => disconnect(),
          disconnectTransport
        }),
      reconnect: async () => replacement
    })

    await probeStarted.promise
    const entry = { process: null, connectionPromise: Promise.resolve(replacement) }
    pool.set(key, entry)
    probeFinished.resolve()
    await expect(pending).resolves.toBe(replacement)
    expect(pool.get(key)).toBe(entry)
    expect(disconnectTransport).not.toHaveBeenCalled()
  })
})
