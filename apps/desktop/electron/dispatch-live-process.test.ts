import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { promisify } from 'node:util'

import { describe, expect, it, vi } from 'vitest'

import { remotePidAlive } from './remote-lifecycle'
import {
  ensureHealthyPooledRemoteBackendForDispatch,
  RemoteLivenessTracker,
  revalidatePooledRemoteBackends,
  revalidateRemoteConnection,
  revalidateSuspectPooledRemoteBackends
} from './remote-liveness'

const exec = promisify(execFile)

// Pause only our disposable server: its PID stays alive while its HTTP loop cannot reply.
describe.skipIf(process.platform === 'win32')('dispatch to a stalled live SSH backend', () => {
  it('keeps a stalled owner and recovers on the same PID', async () => {
    const child = spawn(
      process.execPath,
      [
        '-e',
        `
      require('node:http').createServer((req, res) => res.end('{}'))
        .listen(0, '127.0.0.1', function () { console.log(this.address().port) })
    `
      ],
      { stdio: ['ignore', 'pipe', 'pipe'] }
    )

    try {
      const [data] = await once(child.stdout, 'data')
      const baseUrl = `http://127.0.0.1:${String(data).trim()}`
      const connection = { mode: 'remote', baseUrl }
      const connectionPromise = Promise.resolve(connection)

      const probe = async (_connection: unknown, _path: string, options: { timeoutMs: number }) => {
        try {
          await fetch(`${baseUrl}/api/health`, { signal: AbortSignal.timeout(options.timeoutMs) })
        } catch {
          throw new Error('Timed out connecting to Hermes backend after 10000ms')
        }
      }

      const retire = vi.fn(() => {
        child.kill('SIGKILL')
      })

      const reconnect = vi.fn(async () => connection)

      const confirmDead = async () =>
        !(await remotePidAlive(
          {
            exec: async (command: string) => {
              const result = await exec('/bin/sh', ['-c', command], { timeout: 2_000 })

              return result.stdout
            }
          },
          child.pid
        ))

      const options = {
        connectionPromise,
        currentConnectionPromise: () => connectionPromise,
        probe,
        retire,
        reconnect,
        confirmDead
      }

      child.kill('SIGSTOP')
      await expect(ensureHealthyPooledRemoteBackendForDispatch(options)).rejects.toThrow('Timed out')
      expect(retire).not.toHaveBeenCalled()
      expect(reconnect).not.toHaveBeenCalled()
      expect(await confirmDead()).toBe(false)

      const unavailable = async () => {
        throw new Error('loop still stalled')
      }

      const resetConnection = vi.fn()
      await expect(
        revalidateRemoteConnection({
          connectionPromise,
          currentConnectionPromise: () => connectionPromise,
          probe: unavailable,
          resetConnection,
          confirmDead,
          tracker: new RemoteLivenessTracker(1),
          log: () => {}
        })
      ).resolves.toEqual({ ok: true, rebuilt: false })
      expect(resetConnection).not.toHaveBeenCalled()
      await expect(
        revalidatePooledRemoteBackends({
          entries: [['ssh', { connectionPromise, remoteBaseUrl: baseUrl }]],
          probe: unavailable,
          stopBackend: retire,
          confirmDead,
          tracker: new RemoteLivenessTracker(1),
          log: () => {}
        })
      ).resolves.toEqual({ dropped: [] })
      expect(retire).not.toHaveBeenCalled()
      await expect(
        revalidateSuspectPooledRemoteBackends({
          entries: [['ssh', { connectionPromise, remoteBaseUrl: baseUrl }]],
          probe: unavailable,
          retire,
          rebuild: reconnect,
          confirmDead,
          tracker: new RemoteLivenessTracker(1),
          log: () => {}
        })
      ).resolves.toEqual({ rebuilt: [], retired: [] })
      expect(retire).not.toHaveBeenCalled()
      child.kill('SIGCONT')
      await expect(
        ensureHealthyPooledRemoteBackendForDispatch({
          ...options,
          probe: async () => {
            await fetch(`${baseUrl}/api/health`)
          }
        })
      ).resolves.toBe(connection)
      expect(child.exitCode).toBeNull()
    } finally {
      child.kill('SIGCONT')

      if (child.exitCode === null) {
        child.kill('SIGKILL')
        await once(child, 'exit')
      }
    }
  })

  it('retires a confirmed dead owner but preserves one whose PID check fails', async () => {
    const connectionPromise = Promise.resolve({ mode: 'remote', baseUrl: 'http://127.0.0.1:1' })
    const replacement = { mode: 'remote', baseUrl: 'http://127.0.0.1:2' }
    const retire = vi.fn()
    const reconnect = vi.fn(async () => replacement)

    const options = {
      connectionPromise,
      currentConnectionPromise: () => connectionPromise,
      probe: async () => {
        throw new Error('ECONNREFUSED')
      },
      retire,
      reconnect
    }

    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        ...options,
        confirmDead: async () => {
          throw new Error('SSH probe unavailable')
        }
      })
    ).rejects.toThrow('SSH probe unavailable')
    expect(retire).not.toHaveBeenCalled()
    expect(reconnect).not.toHaveBeenCalled()
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({ ...options, confirmDead: async () => true })
    ).resolves.toBe(replacement)
    expect(retire).toHaveBeenCalledOnce()
    expect(reconnect).toHaveBeenCalledOnce()
  })
})

describe('owner replacement while confirming a failed probe', () => {
  it('never retires the replacement descriptor in dispatch, background, or wake recovery', async () => {
    const connection = { mode: 'remote', baseUrl: 'http://127.0.0.1:1' }
    const old = Promise.resolve(connection)
    const replacement = Promise.resolve({ ...connection, baseUrl: 'http://127.0.0.1:2' })
    let current = old

    const confirmDead = async () => {
      current = replacement

      return true
    }

    const retire = vi.fn()

    const probe = async () => {
      throw new Error('probe failed')
    }

    const reconnect = async () => replacement
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: old,
        currentConnectionPromise: () => current,
        confirmDead,
        probe,
        retire,
        reconnect
      })
    ).resolves.toEqual(await replacement)
    expect(retire).not.toHaveBeenCalled()
    current = old
    await expect(
      revalidatePooledRemoteBackends({
        entries: [['ssh', { connectionPromise: old, remoteBaseUrl: connection.baseUrl }]],
        probe,
        stopBackend: retire,
        confirmDead,
        currentConnectionPromise: () => current,
        tracker: new RemoteLivenessTracker(1),
        log: () => {}
      })
    ).resolves.toEqual({ dropped: [] })
    current = old
    await expect(
      revalidateSuspectPooledRemoteBackends({
        entries: [['ssh', { connectionPromise: old, remoteBaseUrl: connection.baseUrl }]],
        probe,
        retire,
        rebuild: reconnect,
        confirmDead,
        currentConnectionPromise: () => current,
        tracker: new RemoteLivenessTracker(1),
        log: () => {}
      })
    ).resolves.toEqual({ rebuilt: [], retired: [] })
    expect(retire).not.toHaveBeenCalled()
  })
})
