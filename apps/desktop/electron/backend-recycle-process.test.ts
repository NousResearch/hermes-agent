import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, it } from 'vitest'

import { createScopedBackendRecycler, type ScopedBackendRecycleState } from './backend-recycle'
import { normalizeRegistry } from './connection-registry'
import { createPoolStopper } from './pool-stop'

it('restarts real isolated children while another gateway with the same profile and other local profiles remain alive', async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-restart-test-'))
  const children: ChildProcess[] = []
  const exited = (child: ChildProcess) => child.exitCode !== null || child.signalCode !== null

  const stop = async (child: ChildProcess) => {
    if (exited(child)) {
      return
    }

    const done = once(child, 'exit')
    child.kill()
    await done
  }

  const start = async (profile: string, source: string) => {
    const profileHome = path.join(home, source, profile)
    await mkdir(profileHome, { recursive: true })

    const child = spawn(
      process.execPath,
      [
        '-e',
        `
      const http = require('node:http');
      const server = http.createServer((req, res) => res.end(JSON.stringify({ pid: process.pid, home: process.env.HERMES_HOME })));
      server.listen(0, '127.0.0.1', () => process.stdout.write(String(server.address().port) + '\\n'));
    `
      ],
      { env: { ...process.env, HERMES_HOME: profileHome }, stdio: ['ignore', 'pipe', 'pipe'] }
    )

    children.push(child)
    const [data] = await once(child.stdout!, 'data')
    const baseUrl = `http://127.0.0.1:${String(data).trim()}`
    const connection = { mode: 'local', profile, connectionId: source, baseUrl, wsUrl: baseUrl.replace('http:', 'ws:') }
    const proof = (await (await fetch(baseUrl)).json()) as { home: string; pid: number }
    expect(proof).toEqual({ home: profileHome, pid: child.pid })

    return { process: child, connectionPromise: Promise.resolve(connection), connection }
  }

  try {
    const primary = await start('coder', 'local')
    const writer = await start('writer', 'local')
    const other = await start('writer', 'other')

    const pool = new Map([
      ['writer', writer],
      ['conn:other::writer', other]
    ])

    const snapshot: ScopedBackendRecycleState = {
      registry: normalizeRegistry({
        connections: [{ id: 'other', kind: 'remote', label: 'Other', url: other.connection.baseUrl }]
      }),
      routeOptions: { primaryProfile: 'coder' },
      primary,
      pool
    }

    const stopper = createPoolStopper({
      pool,
      stopChild: value => {
        ;(value as ChildProcess).kill()
      },
      waitForExit: async value => {
        const child = value as ChildProcess

        if (!exited(child)) {
          await once(child, 'exit')
        }
      }
    })

    const recycler = createScopedBackendRecycler({
      readState: () => snapshot,
      stopPool: stopper.stop,
      stopPrimary: async () => {
        const child = snapshot.primary.process as ChildProcess
        snapshot.primary = { process: null, connectionPromise: null }
        await stop(child)
      }
    })

    const restart = (profile: string) =>
      recycler.restart({ connectionId: 'local', profile }, async (target, slot) => {
        const next = await start(target.profile, target.connectionId)

        if (slot.primary) {
          snapshot.primary = next
        } else {
          pool.set(slot.key, next)
        }

        return next.connection
      })

    expect(await recycler.capability({ connectionId: 'other', profile: 'writer' })).toEqual({
      supported: false,
      reason: 'externally-managed'
    })
    await expect(
      recycler.restart({ connectionId: 'other', profile: 'writer' }, async () => {
        throw new Error('must not start')
      })
    ).rejects.toThrow('externally-managed')
    const freshWriter = (await restart('writer')) as typeof writer.connection
    expect(exited(writer.process)).toBe(true)
    expect((await (await fetch(freshWriter.baseUrl)).json()).pid).not.toBe(writer.process.pid)
    expect(exited(primary.process)).toBe(false)
    expect((await (await fetch(other.connection.baseUrl)).json()).pid).toBe(other.process.pid)
    const freshPrimary = (await restart('coder')) as typeof primary.connection
    expect(exited(primary.process)).toBe(true)
    expect((await (await fetch(freshPrimary.baseUrl)).json()).pid).not.toBe(primary.process.pid)
    expect((await (await fetch(freshWriter.baseUrl)).json()).pid).toBe(pool.get('writer')!.process.pid)
    expect((await (await fetch(other.connection.baseUrl)).json()).pid).toBe(other.process.pid)
  } finally {
    await Promise.all(children.map(stop))
    await rm(home, { recursive: true, force: true })
  }
}, 15_000)
