import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import readline from 'node:readline'

import { afterEach } from 'vitest'

import type { DurableSendOwner } from './durable-send-types'

const roots: string[] = []
const peers: ChildProcessWithoutNullStreams[] = []
afterEach(async () => {
  for (const child of peers.splice(0)) {
    await new Promise<void>(resolve => {
      if (child.exitCode !== null || child.signalCode !== null) {
        resolve()

        return
      }

      const timer = setTimeout(() => child.kill(), 3000)
      child.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
      child.stdin.end()
    })
  }

  for (const root of roots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

export function createDurableBackendPeer() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-real-owner-'))
  roots.push(root)
  const home = path.join(root, 'profile')
  fs.mkdirSync(home)
  const repo = path.resolve(import.meta.dirname, '../../..')

  const child = spawn(process.env.HERMES_PYTHON!, [path.join(repo, 'tests/gateway/fixtures/durable_desktop_peer.py')], {
    cwd: repo,
    env: {
      PATH: process.env.PATH!,
      HOME: root,
      HERMES_HOME: home,
      PYTHONPATH: repo,
      PYTHONUNBUFFERED: '1',
      HERMES_TEST_ISOLATION: '1',
      LANG: 'C.UTF-8',
      TZ: 'UTC'
    },
    stdio: 'pipe'
  })

  peers.push(child)
  let stderr = ''
  child.stderr.on('data', data => {
    stderr += data
  })
  let sequence = 0
  const pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>()
  readline.createInterface({ input: child.stdout }).on('line', line => {
    const result = JSON.parse(line)
    const waiting = pending.get(result.id)
    pending.delete(result.id)

    if (result.error) {
      waiting?.reject(new Error(JSON.stringify(result.error)))
    } else {
      waiting?.resolve(result.result)
    }
  })
  child.on('exit', () => {
    for (const waiting of pending.values()) {
      waiting.reject(new Error(stderr))
    }

    pending.clear()
  })

  const raw = (method: string, params: Record<string, unknown> = {}): Promise<any> =>
    new Promise((resolve, reject) => {
      const id = ++sequence
      pending.set(id, { resolve, reject })
      child.stdin.write(JSON.stringify({ id, method, params }) + '\n')
    })

  let online = true
  let drop = ''
  const calls: Array<{ owner: unknown; method: string; params: unknown }> = []

  const rpc = async (
    owner: Pick<DurableSendOwner, 'connectionId' | 'profile'>,
    method: string,
    params: Record<string, unknown>
  ) => {
    if (!online) {
      throw new Error('Fixture disconnected before sending')
    }

    calls.push({ owner: { ...owner }, method, params })
    const wire = { ...params, profile: owner.profile }
    const result = await raw(method, wire)

    if (method === drop) {
      drop = ''
      throw new Error('Fixture discarded committed reply')
    }

    return result
  }

  return {
    root,
    raw,
    rpc,
    calls,
    online: (value: boolean) => {
      online = value
    },
    drop: (method: string) => {
      drop = method
    }
  }
}
