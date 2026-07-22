import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { cleanupFailedPoolBackend, terminatePoolBackendEntry, watchPoolBackendChild } from './backend-pool-lifecycle'

interface FakeEntry {
  process: FakeChild | null
}

class FakeChild extends EventEmitter {
  readonly name: string

  constructor(name: string) {
    super()
    this.name = name
  }
}

function watch(pool: Map<string, FakeEntry>, profile: string, entry: FakeEntry) {
  const events: string[] = []

  watchPoolBackendChild(pool, profile, entry, entry.process, {
    onError: error => events.push(`error:${error.message}`),
    onExit: (code, signal) => events.push(`exit:${signal || code}`)
  })

  return events
}

test('an old child exit preserves the replacement pool entry', () => {
  const profile = 'work'
  const oldEntry = { process: new FakeChild('old') }
  const replacement = { process: new FakeChild('replacement') }
  const pool = new Map([[profile, oldEntry]])
  const events = watch(pool, profile, oldEntry)

  pool.set(profile, replacement)
  oldEntry.process.emit('exit', 0, null)

  assert.equal(pool.get(profile), replacement)
  assert.deepEqual(events, ['exit:0'])
})

test('an old child error preserves the replacement pool entry', () => {
  const profile = 'work'
  const oldEntry = { process: new FakeChild('old') }
  const replacement = { process: new FakeChild('replacement') }
  const pool = new Map([[profile, oldEntry]])
  const events = watch(pool, profile, oldEntry)

  pool.set(profile, replacement)
  oldEntry.process.emit('error', new Error('late spawn error'))

  assert.equal(pool.get(profile), replacement)
  assert.deepEqual(events, ['error:late spawn error'])
})

test('spawn failure removes its owned entry and terminates the spawned child', async () => {
  const profile = 'work'
  const child = new FakeChild('spawned')
  const entry = { process: child }
  const pool = new Map([[profile, entry]])
  const calls: string[] = []

  await cleanupFailedPoolBackend(pool, profile, entry, {
    stopBackendChild: process => calls.push(`term:${process?.name}`),
    waitForBackendExit: async process => {
      calls.push(`wait:${process?.name}`)
    }
  })

  assert.equal(pool.has(profile), false)
  assert.deepEqual(calls, ['term:spawned', 'wait:spawned'])
})

test('spawn failure terminates its child without removing a replacement entry', async () => {
  const profile = 'work'
  const child = new FakeChild('failed')
  const failedEntry = { process: child }
  const replacement = { process: new FakeChild('replacement') }
  const pool = new Map([[profile, replacement]])
  const terminated: FakeChild[] = []

  await cleanupFailedPoolBackend(pool, profile, failedEntry, {
    stopBackendChild: process => {
      if (process) {
        terminated.push(process)
      }
    },
    waitForBackendExit: async () => {}
  })

  assert.equal(pool.get(profile), replacement)
  assert.deepEqual(terminated, [child])
})

test('pool termination requests graceful stop before waiting for bounded escalation', async () => {
  const child = new FakeChild('pooled')
  const calls: string[] = []

  await terminatePoolBackendEntry(
    { process: child },
    {
      stopBackendChild: process => calls.push(`term:${process?.name}`),
      waitForBackendExit: async process => {
        calls.push(`wait:${process?.name}`)
      }
    }
  )

  assert.deepEqual(calls, ['term:pooled', 'wait:pooled'])
})
