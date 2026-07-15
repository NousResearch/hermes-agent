import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { spawnValidatedWindowsUpdater, waitForUpdaterSpawn } from './updater-handoff'

class FakeChild extends EventEmitter {
  unrefCalls = 0

  unref() {
    this.unrefCalls += 1

    return this
  }
}

test('waitForUpdaterSpawn rejects an asynchronous spawn error', async () => {
  const child = new FakeChild()
  const pending = waitForUpdaterSpawn(child, { timeoutMs: 100 })
  const error = new Error('spawn hermes-setup.exe ENOENT')

  child.emit('error', error)

  await assert.rejects(pending, error)
})

test('spawnValidatedWindowsUpdater accepts and detaches a spawned updater', async () => {
  const child = new FakeChild()
  const spawnCalls: Array<{ command: string; args: readonly string[]; cwd?: string }> = []

  const pending = spawnValidatedWindowsUpdater('hermes-setup.exe', ['--update'], { cwd: 'C:\\Hermes' }, {
    timeoutMs: 100,
    spawnImpl: (command, args, options) => {
      spawnCalls.push({ command, args, cwd: options.cwd as string })

      return child as never
    }
  })

  child.emit('spawn')

  assert.equal(await pending, child)
  assert.deepEqual(spawnCalls, [
    { command: 'hermes-setup.exe', args: ['--update'], cwd: 'C:\\Hermes' }
  ])
  assert.equal(child.unrefCalls, 1)
})

test('spawnValidatedWindowsUpdater does not detach after an emitted spawn error', async () => {
  const child = new FakeChild()

  const pending = spawnValidatedWindowsUpdater('hermes-setup.exe', ['--repair'], {}, {
    timeoutMs: 100,
    spawnImpl: () => child as never
  })

  child.emit('error', new Error('spawn hermes-setup.exe EACCES'))

  await assert.rejects(pending, /EACCES/)
  assert.equal(child.unrefCalls, 0)
})

test('waitForUpdaterSpawn retains the detached-child timeout fallback', async () => {
  const child = new FakeChild()

  await waitForUpdaterSpawn(child, { timeoutMs: 1 })

  assert.equal(child.listenerCount('spawn'), 0)
  assert.equal(child.listenerCount('error'), 0)
})
