import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createSharedDirWatchRegistry } from './shared-dir-watch'

function fakeWatchHost() {
  const listeners: Array<(eventType: string, filename: string | null) => void> = []
  let closes = 0

  const registry = createSharedDirWatchRegistry({
    platform: 'darwin',
    watch: (_watchDir, listener) => {
      listeners.push(listener)

      return {
        close: () => {
          closes += 1
        }
      }
    }
  })

  return { closes: () => closes, listeners, registry }
}

const snapshotOf = (value: string) => () => value

test('one directory gets one handle no matter how many previews share it', () => {
  const { listeners, registry } = fakeWatchHost()
  const seen: string[][] = [[], []]

  const releaseA = registry.acquire('/docs', {
    onEvent: (_t, f) => {
      seen[0].push(String(f))
    },
    snapshot: snapshotOf('a'),
    onPollChange: () => {}
  })

  const releaseB = registry.acquire('/docs', {
    onEvent: (_t, f) => {
      seen[1].push(String(f))
    },
    snapshot: snapshotOf('b'),
    onPollChange: () => {}
  })

  assert.equal(registry.dirCount(), 1)
  assert.equal(listeners.length, 1)

  listeners[0]('rename', 'note.md')
  assert.deepEqual(seen, [['note.md'], ['note.md']])

  releaseA()
  assert.equal(registry.dirCount(), 1)
  releaseB()
  assert.equal(registry.dirCount(), 0)
})

test('a late subscriber does not resurrect a closed handle', () => {
  const { closes, listeners, registry } = fakeWatchHost()

  const release = registry.acquire('/docs', { onEvent: () => {}, snapshot: snapshotOf('a'), onPollChange: () => {} })
  release()
  release()
  assert.equal(closes(), 1)

  registry.acquire('/docs', { onEvent: () => {}, snapshot: snapshotOf('a'), onPollChange: () => {} })
  assert.equal(listeners.length, 2)
})

test('poll mode notifies only the subscriber whose snapshot changed', () => {
  const listeners: Array<(eventType: string, filename: string | null) => void> = []
  let poll: (() => void) | null = null
  const notified: string[] = []
  let snapA = 'a-v1'

  const registry = createSharedDirWatchRegistry({
    platform: 'win32',
    watch: (_watchDir, listener) => {
      listeners.push(listener)

      return { close: () => {} }
    },
    maxEvents: 2,
    windowMs: 60_000,
    setInterval: fn => {
      poll = fn as () => void

      return 1
    },
    clearInterval: () => {}
  })

  registry.acquire('/docs', {
    onEvent: () => {},
    snapshot: () => snapA,
    onPollChange: () => {
      notified.push('a')
    }
  })
  registry.acquire('/docs', {
    onEvent: () => {},
    snapshot: snapshotOf('b'),
    onPollChange: () => {
      notified.push('b')
    }
  })

  // Flood past the breaker: the handle closes and polling takes over.
  listeners[0]('rename', 'x')
  listeners[0]('rename', 'y')
  listeners[0]('rename', 'z')
  assert.ok(poll, 'breaker must have tripped into poll mode')

  // Only A's target changed on disk.
  snapA = 'a-v2'
  poll!()
  assert.deepEqual(notified, ['a'])

  // Nothing changed: silence.
  poll!()
  assert.deepEqual(notified, ['a'])
})
