import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, test, vi } from 'vitest'

import { createPreviewWatchRegistry } from './preview-watch'

// Fake FSWatcher. Faithful to real Node semantics where it matters:
// EventEmitter-based (so emitting 'error' with no listener would throw,
// exactly like the uncaught-exception crash this fix targets) and SILENT
// after close() (a real FSWatcher delivers no events once closed). `closed`
// is fake bookkeeping only — real FSWatcher exposes no such property.
function fakeWatchImpl() {
  const created: any[] = []

  const impl = (dir: string, listener: (...args: any[]) => void) => {
    const watcher: any = new EventEmitter()
    watcher.dir = dir
    watcher.closed = false

    watcher.close = () => {
      watcher.closed = true
    }

    watcher.emitChange = (filename: any) => {
      if (!watcher.closed) {
        listener('change', filename)
      }
    }

    watcher.emitRename = (filename: any) => {
      if (!watcher.closed) {
        listener('rename', filename)
      }
    }

    created.push(watcher)

    return watcher
  }

  return { impl, created }
}

function makeRegistry(overrides: any = {}) {
  const sent: any[] = []
  const warnings: any[] = []
  const { impl, created } = fakeWatchImpl()

  const registry = createPreviewWatchRegistry({
    fileExists: () => true,
    sendChanged: (payload: any) => sent.push(payload),
    debounceMs: 120,
    watchImpl: impl,
    log: (...args: any[]) => warnings.push(args),
    ...overrides
  })

  return { registry, sent, warnings, created }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

test('change to the watched file debounces into one sendChanged with the watch id', () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = registry.watch('/tmp/preview/note.md')
  const watcher = created[0]

  watcher.emitChange('note.md')
  watcher.emitChange('note.md')
  assert.equal(sent.length, 0)

  vi.advanceTimersByTime(200)
  assert.deepEqual(sent, [{ id, path: '/tmp/preview/note.md' }])
})

test('rename events (atomic save-by-rename) trigger a reload', () => {
  const { registry, sent, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  created[0].emitRename('note.md')

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
})

test('changes to sibling files in the same directory are ignored', () => {
  const { registry, sent, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  created[0].emitChange('other.md')
  created[0].emitChange('note.md.tmp')

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('null filename (documented fs.watch behavior) is treated as a match — reload beats a missed save', () => {
  const { registry, sent, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  created[0].emitChange(null)

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
})

test('filename delivered as a Buffer still matches the target', () => {
  const { registry, sent, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  created[0].emitChange(Buffer.from('note.md'))

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
})

test('a deleted target file does not send', () => {
  const { registry, sent, created } = makeRegistry({
    fileExists: () => false
  })

  registry.watch('/tmp/preview/note.md')
  created[0].emitChange('note.md')

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('watch attaches an error listener synchronously (crash-prevention invariant)', () => {
  const { registry, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')

  // The whole point of the fix: an FSWatcher that emits 'error' with no
  // listener raises an uncaught exception and crashes the main process.
  assert.ok(created[0].listenerCount('error') > 0)
})

test('watcher error does not throw: watch is closed, deregistered, and stops sending', () => {
  const { registry, sent, warnings, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  const watcher = created[0]
  assert.equal(registry.size(), 1)

  // A deleted/unmounted watch dir surfaces as an 'error' event on the
  // FSWatcher. Without a listener Node would crash the process; the
  // registry must absorb it and tear down just this watch.
  watcher.emit('error', new Error('ENOENT: no such file or directory, watch'))

  assert.equal(registry.size(), 0)
  assert.equal(watcher.closed, true)
  assert.equal(warnings.length, 1)

  watcher.emitChange('note.md')
  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('a second error event is a no-op (no duplicate teardown or log)', () => {
  const { registry, warnings, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')

  created[0].emit('error', new Error('first'))
  assert.doesNotThrow(() => created[0].emit('error', new Error('second')))

  assert.equal(registry.size(), 0)
  assert.equal(warnings.length, 1)
})

test('stop after an error returns false and does not double-close', () => {
  const { registry, created } = makeRegistry()

  const { id } = registry.watch('/tmp/preview/note.md')
  created[0].emit('error', new Error('EPERM'))

  assert.equal(registry.stop(id), false)
  assert.equal(created[0].closed, true)
})

test('watcher error cancels a pending debounce timer', () => {
  const { registry, sent, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  const watcher = created[0]

  watcher.emitChange('note.md')
  watcher.emit('error', new Error('EPERM'))

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('a watchImpl that throws synchronously logs and rethrows (renderer sees a rejected invoke, never a crash)', () => {
  const { registry, warnings } = makeRegistry({
    watchImpl: () => {
      throw new Error('ENOENT')
    }
  })

  assert.throws(() => registry.watch('/tmp/preview/note.md'), /ENOENT/)
  assert.equal(warnings.length, 1)
  assert.equal(registry.size(), 0)
})

test('two watches of the same directory are independent', () => {
  const { registry, sent, created } = makeRegistry()

  const a = registry.watch('/tmp/preview/note.md')
  registry.watch('/tmp/preview/other.md')
  assert.equal(created.length, 2)

  created[0].emitChange('note.md')
  vi.advanceTimersByTime(200)

  assert.deepEqual(sent, [{ id: a.id, path: '/tmp/preview/note.md' }])
})

test('stop closes the watcher and reports unknown ids as not found', () => {
  const { registry, created } = makeRegistry()

  const { id } = registry.watch('/tmp/preview/note.md')
  assert.equal(registry.stop(id), true)
  assert.equal(created[0].closed, true)
  assert.equal(registry.stop(id), false)
  assert.equal(registry.stop('never-existed'), false)
})

test('stop before the debounce fires suppresses the send', () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = registry.watch('/tmp/preview/note.md')
  created[0].emitChange('note.md')
  registry.stop(id)

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('closeAll tears down every registered watch', () => {
  const { registry, created } = makeRegistry()

  registry.watch('/tmp/preview/a.md')
  registry.watch('/tmp/preview/b.md')
  assert.equal(registry.size(), 2)

  registry.closeAll()

  assert.equal(registry.size(), 0)
  assert.equal(created[0].closed, true)
  assert.equal(created[1].closed, true)
})

// ---------------------------------------------------------------------------
// Directory watching (watchDirectory) — the plugins-door path.
// Same error containment, debounce, and lifecycle as file watches.
// ---------------------------------------------------------------------------

test('watchDirectory debounces changes and sends the directory path', () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = registry.watchDirectory('/tmp/plugins', { dirExists: () => true })

  // Two rapid changes should coalesce into one debounced send
  created[0].emitChange('new-plugin')
  created[0].emitChange('another-plugin')

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
  assert.equal(sent[0].id, id)
  assert.equal(sent[0].path, '/tmp/plugins')
})

test('watchDirectory error tears down that watch without crashing', () => {
  const { registry, warnings, created } = makeRegistry()

  const { id } = registry.watchDirectory('/tmp/plugins', { dirExists: () => true })
  assert.equal(registry.size(), 1)

  created[0].emit('error', new Error('directory deleted'))

  assert.equal(registry.size(), 0)
  assert.equal(created[0].closed, true)
  assert.ok(warnings.length >= 1)

  // A second error must be a no-op (guard)
  created[0].emit('error', new Error('double error'))
  assert.equal(warnings.length, 1)

  // stop() on the torn-down watch is a no-op
  assert.equal(registry.stop(id), false)
})

test('watchDirectory stop tears down and suppresses pending debounce', () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = registry.watchDirectory('/tmp/plugins', { dirExists: () => true })
  created[0].emitChange('new-plugin')

  assert.equal(registry.stop(id), true)
  assert.equal(created[0].closed, true)

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('watchDirectory dirExists=false suppresses the send after debounce', () => {
  const { registry, sent, created } = makeRegistry()

  registry.watchDirectory('/tmp/plugins', { dirExists: () => false })
  created[0].emitChange('plugin-added')

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 0)
})

test('watchDirectory closeAll reaps directory watches alongside file watches', () => {
  const { registry, created } = makeRegistry()

  registry.watch('/tmp/preview/note.md')
  registry.watchDirectory('/tmp/plugins', { dirExists: () => true })
  assert.equal(registry.size(), 2)

  registry.closeAll()

  assert.equal(registry.size(), 0)
  assert.equal(created[0].closed, true)
  assert.equal(created[1].closed, true)
})
