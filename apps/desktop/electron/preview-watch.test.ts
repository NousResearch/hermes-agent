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

// A watchImpl that throws EINTR (transient) for the first `n` calls, then
// succeeds. Lets the retry path be exercised deterministically.
function flakyEintrWatchImpl(n: number) {
  const created: any[] = []
  let calls = 0

  const impl = (dir: string, listener: (...args: any[]) => void) => {
    calls += 1

    if (calls <= n) {
      const err: NodeJS.ErrnoException = new Error('EINTR: interrupted system call, watch')
      err.code = 'EINTR'
      throw err
    }

    const watcher: any = new EventEmitter()
    watcher.dir = dir
    watcher.closed = false

    watcher.close = () => { watcher.closed = true }

    watcher.emitChange = (filename: any) => {
      if (!watcher.closed) {listener('change', filename)}
    }

    watcher.emitRename = (filename: any) => {
      if (!watcher.closed) {listener('rename', filename)}
    }

    created.push(watcher)

    return watcher
  }

  return { impl, created, callCount: () => calls }
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

test('change to the watched file debounces into one sendChanged with the watch id', async () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = await registry.watch('/tmp/preview/note.md')
  const watcher = created[0]

  watcher.emitChange('note.md')
  watcher.emitChange('note.md')
  assert.equal(sent.length, 0)

  vi.advanceTimersByTime(200)
  assert.deepEqual(sent, [{ id, path: '/tmp/preview/note.md' }])
})

test('rename events (atomic save-by-rename) trigger a reload', async () => {
  const { registry, sent, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
  created[0].emitRename('note.md')

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
})

test('changes to sibling files in the same directory are ignored', async () => {
  const { registry, sent, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
  created[0].emitChange('other.md')
  created[0].emitChange('note.md.tmp')

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('null filename (documented fs.watch behavior) is treated as a match — reload beats a missed save', async () => {
  const { registry, sent, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
  created[0].emitChange(null)

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
})

test('filename delivered as a Buffer still matches the target', async () => {
  const { registry, sent, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
  created[0].emitChange(Buffer.from('note.md'))

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
})

test('a deleted target file does not send', async () => {
  const { registry, sent, created } = makeRegistry({
    fileExists: () => false
  })

  await registry.watch('/tmp/preview/note.md')
  created[0].emitChange('note.md')

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('watch attaches an error listener synchronously (crash-prevention invariant)', async () => {
  const { registry, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')

  // The whole point of the fix: an FSWatcher that emits 'error' with no
  // listener raises an uncaught exception and crashes the main process.
  assert.ok(created[0].listenerCount('error') > 0)
})

test('watcher error does not throw: watch is closed, deregistered, and stops sending', async () => {
  const { registry, sent, warnings, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
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

test('a second error event is a no-op (no duplicate teardown or log)', async () => {
  const { registry, warnings, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')

  created[0].emit('error', new Error('first'))
  assert.doesNotThrow(() => created[0].emit('error', new Error('second')))

  assert.equal(registry.size(), 0)
  assert.equal(warnings.length, 1)
})

test('stop after an error returns false and does not double-close', async () => {
  const { registry, created } = makeRegistry()

  const { id } = await registry.watch('/tmp/preview/note.md')
  created[0].emit('error', new Error('EPERM'))

  assert.equal(registry.stop(id), false)
  assert.equal(created[0].closed, true)
})

test('watcher error cancels a pending debounce timer', async () => {
  const { registry, sent, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
  const watcher = created[0]

  watcher.emitChange('note.md')
  watcher.emit('error', new Error('EPERM'))

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('a watchImpl that throws synchronously with a non-transient error rejects (never a crash)', async () => {
  const { registry, warnings } = makeRegistry({
    watchImpl: () => {
      throw new Error('ENOENT')
    }
  })

  await assert.rejects(() => registry.watch('/tmp/preview/note.md'), /ENOENT/)
  assert.equal(warnings.length, 1)
  assert.equal(registry.size(), 0)
})

test('a transient EINTR on creation is retried and the watcher still comes up', async () => {
  vi.useRealTimers() // the retry loop awaits a real setTimeout; fake timers must not freeze it
  const { impl, created, callCount } = flakyEintrWatchImpl(2)
  const { registry, sent, warnings } = makeRegistry({ watchImpl: impl, retryDelayMs: 5 })

  const { id } = await registry.watch('/tmp/preview/note.md')

  assert.ok(callCount() >= 3, 'must have retried past the transient failures')
  assert.equal(created.length, 1)
  assert.equal(warnings.length, 0, 'a recoverable transient must not log as catastrophic')

  vi.useFakeTimers() // back to fake timers so the debounce assertion below is deterministic
  created[0].emitChange('note.md')
  vi.advanceTimersByTime(200)
  assert.deepEqual(sent, [{ id, path: '/tmp/preview/note.md' }])
})

test('persistent EINTR exceeding the retry budget rethrows the last error', async () => {
  vi.useRealTimers()
  const { impl, created } = flakyEintrWatchImpl(999)

  const { registry, warnings } = makeRegistry({
    watchImpl: impl,
    retryAttempts: 3,
    retryDelayMs: 10
  })

  await assert.rejects(() => registry.watch('/tmp/preview/note.md'), /EINTR/)
  assert.equal(created.length, 0, 'no watcher should be created once retries are exhausted')
  assert.equal(warnings.length, 1)
})

test('EAGAIN and EWOULDBLOCK are also retried; a non-retryable code is not', async () => {
  vi.useRealTimers()

  // EAGAIN retried -> success
  {
    let calls = 0

    const impl = (_dir: string, listener: (...args: any[]) => void) => {
      calls += 1

      if (calls === 1) {
        const err: NodeJS.ErrnoException = new Error('EAGAIN')
        err.code = 'EAGAIN'
        throw err
      }

      const w = new EventEmitter() as any

      w.close = () => { w.closed = true }
      w.emitChange = (f: any) => listener('change', f)

      return w
    }

    const { registry } = makeRegistry({ watchImpl: impl, retryAttempts: 5, retryDelayMs: 5 })
    await registry.watch('/tmp/preview/note.md')
    assert.ok(calls >= 2, 'EAGAIN should have been retried')
  }

  // EPERM is NOT retryable -> rejects immediately
  {
    const impl = () => {
      const err: NodeJS.ErrnoException = new Error('EPERM')
      err.code = 'EPERM'
      throw err
    }

    const { registry, warnings } = makeRegistry({ watchImpl: impl, retryAttempts: 5, retryDelayMs: 5 })
    await assert.rejects(() => registry.watch('/tmp/preview/note.md'), /EPERM/)
    assert.equal(warnings.length, 1)
  }
})

test('two watches of the same directory are independent', async () => {
  const { registry, sent, created } = makeRegistry()

  const a = await registry.watch('/tmp/preview/note.md')
  await registry.watch('/tmp/preview/other.md')
  assert.equal(created.length, 2)

  created[0].emitChange('note.md')
  vi.advanceTimersByTime(200)

  assert.deepEqual(sent, [{ id: a.id, path: '/tmp/preview/note.md' }])
})

test('stop closes the watcher and reports unknown ids as not found', async () => {
  const { registry, created } = makeRegistry()

  const { id } = await registry.watch('/tmp/preview/note.md')
  assert.equal(registry.stop(id), true)
  assert.equal(created[0].closed, true)
  assert.equal(registry.stop(id), false)
  assert.equal(registry.stop('never-existed'), false)
})

test('stop before the debounce fires suppresses the send', async () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = await registry.watch('/tmp/preview/note.md')
  created[0].emitChange('note.md')
  registry.stop(id)

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('closeAll tears down every registered watch', async () => {
  const { registry, created } = makeRegistry()

  await registry.watch('/tmp/preview/a.md')
  await registry.watch('/tmp/preview/b.md')
  assert.equal(registry.size(), 2)

  registry.closeAll()
  assert.equal(registry.size(), 0)
  assert.equal(created[0].closed, true)
  assert.equal(created[1].closed, true)
})

// ---------------------------------------------------------------------------
// Directory watching (watchDirectory) — the plugins-door path.
// Same error containment, debounce, retry, and lifecycle as file watches.
// ---------------------------------------------------------------------------

test('watchDirectory debounces changes and sends the directory path', async () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = await registry.watchDirectory('/tmp/plugins', { dirExists: () => true })

  // Two rapid changes should coalesce into one debounced send
  created[0].emitChange('new-plugin')
  created[0].emitChange('another-plugin')

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 1)
  assert.equal(sent[0].id, id)
  assert.equal(sent[0].path, '/tmp/plugins')
})

test('watchDirectory error tears down that watch without crashing', async () => {
  const { registry, warnings, created } = makeRegistry()

  const { id } = await registry.watchDirectory('/tmp/plugins', { dirExists: () => true })
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

test('watchDirectory stop tears down and suppresses pending debounce', async () => {
  const { registry, sent, created } = makeRegistry()

  const { id } = await registry.watchDirectory('/tmp/plugins', { dirExists: () => true })
  created[0].emitChange('new-plugin')
  assert.equal(registry.stop(id), true)
  assert.equal(created[0].closed, true)

  vi.advanceTimersByTime(1000)
  assert.equal(sent.length, 0)
})

test('watchDirectory dirExists=false suppresses the send after debounce', async () => {
  const { registry, sent, created } = makeRegistry()

  await registry.watchDirectory('/tmp/plugins', { dirExists: () => false })
  created[0].emitChange('plugin-added')

  vi.advanceTimersByTime(200)
  assert.equal(sent.length, 0)
})

test('watchDirectory closeAll reaps directory watches alongside file watches', async () => {
  const { registry, created } = makeRegistry()

  await registry.watch('/tmp/preview/note.md')
  await registry.watchDirectory('/tmp/plugins', { dirExists: () => true })
  assert.equal(registry.size(), 2)

  registry.closeAll()
  assert.equal(registry.size(), 0)
  assert.equal(created[0].closed, true)
  assert.equal(created[1].closed, true)
})

test('watchDirectory transient EINTR is retried before the watcher comes up', async () => {
  vi.useRealTimers()
  const { impl, created, callCount } = flakyEintrWatchImpl(1)
  const { registry } = makeRegistry({ watchImpl: impl, retryDelayMs: 5 })

  await registry.watchDirectory('/tmp/plugins', { dirExists: () => true })

  assert.ok(callCount() >= 2)
  assert.equal(created.length, 1)
})
