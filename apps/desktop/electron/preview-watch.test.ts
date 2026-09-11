import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPreviewWatchRegistry } from './preview-watch'
import type { PreviewWatchImpl, PreviewWatchPayload } from './preview-watch'

// Fake FSWatcher, faithful to real Node semantics where it matters: an
// EventEmitter, and SILENT after close() (a real FSWatcher delivers no events
// once closed). `closed` is fake bookkeeping only — a real FSWatcher exposes
// no such property.
interface FakeWatcher extends EventEmitter {
  close: () => void
  closed: boolean
  emitChange: (filename: Buffer | null | string) => void
  emitRename: (filename: Buffer | null | string) => void
}

function fakeWatchImpl() {
  const created: FakeWatcher[] = []

  const impl: PreviewWatchImpl = (_dirPath, listener) => {
    const watcher = new EventEmitter() as FakeWatcher

    watcher.closed = false

    watcher.close = () => {
      watcher.closed = true
    }

    watcher.emitChange = filename => {
      if (!watcher.closed) {
        listener('change', filename)
      }
    }

    watcher.emitRename = filename => {
      if (!watcher.closed) {
        listener('rename', filename)
      }
    }

    created.push(watcher)

    return watcher
  }

  return { created, impl }
}

function makeRegistry(overrides: Partial<Parameters<typeof createPreviewWatchRegistry>[0]> = {}) {
  const sent: PreviewWatchPayload[] = []
  const { created, impl } = fakeWatchImpl()

  const registry = createPreviewWatchRegistry({
    fileExists: () => true,
    sendChanged: payload => sent.push(payload),
    debounceMs: 120,
    watchImpl: impl,
    ...overrides
  })

  return { created, registry, sent }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('createPreviewWatchRegistry — file watches', () => {
  it('debounces a burst of changes into one sendChanged with the watch id and path', () => {
    const { created, registry, sent } = makeRegistry()

    const { id } = registry.watch('/tmp/preview/note.md')

    created[0].emitChange('note.md')
    created[0].emitChange('note.md')
    expect(sent).toHaveLength(0)

    vi.advanceTimersByTime(200)
    expect(sent).toEqual([{ id, path: '/tmp/preview/note.md' }])
  })

  it('rename events (atomic save-by-rename) trigger a reload', () => {
    const { created, registry, sent } = makeRegistry()

    registry.watch('/tmp/preview/note.md')
    created[0].emitRename('note.md')

    vi.advanceTimersByTime(200)
    expect(sent).toHaveLength(1)
  })

  it('changes to sibling files in the same directory are ignored', () => {
    const { created, registry, sent } = makeRegistry()

    registry.watch('/tmp/preview/note.md')
    created[0].emitChange('other.md')
    created[0].emitChange('note.md.tmp')

    vi.advanceTimersByTime(1000)
    expect(sent).toHaveLength(0)
  })

  it('null filename (documented fs.watch behavior) is treated as a match — a reload beats a missed save', () => {
    const { created, registry, sent } = makeRegistry()

    registry.watch('/tmp/preview/note.md')
    created[0].emitChange(null)

    vi.advanceTimersByTime(200)
    expect(sent).toHaveLength(1)
  })

  it('a deleted target file does not send', () => {
    const { created, registry, sent } = makeRegistry({ fileExists: () => false })

    registry.watch('/tmp/preview/note.md')
    created[0].emitChange('note.md')

    vi.advanceTimersByTime(1000)
    expect(sent).toHaveLength(0)
  })

  it('a second change resets the debounce window', () => {
    const { created, registry, sent } = makeRegistry()

    registry.watch('/tmp/preview/note.md')

    created[0].emitChange('note.md')
    vi.advanceTimersByTime(100)
    created[0].emitChange('note.md')
    vi.advanceTimersByTime(100)
    expect(sent).toHaveLength(0)

    vi.advanceTimersByTime(100)
    expect(sent).toHaveLength(1)
  })

  it('two watches of the same directory are independent and keep their own ids', () => {
    const { created, registry, sent } = makeRegistry()

    const a = registry.watch('/tmp/preview/note.md')
    const b = registry.watch('/tmp/preview/other.md')

    expect(created).toHaveLength(2)
    expect(a.id).not.toBe(b.id)

    created[0].emitChange('note.md')
    vi.advanceTimersByTime(200)

    expect(sent).toEqual([{ id: a.id, path: '/tmp/preview/note.md' }])
  })

  it('stop closes the watcher, suppresses a pending debounce, and reports unknown ids as false', () => {
    const { created, registry, sent } = makeRegistry()

    const { id } = registry.watch('/tmp/preview/note.md')

    created[0].emitChange('note.md')
    expect(registry.stop(id)).toBe(true)
    expect(created[0].closed).toBe(true)

    vi.advanceTimersByTime(1000)
    expect(sent).toHaveLength(0)

    expect(registry.stop(id)).toBe(false)
    expect(registry.stop('never-existed')).toBe(false)
  })

  it('closeAll tears down every registered watch', () => {
    const { created, registry } = makeRegistry()

    registry.watch('/tmp/preview/a.md')
    registry.watch('/tmp/preview/b.md')
    expect(registry.size()).toBe(2)

    registry.closeAll()

    expect(registry.size()).toBe(0)
    expect(created[0].closed).toBe(true)
    expect(created[1].closed).toBe(true)
  })
})

describe('createPreviewWatchRegistry — directory watches', () => {
  it('debounces directory churn and sends the directory path', () => {
    const { created, registry, sent } = makeRegistry()

    const { id } = registry.watchDirectory('/tmp/plugins')

    created[0].emitChange('new-plugin')
    created[0].emitChange('another-plugin')

    vi.advanceTimersByTime(200)
    expect(sent).toEqual([{ id, path: '/tmp/plugins' }])
  })

  it('stop suppresses a pending directory debounce', () => {
    const { created, registry, sent } = makeRegistry()

    const { id } = registry.watchDirectory('/tmp/plugins')
    created[0].emitChange('new-plugin')

    expect(registry.stop(id)).toBe(true)
    expect(created[0].closed).toBe(true)

    vi.advanceTimersByTime(1000)
    expect(sent).toHaveLength(0)
  })

  it('closeAll reaps directory watches alongside file watches', () => {
    const { created, registry } = makeRegistry()

    registry.watch('/tmp/preview/note.md')
    registry.watchDirectory('/tmp/plugins')
    expect(registry.size()).toBe(2)

    registry.closeAll()

    expect(registry.size()).toBe(0)
    expect(created[0].closed).toBe(true)
    expect(created[1].closed).toBe(true)
  })
})
