import { EventEmitter } from 'node:events'
import { pathToFileURL } from 'node:url'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPreviewWatchRegistry, PREVIEW_FILE_CHANGED_CHANNEL } from './preview-watch'
import type { PreviewFileChangedPayload, PreviewWatchImpl } from './preview-watch'

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

// Stands in for a window's WebContents: `send` records what was delivered.
interface FakeOwner {
  destroyed: boolean
  isDestroyed: () => boolean
  send: (channel: string, payload: PreviewFileChangedPayload) => void
  sent: Array<{ channel: string; payload: PreviewFileChangedPayload }>
}

function fakeOwner(): FakeOwner {
  const owner: FakeOwner = {
    destroyed: false,
    isDestroyed: () => owner.destroyed,
    send: (channel, payload) => {
      owner.sent.push({ channel, payload })
    },
    sent: []
  }

  return owner
}

function makeRegistry(overrides: Partial<Parameters<typeof createPreviewWatchRegistry>[0]> = {}) {
  const { created, impl } = fakeWatchImpl()

  const registry = createPreviewWatchRegistry({
    fileExists: () => true,
    debounceMs: 120,
    watchImpl: impl,
    ...overrides
  })

  return { created, registry }
}

const fileUrl = (filePath: string) => pathToFileURL(filePath).toString()

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('createPreviewWatchRegistry — delivery to the owning window', () => {
  it('delivers each change only to the window that created the watch, under that watch id', () => {
    const { created, registry } = makeRegistry()
    const secondary = fakeOwner()
    const main = fakeOwner()

    const a = registry.watch('/tmp/preview/note.md', secondary)
    const b = registry.watch('/tmp/preview/note.md', main)

    expect(a.id).not.toBe(b.id)

    created[0].emitChange('note.md')
    vi.advanceTimersByTime(200)

    expect(secondary.sent).toEqual([
      {
        channel: PREVIEW_FILE_CHANGED_CHANNEL,
        payload: { id: a.id, path: '/tmp/preview/note.md', url: fileUrl('/tmp/preview/note.md') }
      }
    ])
    // The other window's watch is independent: nothing foreign was delivered.
    expect(main.sent).toHaveLength(0)

    created[1].emitChange('note.md')
    vi.advanceTimersByTime(200)

    expect(main.sent).toEqual([
      {
        channel: PREVIEW_FILE_CHANGED_CHANNEL,
        payload: { id: b.id, path: '/tmp/preview/note.md', url: fileUrl('/tmp/preview/note.md') }
      }
    ])
    expect(secondary.sent).toHaveLength(1)
  })

  it('a destroyed owner gets no send and its watch is dropped, not leaked', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    const { id } = registry.watch('/tmp/preview/note.md', owner)

    // Window closed while the watch lived: the next fs tick must stop the
    // watch (no leak) and skip the send (nothing left to deliver to).
    owner.destroyed = true
    expect(registry.size()).toBe(1)

    created[0].emitChange('note.md')
    vi.advanceTimersByTime(200)

    expect(owner.sent).toHaveLength(0)
    expect(registry.size()).toBe(0)
    expect(registry.stop(id)).toBe(false)
  })
})

describe('createPreviewWatchRegistry — file watches', () => {
  it('debounces a burst of changes into one delivery with id, path and url', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    const { id } = registry.watch('/tmp/preview/note.md', owner)

    created[0].emitChange('note.md')
    created[0].emitChange('note.md')
    expect(owner.sent).toHaveLength(0)

    vi.advanceTimersByTime(200)
    expect(owner.sent).toEqual([
      {
        channel: PREVIEW_FILE_CHANGED_CHANNEL,
        payload: { id, path: '/tmp/preview/note.md', url: fileUrl('/tmp/preview/note.md') }
      }
    ])
  })

  it('rename events (atomic save-by-rename) trigger a reload', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    registry.watch('/tmp/preview/note.md', owner)
    created[0].emitRename('note.md')

    vi.advanceTimersByTime(200)
    expect(owner.sent).toHaveLength(1)
  })

  it('changes to sibling files in the same directory are ignored', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    registry.watch('/tmp/preview/note.md', owner)
    created[0].emitChange('other.md')
    created[0].emitChange('note.md.tmp')

    vi.advanceTimersByTime(1000)
    expect(owner.sent).toHaveLength(0)
  })

  it('null filename (documented fs.watch behavior) is treated as a match — a reload beats a missed save', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    registry.watch('/tmp/preview/note.md', owner)
    created[0].emitChange(null)

    vi.advanceTimersByTime(200)
    expect(owner.sent).toHaveLength(1)
  })

  it('a deleted target file does not send', () => {
    const { created, registry } = makeRegistry({ fileExists: () => false })
    const owner = fakeOwner()

    registry.watch('/tmp/preview/note.md', owner)
    created[0].emitChange('note.md')

    vi.advanceTimersByTime(1000)
    expect(owner.sent).toHaveLength(0)
  })

  it('a second change resets the debounce window', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    registry.watch('/tmp/preview/note.md', owner)

    created[0].emitChange('note.md')
    vi.advanceTimersByTime(100)
    created[0].emitChange('note.md')
    vi.advanceTimersByTime(100)
    expect(owner.sent).toHaveLength(0)

    vi.advanceTimersByTime(100)
    expect(owner.sent).toHaveLength(1)
  })

  it('stop closes the watcher, suppresses a pending debounce, and reports unknown ids as false', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    const { id } = registry.watch('/tmp/preview/note.md', owner)

    created[0].emitChange('note.md')
    expect(registry.stop(id)).toBe(true)
    expect(created[0].closed).toBe(true)

    vi.advanceTimersByTime(1000)
    expect(owner.sent).toHaveLength(0)

    expect(registry.stop(id)).toBe(false)
    expect(registry.stop('never-existed')).toBe(false)
  })

  it('closeAll tears down every registered watch', () => {
    const { created, registry } = makeRegistry()

    registry.watch('/tmp/preview/a.md', fakeOwner())
    registry.watch('/tmp/preview/b.md', fakeOwner())
    expect(registry.size()).toBe(2)

    registry.closeAll()

    expect(registry.size()).toBe(0)
    expect(created[0].closed).toBe(true)
    expect(created[1].closed).toBe(true)
  })
})

describe('createPreviewWatchRegistry — directory watches', () => {
  it('debounces directory churn and sends the directory path to its owner', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    const { id } = registry.watchDirectory('/tmp/plugins', owner)

    created[0].emitChange('new-plugin')
    created[0].emitChange('another-plugin')

    vi.advanceTimersByTime(200)
    expect(owner.sent).toEqual([
      {
        channel: PREVIEW_FILE_CHANGED_CHANNEL,
        payload: { id, path: '/tmp/plugins', url: fileUrl('/tmp/plugins') }
      }
    ])
  })

  it('stop suppresses a pending directory debounce', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    const { id } = registry.watchDirectory('/tmp/plugins', owner)
    created[0].emitChange('new-plugin')

    expect(registry.stop(id)).toBe(true)
    expect(created[0].closed).toBe(true)

    vi.advanceTimersByTime(1000)
    expect(owner.sent).toHaveLength(0)
  })

  it('closeAll reaps directory watches alongside file watches', () => {
    const { created, registry } = makeRegistry()

    registry.watch('/tmp/preview/note.md', fakeOwner())
    registry.watchDirectory('/tmp/plugins', fakeOwner())
    expect(registry.size()).toBe(2)

    registry.closeAll()

    expect(registry.size()).toBe(0)
    expect(created[0].closed).toBe(true)
    expect(created[1].closed).toBe(true)
  })

  it('a destroyed owner takes its directory watch down too', () => {
    const { created, registry } = makeRegistry()
    const owner = fakeOwner()

    registry.watchDirectory('/tmp/plugins', owner)
    owner.destroyed = true

    created[0].emitChange('new-plugin')
    vi.advanceTimersByTime(200)

    expect(owner.sent).toHaveLength(0)
    expect(registry.size()).toBe(0)
  })
})
