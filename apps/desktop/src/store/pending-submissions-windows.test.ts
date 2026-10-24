import { afterEach, expect, it } from 'vitest'

import { getQueuedPrompts } from './composer-queue'
import { reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'

// Chromium gives each renderer process its own cached copy of the origin's localStorage and
// propagates writes asynchronously, last writer wins per key. Two windows: each writes against
// its own (stale) view, then the copies converge.
function storageView(shared: Map<string, string>) {
  const local = new Map(shared)
  const written = new Map<string, null | string>()

  const storage = {
    get length() { return local.size },
    key: (index: number) => [...local.keys()][index] ?? null,
    getItem: (key: string) => local.get(key) ?? null,
    setItem: (key: string, value: string) => { local.set(key, value); written.set(key, value) },
    removeItem: (key: string) => { local.delete(key); written.set(key, null) },
    clear: () => undefined
  }

  const flush = () => { for (const [key, value] of written) { if (value === null) { shared.delete(key) } else { shared.set(key, value) } } }

  return { storage, flush }
}

const real = window.localStorage

afterEach(() => { Object.defineProperty(window, 'localStorage', { configurable: true, value: real }); real.clear() })

it("one window's pending-submission write never drops another window's concurrent entry", () => {
  const shared = new Map<string, string>()
  const a = storageView(shared)
  const b = storageView(shared)

  Object.defineProperty(window, 'localStorage', { configurable: true, value: a.storage })
  trackPendingSubmission('chat', { id: 'from-a', text: 'A', displayText: '/a' })
  Object.defineProperty(window, 'localStorage', { configurable: true, value: b.storage })
  trackPendingSubmission('chat', { id: 'from-b', text: 'B', displayText: '/b' })
  a.flush()
  b.flush()

  const converged = storageView(shared)
  Object.defineProperty(window, 'localStorage', { configurable: true, value: converged.storage })
  // The server queue names both admissions; each keeps the display text its own window tracked.
  reconcilePendingSubmissions('chat', [
    { admission_id: 'from-a', status: 'queued', user: 'A' },
    { admission_id: 'from-b', status: 'queued', user: 'B' }
  ])
  expect(getQueuedPrompts('chat').map(entry => [entry.id, entry.displayText])).toEqual([['from-a', '/a'], ['from-b', '/b']])
})
