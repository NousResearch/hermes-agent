/**
 * Shared directory watches for preview file/dir notifications.
 *
 * Every preview file watch and plugin-dir watch used to create its OWN
 * fs.watch on the containing directory and then discard every sibling event
 * after wakeup: N previews in one dir meant N handles plus N wakeups per
 * write. This registry keeps ONE guarded fs.watch per directory and fans
 * events out to every subscriber; each subscriber keeps its own
 * filter/debounce/emit, so sharing changes no notification semantics.
 *
 * Storm protection stays per-directory (that is where the storm happens):
 * when the breaker trips to poll mode, each subscriber's snapshot is
 * compared individually, so a change to one file notifies only its own
 * watchers instead of every preview in the directory.
 */

import fs from 'node:fs'

import { guardedWatch, type GuardedWatch, type WatchListener } from './watch-storm-breaker'

export interface SharedDirSubscriber {
  /** Per-subscriber event filter (e.g. basename match + debounce). */
  onEvent: WatchListener
  /** Cheap fingerprint of THIS subscriber's target (file stat, readdir). */
  snapshot: () => string
  /** Called in poll mode when this subscriber's snapshot changed. */
  onPollChange: () => void
}

interface SubscriberState extends SharedDirSubscriber {
  last: string | null
}

interface SharedDirWatchDeps {
  platform?: string
  watch?: (watchDir: string, listener: WatchListener) => { close(): void }
  onTrip?: (watchDir: string, info: { events: number; windowMs: number }) => void
  /** Storm-breaker tuning (tests trip poll mode deterministically). */
  maxEvents?: number
  windowMs?: number
  pollMs?: number
  setInterval?: (fn: () => void, ms: number) => unknown
  clearInterval?: (handle: unknown) => void
}

function safeSnapshot(snapshot: () => string): string | null {
  try {
    return snapshot()
  } catch {
    return null
  }
}

export function createSharedDirWatchRegistry(deps: SharedDirWatchDeps = {}) {
  const platform = deps.platform ?? process.platform
  const makeWatch = deps.watch ?? ((watchDir: string, listener: WatchListener) => fs.watch(watchDir, listener))
  const watches = new Map<string, { subscribers: Set<SubscriberState>; watcher: GuardedWatch }>()

  /**
   * Subscribe to raw events on a directory. Returns a release function;
   * the underlying handle closes when the last subscriber leaves.
   */
  function acquire(watchDir: string, subscriber: SharedDirSubscriber): () => void {
    let shared = watches.get(watchDir)

    if (!shared) {
      const subscribers = new Set<SubscriberState>()

      const watcher = guardedWatch({
        platform,
        watch: listener => makeWatch(watchDir, listener),
        maxEvents: deps.maxEvents,
        windowMs: deps.windowMs,
        pollMs: deps.pollMs,
        setInterval: deps.setInterval,
        clearInterval: deps.clearInterval,
        onEvent: (eventType, filename) => {
          for (const sub of [...subscribers]) {
            sub.onEvent(eventType, filename)
          }
        },
        // Combined fingerprint. Deliberately SIDE-EFFECT FREE: the poll
        // loop compares this string, and per-subscriber diffing happens in
        // onPollChange below — writing `last` here would make every later
        // per-subscriber compare a no-op.
        snapshot: () => JSON.stringify([...subscribers].map(sub => safeSnapshot(sub.snapshot))),
        onPollChange: () => {
          for (const sub of [...subscribers]) {
            const next = safeSnapshot(sub.snapshot)

            if (next !== sub.last) {
              sub.last = next
              sub.onPollChange()
            }
          }
        },
        onTrip: info =>
          (deps.onTrip ?? ((dir, trip) =>
            console.warn(
              `[hermes] fs.watch storm on ${dir} (${trip.events} events within ${trip.windowMs}ms); polling instead`
            )))(watchDir, info)
      })

      shared = { subscribers, watcher }
      watches.set(watchDir, shared)
    }

    const state: SubscriberState = { ...subscriber, last: safeSnapshot(subscriber.snapshot) }
    shared.subscribers.add(state)

    let released = false

    return () => {
      if (released) {
        return
      }

      released = true
      const current = watches.get(watchDir)

      if (!current) {
        return
      }

      current.subscribers.delete(state)

      if (current.subscribers.size === 0) {
        current.watcher.close()
        watches.delete(watchDir)
      }
    }
  }

  return {
    acquire,
    /** Distinct watched directories (test seam). */
    dirCount: () => watches.size
  }
}
