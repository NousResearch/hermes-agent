import { useCallback, useSyncExternalStore } from 'react'

/**
 * Shared UI clock: the current epoch ms, re-rendering every `intervalMs`
 * while `enabled` — with ONE interval per cadence no matter how many
 * components subscribe.
 *
 * Replaces per-component `setInterval` clocks (kanban card ticking, per-row
 * elapsed timers): 50 cards each owning a 5s timer means 50 timers plus 50
 * wakeups; here 50 subscribers share one. Viewed-gating matches
 * `use-viewed-interval` (no ticks while unfocused/hidden, leading tick on
 * return so the UI catches up immediately).
 */

interface ClockEntry {
  intervalId: number | null
  listeners: Set<() => void>
  value: number
}

const clocks = new Map<number, ClockEntry>()

function entryFor(intervalMs: number): ClockEntry {
  let entry = clocks.get(intervalMs)

  if (!entry) {
    entry = { intervalId: null, listeners: new Set(), value: 0 }
    clocks.set(intervalMs, entry)
  }

  return entry
}

function stopEntry(entry: ClockEntry): void {
  if (entry.intervalId !== null) {
    window.clearInterval(entry.intervalId)
    entry.intervalId = null
  }
}

function resyncEntry(entry: ClockEntry, intervalMs: number): void {
  const viewed = document.visibilityState === 'visible' && document.hasFocus()

  if (entry.listeners.size === 0 || !viewed) {
    stopEntry(entry)

    return
  }

  if (entry.intervalId === null) {
    entry.value = Date.now()
    entry.intervalId = window.setInterval(() => {
      entry.value = Date.now()

      for (const notify of [...entry.listeners]) {
        notify()
      }
    }, intervalMs)

    // Leading tick: subscribers render with the fresh value immediately
    // instead of waiting out the first interval.
    for (const notify of [...entry.listeners]) {
      notify()
    }
  }
}

function resyncAllEntries(): void {
  for (const [intervalMs, entry] of clocks) {
    resyncEntry(entry, intervalMs)
  }
}

let globalSyncInstalled = false

function ensureGlobalSync(): void {
  if (globalSyncInstalled || typeof window === 'undefined') {
    return
  }

  globalSyncInstalled = true
  window.addEventListener('focus', resyncAllEntries)
  window.addEventListener('blur', resyncAllEntries)
  document.addEventListener('visibilitychange', resyncAllEntries)
}

export function useNow(intervalMs: number, enabled = true): number {
  ensureGlobalSync()

  const subscribe = useCallback(
    (notify: () => void) => {
      if (!enabled) {
        return () => {}
      }

      const entry = entryFor(intervalMs)
      entry.listeners.add(notify)
      resyncEntry(entry, intervalMs)

      return () => {
        entry.listeners.delete(notify)
        resyncEntry(entry, intervalMs)
      }
    },
    [enabled, intervalMs]
  )

  const getSnapshot = useCallback(() => {
    if (!enabled) {
      return Date.now()
    }

    const entry = clocks.get(intervalMs)

    return entry && entry.value ? entry.value : Date.now()
  }, [enabled, intervalMs])

  return useSyncExternalStore(subscribe, getSnapshot)
}

/** Test seam: drop all shared clocks (intervals are cleared via unsubscribe). */
export function __resetNowClocksForTests(): void {
  for (const entry of clocks.values()) {
    stopEntry(entry)
    entry.listeners.clear()
    entry.value = 0
  }

  clocks.clear()
}
