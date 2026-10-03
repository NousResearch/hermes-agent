import { useEffect, useState } from 'react'

/** The one cadence every idle countdown/count-up label shares. */
export const IDLE_TICK_MS = 1000

type TickListener = () => void

/** Subscribers to the single shared interval — N clocks, one timer. */
const listeners = new Set<TickListener>()
let intervalId: number | null = null

function isViewed(): boolean {
  return typeof document !== 'undefined' && document.visibilityState === 'visible' && document.hasFocus()
}

function notify(): void {
  for (const listener of [...listeners]) {
    listener()
  }
}

function sync(): void {
  if (listeners.size === 0) {
    return
  }

  if (!isViewed()) {
    // Parked: hidden or unfocused while always-mounted clocks stay subscribed.
    // Clearing the interval (instead of skipping inside it) means zero wakeups.
    if (intervalId !== null) {
      window.clearInterval(intervalId)
      intervalId = null
    }

    return
  }

  if (intervalId === null) {
    // Leading tick on (re)entry so countdowns catch up immediately; a tick
    // that changes nothing sets identical state and React bails out.
    notify()
    intervalId = window.setInterval(notify, IDLE_TICK_MS)
  }
}

function unsubscribe(listener: TickListener): void {
  listeners.delete(listener)

  if (listeners.size === 0) {
    if (intervalId !== null) {
      window.clearInterval(intervalId)
      intervalId = null
    }

    window.removeEventListener('focus', sync)
    window.removeEventListener('blur', sync)
    document.removeEventListener('visibilitychange', sync)
  }
}

function subscribe(listener: TickListener): () => void {
  const first = listeners.size === 0
  listeners.add(listener)

  if (first) {
    window.addEventListener('focus', sync)
    window.addEventListener('blur', sync)
    document.addEventListener('visibilitychange', sync)
  }

  sync()

  return () => {
    unsubscribe(listener)
  }
}

/** Shared visibility+focus-gated 1s `nowMs` for idle countdown/count-up labels.
 *
 * #122413: every per-second clock ran its own always-on `setInterval`, so a
 * visible-but-untouched window re-rendered (and re-resolved date formatters +
 * fonts) every second forever. Consumers share one interval that only runs
 * while the document is actually being viewed; while parked there are zero
 * wakeups, and a tick that changes nothing produces no re-render.
 */
export function useIdleClock(enabled = true): number {
  const [nowMs, setNowMs] = useState(() => Date.now())

  useEffect(() => {
    if (!enabled) {
      return
    }

    return subscribe(() => setNowMs(Date.now()))
  }, [enabled])

  return nowMs
}
