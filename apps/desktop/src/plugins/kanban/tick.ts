/**
 * Board tick shared by every kanban card.
 *
 * Each card's elapsed label used to own a 5s `setInterval`: 50 cards meant
 * 50 timers plus 50 wakeups every 5s, forever. Here all cards share one
 * interval; the last unmount stops it.
 *
 * Local to the kanban plugin on purpose: plugins import only
 * @hermes/plugin-sdk (and react), so the app-wide shared clock
 * (@/hooks/use-now) is out of reach here.
 */

import { useEffect, useState } from 'react'

const TICK_MS = 5_000

const listeners = new Set<() => void>()
let intervalId: number | null = null

function tick(): void {
  for (const notify of [...listeners]) {
    notify()
  }
}

function sync(): void {
  if (listeners.size === 0) {
    if (intervalId !== null) {
      window.clearInterval(intervalId)
      intervalId = null
    }

    return
  }

  intervalId ??= window.setInterval(tick, TICK_MS)
}

/** Current epoch ms, re-rendering on the shared 5s board tick. */
export function useBoardTick(): number {
  const [, force] = useState(0)

  useEffect(() => {
    const notify = () => force(n => n + 1)
    listeners.add(notify)
    sync()

    return () => {
      listeners.delete(notify)
      sync()
    }
  }, [])

  return Date.now()
}
