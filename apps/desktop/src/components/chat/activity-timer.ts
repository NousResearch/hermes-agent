import { useEffect, useRef, useState } from 'react'

// Module-level registry so timers survive component unmount/remount (e.g.
// when a tool row scrolls out and back). Keyed by caller-supplied timerKey;
// anonymous timers (no key) start fresh each mount.
const startedAtByKey = new Map<string, number>()

function startedAt(key?: string): number {
  if (!key) {
    return Date.now()
  }

  const existing = startedAtByKey.get(key)

  if (existing !== undefined) {
    return existing
  }

  const now = Date.now()
  startedAtByKey.set(key, now)

  return now
}

export function formatElapsed(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`
  }

  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

export function useElapsedSeconds(active = true, timerKey?: string): number {
  const start = useRef(startedAt(timerKey))
  const lastKey = useRef(timerKey)
  const [elapsed, setElapsed] = useState(() => Math.max(0, Math.floor((Date.now() - start.current) / 1000)))

  if (lastKey.current !== timerKey) {
    start.current = startedAt(timerKey)
    lastKey.current = timerKey
  }

  useEffect(() => {
    if (!active) {
      return
    }

    if (timerKey) {
      start.current = startedAt(timerKey)
    }

    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - start.current) / 1000)))
    tick()
    const id = window.setInterval(tick, 1000)

    return () => window.clearInterval(id)
  }, [active, timerKey])

  return elapsed
}

export function __resetElapsedTimerRegistryForTests() {
  startedAtByKey.clear()
}
