import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

import type { Translations } from '@/i18n'

interface SessionSliceStore<T> {
  get(): Record<string, T | undefined>
  listen(listener: () => void): () => void
}

/** Subscribe to one stable per-session store entry without reacting to other sessions' writes. */
export function useSessionValue<T>(store: SessionSliceStore<T>, sessionId: string | null): T | undefined {
  const subscribe = useCallback((onChange: () => void) => store.listen(onChange), [store])

  return useSyncExternalStore(subscribe, () => (sessionId ? store.get()[sessionId] : undefined))
}

/** Wall clock that re-renders once a second while ``active``, for the heartbeat and loop countdowns. */
export function useCountdownClock(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!active) {
      return
    }

    const tick = () => setNow(Date.now())
    const interval = window.setInterval(tick, 1_000)

    return () => window.clearInterval(interval)
  }, [active])

  return now
}

export interface ConfirmState {
  title: string
  description: string
  destructive?: boolean
  confirmLabel?: string
  onConfirm: () => Promise<void> | void
}

/** Formats a countdown precisely without exposing the user's local timezone. */
export function formatCountdown(targetTimestamp: number, now = Date.now()): string {
  const targetMs = targetTimestamp > 1e11 ? targetTimestamp : targetTimestamp * 1000
  const remainingSeconds = Math.max(0, Math.ceil((targetMs - now) / 1000))
  const days = Math.floor(remainingSeconds / 86_400)
  const hours = Math.floor((remainingSeconds % 86_400) / 3_600)
  const minutes = Math.floor((remainingSeconds % 3_600) / 60)
  const seconds = remainingSeconds % 60
  const padded = (value: number) => String(value).padStart(2, '0')
  const clock = `${padded(hours)}:${padded(minutes)}:${padded(seconds)}`

  return days > 0 ? `${days}d ${clock}` : clock
}

export function formatInterval(seconds: number, t: Translations): string {
  if (seconds % 3600 === 0) {
    return t.statusStack.control.loopEveryHours(seconds / 3600)
  }

  if (seconds % 60 === 0) {
    return t.statusStack.control.loopEveryMinutes(seconds / 60)
  }

  return t.statusStack.control.loopEverySeconds(seconds)
}

export function formatHeartbeatInterval(seconds: number, t: Translations): string {
  if (seconds % 3600 === 0) {
    return t.statusStack.control.heartbeatEveryHours(seconds / 3600)
  }

  if (seconds % 60 === 0) {
    return t.statusStack.control.heartbeatEveryMinutes(seconds / 60)
  }

  return t.statusStack.control.heartbeatEverySeconds(seconds)
}
