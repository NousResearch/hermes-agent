import { useCallback, useSyncExternalStore } from 'react'

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

export interface ConfirmState {
  title: string
  description: string
  destructive?: boolean
  confirmLabel?: string
  onConfirm: () => Promise<void> | void
}

export function formatApproximateTime(targetTimestamp: number): string {
  if (!targetTimestamp) {
    return ''
  }

  const targetMs = targetTimestamp > 1e11 ? targetTimestamp : targetTimestamp * 1000
  const diffMs = targetMs - Date.now()

  if (diffMs <= 0) {
    return '~0s'
  }

  const seconds = Math.round(diffMs / 1000)

  if (seconds < 60) {
    return `~${seconds}s`
  }

  const minutes = Math.round(seconds / 60)

  if (minutes < 60) {
    return `~${minutes}m`
  }

  const hours = Math.round(minutes / 60)

  if (hours < 24) {
    return `~${hours}h`
  }

  const days = Math.round(hours / 24)

  return `~${days}d`
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
