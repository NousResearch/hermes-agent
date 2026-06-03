import { useEffect, useRef } from 'react'

export const SESSION_LIST_AUTO_REFRESH_MS = 15_000

export interface RefreshSessionsOptions {
  showLoading?: boolean
}

interface UseSessionListAutoRefreshOptions {
  enabled: boolean
  intervalMs?: number
  refreshSessions: (options?: RefreshSessionsOptions) => Promise<void>
}

export function useSessionListAutoRefresh({
  enabled,
  intervalMs = SESSION_LIST_AUTO_REFRESH_MS,
  refreshSessions
}: UseSessionListAutoRefreshOptions) {
  const refreshSessionsRef = useRef(refreshSessions)

  useEffect(() => {
    refreshSessionsRef.current = refreshSessions
  }, [refreshSessions])

  useEffect(() => {
    if (!enabled || intervalMs <= 0) {
      return undefined
    }

    const timer = window.setInterval(() => {
      void refreshSessionsRef.current({ showLoading: false }).catch(() => undefined)
    }, intervalMs)

    return () => window.clearInterval(timer)
  }, [enabled, intervalMs])
}
