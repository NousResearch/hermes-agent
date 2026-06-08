import { useEffect } from 'react'

import { setSessionPresence } from '@/store/session'
import type { SessionPresenceListResponse } from '@/types/hermes'

const REFRESH_MS = 5_000

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export function useSessionPresence(gatewayState: string | undefined, requestGateway: GatewayRequester) {
  useEffect(() => {
    if (gatewayState !== 'open') {
      setSessionPresence([])

      return
    }

    let cancelled = false

    const refresh = async () => {
      try {
        const result = await requestGateway<SessionPresenceListResponse>('session.presence_list')

        if (!cancelled) {
          setSessionPresence(Array.isArray(result.sessions) ? result.sessions : [])
        }
      } catch {
        if (!cancelled) {
          setSessionPresence([])
        }
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), REFRESH_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [gatewayState, requestGateway])
}
