import { type MutableRefObject, useCallback, useEffect } from 'react'

import type { GatewayRequest } from '@/lib/gateway-rpc'
import { $currentCwd, setContextSuggestions } from '@/store/session'


interface ContextSuggestionsOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  currentCwd: string
  gatewayState: string | undefined
  requestGateway: GatewayRequest
}

export function useContextSuggestions({
  activeSessionId,
  activeSessionIdRef,
  currentCwd,
  gatewayState,
  requestGateway
}: ContextSuggestionsOptions) {
  const refresh = useCallback(async () => {
    if (!activeSessionId) {
      setContextSuggestions([])

      return
    }

    const sessionId = activeSessionId
    const cwd = currentCwd || ''

    // Race guard: only commit if the session+cwd we sent for still match
    // by the time the gateway responds.
    const stillCurrent = () => activeSessionIdRef.current === sessionId && $currentCwd.get() === cwd

    try {
      const result = await requestGateway('complete.path', {
        session_id: sessionId,
        word: '@file:',
        cwd: cwd || undefined
      })

      if (stillCurrent()) {
        setContextSuggestions((result.items || []).filter(i => i.text))
      }
    } catch {
      if (stillCurrent()) {
        setContextSuggestions([])
      }
    }
  }, [activeSessionId, activeSessionIdRef, currentCwd, requestGateway])

  useEffect(() => {
    if (gatewayState === 'open' && activeSessionId) {
      void refresh()
    }
  }, [activeSessionId, gatewayState, refresh])
}
