import type { GatewayEvent } from '@hermes/shared'
import { useEffect } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { onGatewayEvent } from '@/contrib/events'
import { requestForOwnedSession } from '@/store/session-states'
import { $traceError, $traceLoading, $traceLog, appendTraceEvents, type RelayEvent } from '@/store/trace'

interface TraceEventsResult {
  events?: unknown[]
  recording?: boolean
  session_id: string
  truncated?: boolean
}

interface TraceEventPayload {
  event?: unknown
  trace_session_id?: string
}

const isRelayEvent = (value: unknown): value is RelayEvent =>
  !!value && typeof value === 'object' && typeof (value as RelayEvent).uuid === 'string'

/**
 * Load the Relay events Hermes recorded for `sessionId` (`trace.events`) and keep
 * the log growing from the live `trace.event` stream that same call subscribes to.
 * Live frames that race the fetch are held and merged once it lands.
 */
export function useSessionTrace(sessionId: null | string) {
  const { requestGateway } = useGatewayRequest()

  useEffect(() => {
    $traceLog.set(null)
    $traceError.set(null)

    if (!sessionId) {
      return
    }

    let cancelled = false
    let loaded = false
    const early: { event: RelayEvent; root: string }[] = []

    const off = onGatewayEvent('trace.event', (event: GatewayEvent) => {
      const payload = (event as { payload?: TraceEventPayload }).payload

      if (event.session_id !== sessionId || !isRelayEvent(payload?.event)) {
        return
      }

      const root = String(payload.trace_session_id ?? '')

      if (loaded) {
        appendTraceEvents(sessionId, root, [payload.event])
      } else {
        early.push({ event: payload.event, root })
      }
    })

    $traceLoading.set(true)

    void requestForOwnedSession<TraceEventsResult>(sessionId, requestGateway, 'trace.events', { session_id: sessionId })
      .then(result => {
        if (cancelled) {
          return
        }

        loaded = true
        $traceLog.set({
          sessionId,
          storedId: result.session_id,
          events: (result.events ?? []).filter(isRelayEvent),
          recording: result.recording !== false,
          truncated: Boolean(result.truncated)
        })

        for (const { event, root } of early) {
          appendTraceEvents(sessionId, root, [event])
        }
      })
      .catch(error => {
        if (!cancelled) {
          $traceError.set(error instanceof Error ? error.message : String(error))
        }
      })
      .finally(() => {
        if (!cancelled) {
          $traceLoading.set(false)
        }
      })

    return () => {
      cancelled = true
      off()
    }
  }, [requestGateway, sessionId])
}
