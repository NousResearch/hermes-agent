import { useCallback, useEffect } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import {
  $trace,
  $traceError,
  $traceLoading,
  $traceTurns,
  setTrace,
  toTraceDoc,
  toTurnSummaries,
  type TraceDoc
} from '@/store/trace'

/**
 * Fetch the execution trace for a session from the gateway (`trace.get`) and
 * publish it into the trace store. Re-fetches when the session or turn changes.
 */
export function useSessionTrace(sessionId: null | string, turn?: number) {
  const { requestGateway } = useGatewayRequest()

  const load = useCallback(async () => {
    if (!sessionId) {
      setTrace(null)

      return
    }

    $traceLoading.set(true)
    $traceError.set(null)

    try {
      const params: Record<string, unknown> = { session_id: sessionId }

      if (typeof turn === 'number') {
        params.turn = turn
      }

      const wire = await requestGateway<Record<string, unknown>>('trace.get', params)
      setTrace(toTraceDoc(wire))
    } catch (error) {
      $traceError.set(error instanceof Error ? error.message : String(error))
      $trace.set(null)
    } finally {
      $traceLoading.set(false)
    }
  }, [requestGateway, sessionId, turn])

  useEffect(() => {
    void load()
  }, [load])
}

/**
 * Fetch the per-turn summaries for a session (`trace.turns`) so the overlay can
 * offer a turn strip. Publishes into the trace store.
 */
export function useTraceTurns(sessionId: null | string) {
  const { requestGateway } = useGatewayRequest()

  const reloadTurns = useCallback(async () => {
    if (!sessionId) {
      $traceTurns.set([])

      return
    }

    try {
      const wire = await requestGateway<{ turns?: unknown[] }>('trace.turns', { session_id: sessionId })
      $traceTurns.set(toTurnSummaries(wire as { turns?: never[] }))
    } catch {
      // Keep the previous list on a transient failure — never blank it, or the
      // nav (and any latest-index math) flickers mid-turn.
    }
  }, [requestGateway, sessionId])

  useEffect(() => {
    void reloadTurns()
  }, [reloadTurns])

  return { reloadTurns }
}

/**
 * Imperative one-shot fetch of a single turn's trace, WITHOUT touching the
 * `$trace` store. The agents overlay uses this to grab a just-finished turn's
 * exact DB trace and fold it into the live view, instead of letting the
 * declarative `useSessionTrace` swap the store (which would race the live
 * stitch and churn the view).
 */
export function useTraceFetcher() {
  const { requestGateway } = useGatewayRequest()

  const fetchTurn = useCallback(
    async (sessionId: null | string, turn: number): Promise<null | TraceDoc> => {
      if (!sessionId) {
        return null
      }

      try {
        const wire = await requestGateway<Record<string, unknown>>('trace.get', { session_id: sessionId, turn })

        return toTraceDoc(wire)
      } catch {
        return null
      }
    },
    [requestGateway]
  )

  return { fetchTurn }
}
