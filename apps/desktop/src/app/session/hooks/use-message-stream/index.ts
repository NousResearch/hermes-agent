import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { ScopedServerRequest } from '@/store/gateway'
import { broadcastSessionsChanged } from '@/store/session-sync'

import type { ClientSessionState } from '../../../types'

import { useCompleteTurn } from './complete-turn'
import { useDeltaFlush } from './delta-flush'
import { useGatewayEventHandler } from './gateway-event'
import { handleServerRequest as dispatchServerRequest } from './gateway-event/server-requests'
import { type UpdateSessionState, useMutateStream } from './mutate-stream'
import { useToolUpsert } from './tool-upsert'

interface MessageStreamOptions {
  activeGatewayProfile?: string
  activeSessionIdRef: MutableRefObject<string | null>
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>
  queryClient: QueryClient
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  updateSessionState: UpdateSessionState
}

export function useMessageStream({
  activeGatewayProfile = 'default',
  activeSessionIdRef,
  hydrateFromStoredSession,
  queryClient,
  refreshHermesConfig,
  refreshSessions,
  sessionStateByRuntimeIdRef,
  updateSessionState
}: MessageStreamOptions) {
  const sessionInterrupted = useCallback(
    (sessionId: string) => sessionStateByRuntimeIdRef.current.get(sessionId)?.interrupted ?? false,
    [sessionStateByRuntimeIdRef]
  )

  const mutateStream = useMutateStream(updateSessionState)

  // Turn-complete triggers a full sidebar refresh (recents + cron + messaging
  // REST fan-out, each scanning profile state.dbs server-side) plus a
  // cross-window broadcast that makes every other window do the same. Parallel
  // tiles / multi-window finishing near-simultaneously used to multiply that.
  // Coalesce completions into one trailing refresh per burst — a ~300ms title
  // lag is invisible; the redundant aggregator scans are not.
  const sessionsRefreshTimerRef = useRef<null | number>(null)

  const scheduleSessionsRefresh = useCallback(() => {
    if (sessionsRefreshTimerRef.current !== null) {
      return
    }

    const run = () => {
      sessionsRefreshTimerRef.current = null
      void refreshSessions().catch(() => undefined)
      // Sync freshly-titled rows to other windows (e.g. main, when the turn
      // ran in the pop-out).
      broadcastSessionsChanged()
    }

    if (typeof window === 'undefined') {
      run()

      return
    }

    sessionsRefreshTimerRef.current = window.setTimeout(run, 300)
  }, [refreshSessions])

  useEffect(
    () => () => {
      if (sessionsRefreshTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(sessionsRefreshTimerRef.current)
        sessionsRefreshTimerRef.current = null
      }
    },
    []
  )

  const nativeSubagentSessionsRef = useRef<Set<string>>(new Set())

  // Turns that auto-compacted: skip post-turn hydrate so live scrollback survives.
  const compactedTurnRef = useRef<Set<string>>(new Set())

  // Last session we applied a session.info cwd for — lets us tell an agent
  // relocating the SAME session (follow it) from a session switch (don't yank).
  const lastCwdInfoSessionRef = useRef<null | string>(null)

  const { flushQueuedDeltas, dropQueuedDeltas, appendAssistantDelta, appendReasoningDelta } =
    useDeltaFlush(mutateStream)

  const upsertToolCall = useToolUpsert(mutateStream, flushQueuedDeltas, sessionInterrupted, nativeSubagentSessionsRef)

  const { finalizeInterimAssistantMessage, completeAssistantMessage, failAssistantMessage } = useCompleteTurn(
    updateSessionState,
    hydrateFromStoredSession,
    scheduleSessionsRefresh,
    compactedTurnRef
  )

  const handleGatewayEvent = useGatewayEventHandler({
    activeGatewayProfile,
    appendAssistantDelta,
    appendReasoningDelta,
    activeSessionIdRef,
    compactedTurnRef,
    lastCwdInfoSessionRef,
    nativeSubagentSessionsRef,
    completeAssistantMessage,
    failAssistantMessage,
    flushQueuedDeltas,
    dropQueuedDeltas,
    finalizeInterimAssistantMessage,
    hydrateFromStoredSession,
    queryClient,
    refreshHermesConfig,
    scheduleSessionsRefresh,
    sessionInterrupted,
    sessionStateByRuntimeIdRef,
    updateSessionState,
    upsertToolCall
  })

  // Server→client requests (clarify, approval, sudo, …) from every socket the
  // registry owns. The request answers itself over the socket it arrived on,
  // so no owner routing is involved here — only which card to show.
  const handleServerRequest = useCallback(
    (request: ScopedServerRequest): boolean =>
      dispatchServerRequest(
        request,
        { activeSessionIdRef, sessionInterrupted, updateSessionState, upsertToolCall },
        activeSessionIdRef.current
      ),
    [activeSessionIdRef, sessionInterrupted, updateSessionState, upsertToolCall]
  )

  return {
    appendAssistantDelta,
    appendReasoningDelta,
    completeAssistantMessage,
    handleGatewayEvent,
    handleServerRequest,
    finalizeInterimAssistantMessage,
    upsertToolCall
  }
}
