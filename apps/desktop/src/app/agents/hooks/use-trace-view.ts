import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { $activeSessionId } from '@/store/session'
import {
  $hoveredSpanId,
  $selectedSpanId,
  $traceError,
  $traceLoading,
  $traceLog,
  $traceSelection,
  type TraceDoc,
  type TraceTurnSummary
} from '@/store/trace'

import { foldRelayEvents, type TraceLabels, traceTurns, turnTrace } from '../fold-relay-events'

import { useSessionTrace } from './use-session-trace'

export interface TraceView {
  activeIndex: null | number
  error: null | string
  liveIndex: null | number
  loading: boolean
  recording: boolean
  selectTurn: (index: number) => void
  selection: ReturnType<typeof $traceSelection.get>
  sessionId: null | string
  trace: null | TraceDoc
  truncated: boolean
  turns: TraceTurnSummary[]
}

/**
 * The agents overlay's view-model: the active session's Relay event log folded
 * into one span tree, narrowed to the selected turn. Live and settled turns are the
 * same fold over the same log, so a turn finishing never swaps what is on screen.
 */
export function useTraceView(labels: TraceLabels): TraceView {
  const sessionId = useStore($activeSessionId)
  const log = useStore($traceLog)
  const loading = useStore($traceLoading)
  const error = useStore($traceError)
  const selection = useStore($traceSelection)
  const [nowSec, setNowSec] = useState(() => Date.now() / 1000)

  useSessionTrace(sessionId)

  const doc = useMemo(() => (log ? foldRelayEvents(log.events, labels, nowSec) : null), [labels, log, nowSec])
  const turns = useMemo(() => (doc ? traceTurns(doc) : []), [doc])
  const live = turns.some(turn => turn.running)
  const latestIndex = turns.length - 1

  // Drop the ephemeral hover/selection when the panel closes so a stale span
  // can't auto-zoom the next time it opens.
  useEffect(
    () => () => {
      $selectedSpanId.set(null)
      $hoveredSpanId.set(null)
    },
    []
  )

  // Follow-latest on session switch.
  useEffect(() => {
    $traceSelection.set('latest')
  }, [sessionId])

  // While a turn runs, tick so its open bars grow toward "now".
  useEffect(() => {
    if (!live) {
      return
    }

    const id = window.setInterval(() => setNowSec(Date.now() / 1000), 400)

    return () => window.clearInterval(id)
  }, [live])

  const activeIndex =
    selection === 'all' ? null : selection === 'latest' ? (latestIndex >= 0 ? latestIndex : null) : selection

  const activeTurnId = activeIndex === null ? undefined : turns[activeIndex]?.id
  const trace = useMemo(() => (doc && activeTurnId ? turnTrace(doc, activeTurnId) : doc), [activeTurnId, doc])
  const liveIndex = live ? turns.findLastIndex(turn => turn.running) : null
  const selectTurn = (index: number) => $traceSelection.set(index === latestIndex ? 'latest' : index)

  return {
    activeIndex,
    error,
    liveIndex,
    loading,
    recording: log?.recording !== false,
    selectTurn,
    selection,
    sessionId,
    trace,
    truncated: Boolean(log?.truncated),
    turns
  }
}
