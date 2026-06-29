import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { chatMessageText } from '@/lib/chat-messages'
import { $liveTurnBySession } from '@/store/live-turn'
import { $activeSessionId, $messages } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'
import {
  $hoveredSpanId,
  $selectedSpanId,
  $trace,
  $traceError,
  $traceLoading,
  $traceSelection,
  $traceTurns,
  rebaseTrace,
  type TraceDoc
} from '@/store/trace'

import { buildLiveTrace } from '../build-live-trace'

import { useSessionTrace, useTraceFetcher, useTraceTurns } from './use-session-trace'

export interface TraceView {
  activeIndex: null | number
  error: null | string
  liveIndex: null | number
  loading: boolean
  selectTurn: (index: number) => void
  selection: ReturnType<typeof $traceSelection.get>
  sessionId: null | string
  trace: null | TraceDoc
}

/**
 * The agents overlay's view-model: resolves the one trace to render from three
 * sources — the live event stitch while a followed turn is in flight, the
 * server-exact DB trace it folds into once settled, and any pinned/historical
 * turn or whole-session view. Keeps the route root pure layout.
 */
export function useTraceView(): TraceView {
  const sessionId = useStore($activeSessionId)
  const turns = useStore($traceTurns)
  const dbTrace = useStore($trace)
  const loading = useStore($traceLoading)
  const error = useStore($traceError)
  const subsBySession = useStore($subagentsBySession)
  const liveTurnBySession = useStore($liveTurnBySession)
  const selection = useStore($traceSelection)

  const [nowMs, setNowMs] = useState(() => Date.now())
  // Frozen last live render — kept after a turn settles so the time mapping (and
  // thus the view) can't shift. Cleared on session switch.
  const liveTraceRef = useRef<null | TraceDoc>(null)
  const finalizedRef = useRef(false)
  // The just-finished followed turn re-fetched from the DB and rebased onto the
  // live start: settled exactness folded into the live view (the "B" in A→B).
  const [foldedTrace, setFoldedTrace] = useState<null | TraceDoc>(null)
  const sessionRef = useRef(sessionId)

  if (sessionRef.current !== sessionId) {
    sessionRef.current = sessionId
    liveTraceRef.current = null
    finalizedRef.current = false
  }

  const liveSubs = useMemo(() => (sessionId ? (subsBySession[sessionId] ?? []) : []), [sessionId, subsBySession])
  const liveTurn = sessionId ? liveTurnBySession[sessionId] : undefined
  // Live = turn busy OR subagents in flight (robust to opening mid-turn).
  const isLive = !!liveTurn?.busy || liveSubs.some(s => s.status === 'running' || s.status === 'queued')
  const hasLiveData = isLive || (liveTurn?.tools.length ?? 0) > 0 || liveSubs.length > 0

  const following = selection === 'latest'
  const latestIndex = turns.length - 1
  // Latch onto the live stitch while following: once a turn has produced live
  // data we keep showing it (frozen, then folded with DB exactness after it
  // settles) and NEVER let the declarative DB fetch swap the store under us.
  // That swap + the turn-list reload races were the churn ("before subagents →
  // all → end"). DB-by-store is only for an explicitly pinned turn/all, or the
  // latest turn on a fresh idle open (no live data this session).
  const showLive = following && (hasLiveData || liveTraceRef.current !== null)

  const activeIndex =
    selection === 'all' ? null : selection === 'latest' ? (latestIndex >= 0 ? latestIndex : null) : selection

  const liveIndex = isLive && latestIndex >= 0 ? latestIndex : null

  const { reloadTurns } = useTraceTurns(sessionId)
  const { fetchTurn } = useTraceFetcher()

  // DB fetch target: a pinned turn number, else the latest settled turn. Skipped
  // entirely (undefined) while we render the live stitch or the whole session.
  const dbTurnArg =
    showLive || selection === 'all'
      ? undefined
      : typeof selection === 'number'
        ? selection
        : latestIndex >= 0
          ? latestIndex
          : undefined

  useSessionTrace(sessionId, dbTurnArg)

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
    setFoldedTrace(null)
  }, [sessionId])

  // A new live turn (or leaving 'latest') invalidates a folded snapshot.
  useEffect(() => {
    if (isLive || selection !== 'latest') {
      setFoldedTrace(null)
    }
  }, [isLive, selection])

  // While the turn streams, tick so running bars grow toward "now".
  useEffect(() => {
    if (!isLive) {
      return
    }

    const id = window.setInterval(() => setNowMs(Date.now()), 400)

    return () => window.clearInterval(id)
  }, [isLive])

  // Refresh the turn list on live edges so the nav reflects the in-flight /
  // finished turn. Does NOT touch the displayed (live) trace.
  const prevLive = useRef(isLive)
  useEffect(() => {
    if (isLive !== prevLive.current) {
      void reloadTurns()
    }

    prevLive.current = isLive
  }, [isLive, reloadTurns])

  // Fold: once a followed turn settles, pull its exact DB trace and rebase it
  // onto the frozen live start, then swap. rebaseTrace keeps it on the same
  // on-screen window, and the waterfall preserves the view across the tmap
  // change, so the swap is seamless — approximate live bars become server-exact
  // in place, with no reframe.
  useEffect(() => {
    if (!following || isLive || !sessionId || !liveTraceRef.current) {
      return
    }

    const liveStart = liveTraceRef.current.start
    let cancelled = false

    void (async () => {
      await reloadTurns()
      const idx = $traceTurns.get().length - 1

      if (idx < 0) {
        return
      }

      const db = await fetchTurn(sessionId, idx)

      if (!cancelled && db && db.spans.length > 0) {
        setFoldedTrace(rebaseTrace(db, liveStart))
      }
    })()

    return () => {
      cancelled = true
    }
  }, [following, isLive, sessionId, reloadTurns, fetchTurn])

  const trace = useMemo<null | TraceDoc>(() => {
    if (!showLive) {
      return dbTrace
    }

    const lastUser = $messages.get().findLast(m => m.role === 'user' && !m.hidden)
    const rootLabel = lastUser ? chatMessageText(lastUser).trim().slice(0, 80) : undefined

    if (isLive) {
      finalizedRef.current = false
      liveTraceRef.current = buildLiveTrace(liveTurn, liveSubs, nowMs, rootLabel, true)

      return liveTraceRef.current
    }

    // Settled & folded: server-exact spans rebased onto the live start.
    if (foldedTrace) {
      return foldedTrace
    }

    // Settled, fold not in yet: build the finalized snapshot once (running → ok,
    // no pulse) and freeze it as the bridge until the DB fold lands.
    if (!finalizedRef.current) {
      liveTraceRef.current = buildLiveTrace(liveTurn, liveSubs, nowMs, rootLabel, false)
      finalizedRef.current = true
    }

    return liveTraceRef.current
  }, [showLive, isLive, dbTrace, liveTurn, liveSubs, nowMs, foldedTrace])

  const selectTurn = (index: number) => $traceSelection.set(index === latestIndex ? 'latest' : index)

  return { activeIndex, error, liveIndex, loading, selectTurn, selection, sessionId, trace }
}
