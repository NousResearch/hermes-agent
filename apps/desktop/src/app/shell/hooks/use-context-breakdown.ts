import { useEffect, useRef, useState } from 'react'

import type { ContextBreakdown } from '@/types/hermes'
import {
  type ContextUsageScope,
  type ContextUsageVersion,
  loadContextUsageSnapshot,
  saveContextUsageSnapshot
} from '@/store/context-usage-cache'

interface ContextBreakdownOptions {
  busy: boolean
  enabled: boolean
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId: null | string
  /** Extra refetch signal — e.g. the model id, so a switch re-resolves the
   *  window without waiting for the next turn. */
  refreshKey?: string
  /** Durable last-known read for this session (stored id + owning scope +
   *  row version). Paints while no live breakdown exists — live data always
   *  replaces it. */
  persist?: {
    scope?: ContextUsageScope
    storedSessionId: string
    version: ContextUsageVersion | null
  } | null
}

/** A brand-new session's breakdown is only the system-prompt + tools + rules
 *  baseline the backend estimates before anything is sent. Read verbatim that
 *  paints a fresh chat as "1% used" — nothing occupies the window until the
 *  first turn. Zero the occupancy (keep the window) so an empty conversation
 *  reads 0%, the same as the no-runtime draft path. */
function isEmptyConversation(breakdown: ContextBreakdown): boolean {
  return (
    breakdown.context_source === 'local_estimate' &&
    !breakdown.categories.some(category => category.id === 'conversation' && category.tokens > 0)
  )
}

/** A payload that carries no data at all: no window, no categories, no usage.
 *  The backend returns exactly this shape when the session has no live agent
 *  yet (a restored record pre-build, or a reaped runtime) — it must not paint
 *  as "0%" over a restored read, nor as a definitive empty. */
function isNoData(breakdown: ContextBreakdown): boolean {
  return (breakdown.context_max ?? 0) <= 0 && (breakdown.context_used ?? 0) <= 0 && breakdown.categories.length === 0
}

/** The focused session's context breakdown, fetched as soon as the statusbar
 *  gauge is on screen rather than when its popover opens.
 *
 *  The backend only reports measured context occupancy (`last_prompt_tokens`)
 *  once a turn has run in THIS process, so a resumed session reports none —
 *  which is why turning the gauge on used to do nothing at all until you sent
 *  a message. `session.context_breakdown` estimates the same figure from the
 *  live system prompt + tools + transcript, so it answers for a session that
 *  hasn't spoken yet. It is a read-only chars/4 pass: no provider call, no
 *  prompt-cache impact.
 *
 *  Refetches when the focused session changes and when a turn ends (the
 *  transcript just grew). Held keyed by the session it describes so switching
 *  sessions drops the previous numbers instead of painting them under the new
 *  session's name. */
export function useContextBreakdown({
  busy,
  enabled,
  persist,
  refreshKey,
  requestGateway,
  sessionId
}: ContextBreakdownOptions) {
  const [fetched, setFetched] = useState<{ breakdown: ContextBreakdown; sessionId: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const persistRef = useRef(persist)
  persistRef.current = persist

  useEffect(() => {
    // Mid-turn the transcript changes on every delta and the gateway already
    // streams measured usage, so an estimate would be both stale and wasteful.
    if (!enabled || !sessionId || busy) {
      return
    }

    let cancelled = false
    setLoading(true)

    void requestGateway<ContextBreakdown>('session.context_breakdown', { session_id: sessionId })
      .then(breakdown => {
        if (!cancelled && breakdown) {
          setFetched({ breakdown, sessionId })

          // Bank the read so a later restart can paint it while the live
          // agent rebinds. Empties carry nothing worth restoring.
          const saved = persistRef.current

          if (saved && !isNoData(breakdown) && !isEmptyConversation(breakdown)) {
            saveContextUsageSnapshot(
              saved.storedSessionId,
              {
                context_max: breakdown.context_max,
                context_percent: breakdown.context_percent,
                context_used: breakdown.context_used,
                model: breakdown.model
              },
              saved.scope,
              saved.version
            )
          }
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
    // refreshKey intentionally re-runs the fetch (model switches re-resolve).
  }, [busy, enabled, refreshKey, requestGateway, sessionId])

  return {
    breakdown: resolveBreakdown(fetched, sessionId, persist),
    loading
  }
}

function resolveBreakdown(
  fetched: { breakdown: ContextBreakdown; sessionId: string } | null,
  sessionId: null | string,
  persist?: ContextBreakdownOptions['persist']
) {
  const breakdown = fetched && fetched.sessionId === sessionId ? fetched.breakdown : null

  if (breakdown && !isNoData(breakdown)) {
    if (!isEmptyConversation(breakdown)) {
      return breakdown
    }

    return { ...breakdown, categories: [], context_percent: 0, context_used: 0, estimated_total: 0 }
  }

  // No live data yet (fetch pending), or a no-data payload (no agent bound):
  // fall back to the durable last-known read instead of painting zero.
  if (!persist?.storedSessionId) {
    return null
  }

  const restored = loadContextUsageSnapshot(persist.storedSessionId, persist.scope, persist.version)

  if (!restored) {
    return null
  }

  return {
    categories: [],
    context_estimated: true,
    context_max: restored.context_max,
    context_percent: restored.context_percent,
    context_source: 'restored',
    context_used: restored.context_used,
    estimated_total: restored.context_used,
    model: restored.model ?? ''
  } satisfies ContextBreakdown
}
