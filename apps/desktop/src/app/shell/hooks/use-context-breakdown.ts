import { useEffect, useState } from 'react'

import type { ContextBreakdown } from '@/types/hermes'

interface ContextBreakdownOptions {
  busy: boolean
  enabled: boolean
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId: null | string
}

// Early in a session's life the gateway may not have the session registered yet
// (the RPC errors) or may have it registered without a hydrated agent (the RPC
// answers `context_max: 0`). Both are racing the gateway's own startup, not
// real answers — so retry a few times with a short gap before giving up.
// Without this the gauge stays blank until a toggle or a sent message changes
// an effect dependency and triggers the fetch again.
const RETRY_LIMIT = 4
const RETRY_DELAY_MS = 1_500

function isUsable(breakdown: ContextBreakdown | undefined) {
  // A zero ceiling means the backend has no hydration data yet, not an empty
  // context — a session always has at least the system prompt below it.
  return Boolean(breakdown && breakdown.context_max > 0)
}

function wait(ms: number) {
  return new Promise<void>(resolve => {
    setTimeout(resolve, ms)
  })
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
export function useContextBreakdown({ busy, enabled, requestGateway, sessionId }: ContextBreakdownOptions) {
  const [fetched, setFetched] = useState<{ breakdown: ContextBreakdown; sessionId: string } | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Mid-turn the transcript changes on every delta and the gateway already
    // streams measured usage, so an estimate would be both stale and wasteful.
    if (!enabled || !sessionId || busy) {
      return
    }

    let cancelled = false
    setLoading(true)

    // Attempt is per effect run, so a dependency change (session switch,
    // toggle, turn boundary) starts a fresh retry budget.
    const fetchOnce = async (attempt: number): Promise<void> => {
      try {
        const breakdown = await requestGateway<ContextBreakdown>('session.context_breakdown', {
          session_id: sessionId
        })

        if (!cancelled && isUsable(breakdown)) {
          setFetched({ breakdown, sessionId })

          return
        }
      } catch {
        // The gateway can answer "session not found" before registration —
        // transient, same as an empty success. Fall through to the retry.
      }

      // Exhausted the budget (or got through): the caller's finally clears
      // loading; retrying further would only spam the gateway.
      if (cancelled || attempt >= RETRY_LIMIT) {
        return
      }

      await wait(RETRY_DELAY_MS)

      if (cancelled) {
        return
      }

      await fetchOnce(attempt + 1)
    }

    void fetchOnce(0).finally(() => {
      if (!cancelled) {
        setLoading(false)
      }
    })

    return () => {
      cancelled = true
    }
  }, [busy, enabled, requestGateway, sessionId])

  return {
    breakdown: fetched && fetched.sessionId === sessionId ? fetched.breakdown : null,
    loading
  }
}
