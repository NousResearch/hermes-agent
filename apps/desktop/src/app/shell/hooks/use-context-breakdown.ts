import { useEffect, useState } from 'react'

import type { ContextBreakdown } from '@/types/hermes'

interface ContextBreakdownOptions {
  busy: boolean
  enabled: boolean
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId: null | string
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
 *  session's name.
 *
 *  The refetch after a turn end can land in windows where the answer is not
 *  trustworthy (#94001): right after a `session.reclaimed` resume the new
 *  agent may still be building (the backend answers a ZEROED breakdown,
 *  which would blank the meter), or the RPC can fail outright during the
 *  rebind. Neither may freeze the gauge on pre-compression numbers: a failed
 *  or agentless answer is retried on a bounded backoff, and once retries are
 *  exhausted the cached breakdown is EVICTED — the meter shows nothing rather
 *  than numbers known to be stale. Without eviction the stale value would
 *  survive until the session is switched, which is the manual recovery every
 *  report describes. */

/** A breakdown the backend computed from a live agent carries a real
 *  context_max (the compressor's context_length). Zero means the
 *  `agent is None` branch answered from empty metadata — not data. */
function isZeroedBreakdown(breakdown: ContextBreakdown): boolean {
  return !(breakdown.context_max > 0)
}

const RETRY_DELAYS_MS = [1_000, 4_000, 12_000]

export function useContextBreakdown({ busy, enabled, requestGateway, sessionId }: ContextBreakdownOptions) {
  const [fetched, setFetched] = useState<{ breakdown: ContextBreakdown; sessionId: string } | null>(null)
  const [loading, setLoading] = useState(false)
  // Bounded retry: `attempt` indexes RETRY_DELAYS_MS; advancing it re-runs the
  // fetch effect. `exhausted` marks the end state — cached numbers evicted,
  // backoff stopped, meter dark until the next legitimate trigger (session
  // change or turn end) starts a fresh ladder.
  const [attempt, setAttempt] = useState(0)
  const [exhausted, setExhausted] = useState(false)

  // A session switch must not inherit the previous session's retry state —
  // its first successful fetch is authoritative for it. (The fetch effect's
  // own cleanup cancels any pending backoff timer from the old session.)
  useEffect(() => {
    setAttempt(0)
    setExhausted(false)
  }, [sessionId])

  useEffect(() => {
    // Mid-turn the transcript changes on every delta and the gateway already
    // streams measured usage, so an estimate would be both stale and wasteful.
    if (!enabled || !sessionId || busy) {
      return
    }

    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    setLoading(true)

    const scheduleRetry = () => {
      if (attempt < RETRY_DELAYS_MS.length) {
        timer = setTimeout(() => setAttempt(a => a + 1), RETRY_DELAYS_MS[attempt])
      } else {
        // Ladder exhausted: evict whatever is cached. A dark meter is honest;
        // pre-compression numbers are a lie (#94001).
        setFetched(null)
        setExhausted(true)
        setLoading(false)
      }
    }

    void requestGateway<ContextBreakdown>('session.context_breakdown', { session_id: sessionId })
      .then(breakdown => {
        if (cancelled) {
          return
        }

        // Zeroed = the backend had no agent to measure against (post-reclaim
        // rebuild window). Don't cache it — it would blank the meter now and
        // stand in for real data later. Retry on the bounded ladder instead.
        if (!breakdown || isZeroedBreakdown(breakdown)) {
          scheduleRetry()

          return
        }

        setFetched({ breakdown, sessionId })
        setExhausted(false)
        setLoading(false)
      })
      .catch(() => {
        // The fetch itself failed (rebind race after session.reclaimed, etc.).
        // Same bounded ladder; on exhaustion, evict rather than freeze.
        if (cancelled) {
          return
        }

        scheduleRetry()
      })

    return () => {
      cancelled = true

      if (timer) {
        clearTimeout(timer)
      }
    }
    // `attempt` drives the bounded-retry re-runs; everything else is a trigger.
  }, [attempt, busy, enabled, requestGateway, sessionId])

  // While the retry ladder is still running, the last good breakdown stays on
  // screen (the meter ticking down beats flickering dark); once exhausted the
  // cache is gone, so this returns null and the meter goes dark honestly.
  return {
    breakdown: fetched && fetched.sessionId === sessionId ? fetched.breakdown : null,
    loading
  }
}