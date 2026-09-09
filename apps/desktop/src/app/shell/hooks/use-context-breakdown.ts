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
 *  Two events change the transcript WITHOUT a busy toggle, so the refetch
 *  must be driven explicitly:
 *
 *  - **Compression** — manual `/compress` runs the `session.compress` RPC
 *    outside any turn (busy never flips), and auto-compression can commit
 *    mid-turn. The post-compression transcript measures a fraction of the
 *    pre-compression size, so a served pre-compression breakdown is not
 *    merely stale, it is wrong by 5-10x (#94001).
 *  - **Reclaim** (`session.reclaimed`) — the runtime is reaped and the
 *    session re-resumes; the first refetch can race the agent rebuild.
 *
 *  `invalidateContextBreakdown` bumps a per-session generation that both
 *  call sites (slash.ts after a successful compress, lifecycle.ts on
 *  reclaim) fire, forcing an immediate refetch. An untrustworthy answer —
 *  an outright failure, or a ZEROED breakdown from the backend's
 *  agent-is-None branch (`context_max: 0`) — retries on a bounded backoff;
 *  once the ladder is exhausted the cached breakdown is EVICTED so the
 *  meter goes dark honestly rather than showing numbers known to be
 *  stale. */

/** A breakdown the backend computed from a live agent carries a real
 *  context_max (the compressor's context_length). Zero means the
 *  `agent is None` branch answered from empty metadata — not data. */
function isZeroedBreakdown(breakdown: ContextBreakdown): boolean {
  return !(breakdown.context_max > 0)
}

const RETRY_DELAYS_MS = [1_000, 4_000, 12_000]

/** Per-session invalidation generations. `invalidateContextBreakdown` bumps
 *  the generation for one session and notifies subscribers; the hook
 *  subscribes for its own session, so a bump re-runs the fetch immediately. */
const invalidationGenerations = new Map<string, number>()
const invalidationListeners = new Set<(sessionId: string, generation: number) => void>()

/** Force a refetch of the context breakdown for `sessionId` on the next
 *  render. Call when the transcript is known to have changed outside a
 *  turn: after a successful `session.compress`, and on
 *  `session.reclaimed`. */
export function invalidateContextBreakdown(sessionId: string): void {
  const id = sessionId.trim()

  if (id) {
    const generation = (invalidationGenerations.get(id) ?? 0) + 1

    invalidationGenerations.set(id, generation)

    for (const listener of invalidationListeners) {
      listener(id, generation)
    }
  }
}

/** Clear all invalidation generations (test isolation — the Map is
 *  module-level and would otherwise leak across tests). */
export function _resetContextBreakdownInvalidationsForTests(): void {
  invalidationGenerations.clear()
}

export function useContextBreakdown({ busy, enabled, requestGateway, sessionId }: ContextBreakdownOptions) {
  const [fetched, setFetched] = useState<{ breakdown: ContextBreakdown; sessionId: string } | null>(null)
  const [loading, setLoading] = useState(false)
  // Bounded retry: `attempt` indexes RETRY_DELAYS_MS; advancing it re-runs the
  // fetch effect. `exhausted` marks the end state — cached numbers evicted,
  // backoff stopped, meter dark until the next legitimate trigger (session
  // change, turn end, or invalidation) starts a fresh ladder.
  const [attempt, setAttempt] = useState(0)
  const [exhausted, setExhausted] = useState(false)
  const [generation, setGeneration] = useState(0)

  // Subscribe to invalidation bumps for THIS session: a bump from
  // invalidateContextBreakdown (post-compress, reclaim) re-runs the fetch
  // effect immediately, without waiting for a busy toggle or session switch.
  useEffect(() => {
    if (!sessionId) {
      return
    }

    const listener = (id: string, next: number) => {
      if (id === sessionId) {
        setGeneration(current => (current === next ? current : next))
      }
    }

    invalidationListeners.add(listener)

    // Adopt any generation bumped before this effect subscribed (e.g. the
    // invalidation fired while the meter was hidden, or between sessions).
    const pending = invalidationGenerations.get(sessionId) ?? 0

    setGeneration(current => (current === pending ? current : pending))

    return () => {
      invalidationListeners.delete(listener)
    }
  }, [sessionId])

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
    // `attempt` + `generation` drive the refetch re-runs; everything else is a trigger.
  }, [attempt, busy, enabled, generation, requestGateway, sessionId])

  // While the retry ladder is still running, the last good breakdown stays on
  // screen (the meter ticking down beats flickering dark for a transient window);
  // once exhausted the cache is gone, so this returns null and the meter goes
  // dark honestly.
  return {
    breakdown: fetched && fetched.sessionId === sessionId ? fetched.breakdown : null,
    loading
  }
}