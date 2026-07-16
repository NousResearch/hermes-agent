import { useEffect, useLayoutEffect, useRef, useState } from 'react'

// DOM is bounded by a rendered-PART budget, not a message/turn count: a single
// assistant message folds every tool call into a part, so heavy sessions are
// ~40 turns / ~100 messages but ~1000 parts — and parts are what drive node
// count. This is the steady-state window ("Show earlier" adds another page).
export const RENDER_BUDGET = 300

// First paint after a session switch renders only the newest tail. Profiled on
// a 1,294-message session (MBP, 2026-07-15): committing the full 300-part
// budget synchronously held the main thread ~8s — dominated by the sticky
// user-bubble clamp measure cascade, Shiki tokenization of every visible code
// fence, and the resulting reflow/GC churn, in 350-640ms long tasks. The tail
// slice cuts that first commit to a fraction; the rest of the window arrives
// via the idle-time raise below, invisibly (the user is pinned to the bottom).
export const SWITCH_RENDER_BUDGET = 80

// Each idle raise mounts at most this many additional parts. Old groups keep
// their keys and skip re-render, so a step's cost is only the newly mounted
// parts — stepping keeps every post-paint commit small enough that typing and
// clicks stay responsive while history backfills.
export const RAISE_STEP = 110

// requestIdleCallback ceiling: if the main thread never goes idle (e.g. a
// stream is running), raise anyway so "Show earlier" reach isn't degraded.
const RAISE_IDLE_TIMEOUT_MS = 2000

// setTimeout fallback when requestIdleCallback is unavailable: long enough to
// clear the switch commit + settle frames, short enough to feel immediate.
const RAISE_FALLBACK_MS = 400

type IdleWindow = Window & {
  cancelIdleCallback?: (handle: number) => void
  requestIdleCallback?: (callback: () => void, opts?: { timeout: number }) => number
}

/**
 * Two-phase transcript render budget — the session-SWITCH perf lever.
 *
 * Phase 1: every session switch resets the budget to SWITCH_RENDER_BUDGET so
 * the first commit is cheap. Phase 2: idle-time steps of RAISE_STEP raise it
 * to RENDER_BUDGET. `onBeforeRaise` runs synchronously before each step
 * commits so the caller can record scroll geometry (the step prepends content
 * above the viewport).
 *
 * The raise never lowers a budget the user already grew via "Show earlier".
 */
export function useTwoPhaseRenderBudget(
  sessionKey: string | null | undefined,
  onBeforeRaise: () => void
): [number, (updater: (budget: number) => number) => void] {
  const [renderBudget, setRenderBudget] = useState(SWITCH_RENDER_BUDGET)
  const onBeforeRaiseRef = useRef(onBeforeRaise)
  // Read by the idle step below. A React state UPDATER runs during the next
  // render, not synchronously at the setState call — deciding "am I done?"
  // from an updater side-channel breaks the reschedule chain. The ref reflects
  // the committed budget every render, so the step's check is reliable.
  const budgetRef = useRef(renderBudget)
  // On a switch, layout effects run BEFORE the previous effect's cleanup gets
  // to cancel its pending idle callback — a stale callback firing in that gap
  // would raise the new session's freshly reset budget. Each effect captures
  // its own key and its step bails when this ref has moved on.
  const liveKeyRef = useRef(sessionKey)

  onBeforeRaiseRef.current = onBeforeRaise
  budgetRef.current = renderBudget

  // Reset to the slim budget on mount + every switch. Layout effect so the
  // reset commits in the same pass as the caller's scroll pinning.
  useLayoutEffect(() => {
    liveKeyRef.current = sessionKey
    budgetRef.current = SWITCH_RENDER_BUDGET
    setRenderBudget(SWITCH_RENDER_BUDGET)
  }, [sessionKey])

  useEffect(() => {
    const idleWindow = window as IdleWindow
    const useIdle = typeof idleWindow.requestIdleCallback === 'function'
    const ownKey = sessionKey
    let handle: number | null = null
    let cancelled = false

    const schedule = (callback: () => void) => {
      handle = useIdle
        ? (idleWindow.requestIdleCallback as NonNullable<IdleWindow['requestIdleCallback']>)(callback, {
            timeout: RAISE_IDLE_TIMEOUT_MS
          })
        : window.setTimeout(callback, RAISE_FALLBACK_MS)
    }

    const step = () => {
      if (cancelled || liveKeyRef.current !== ownKey || budgetRef.current >= RENDER_BUDGET) {
        return
      }

      onBeforeRaiseRef.current()
      setRenderBudget(budget => (budget < RENDER_BUDGET ? Math.min(budget + RAISE_STEP, RENDER_BUDGET) : budget))
      // Reschedule unconditionally; the next step no-ops (and stops the chain)
      // once the committed budget reaches RENDER_BUDGET.
      schedule(step)
    }

    schedule(step)

    return () => {
      cancelled = true

      if (handle !== null) {
        if (useIdle) {
          idleWindow.cancelIdleCallback?.(handle)
        } else {
          window.clearTimeout(handle)
        }
      }
    }
  }, [sessionKey])

  return [renderBudget, setRenderBudget]
}
