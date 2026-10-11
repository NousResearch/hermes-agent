import { useCallback, useEffect, useRef } from 'react'

import {
  appendAssistantTextPart,
  appendReasoningPart,
  type ChatMessagePart,
  chatMessageText,
  reasoningPart
} from '@/lib/chat-messages'
import { dedupeGeneratedImageEchoesInParts } from '@/lib/generated-images'

import type { MutateStream } from './mutate-stream'
import { MAX_STREAM_FLUSH_GAP_MS, STREAM_DELTA_FLUSH_MS } from './utils'

interface QueuedStreamDelta {
  occurredAt: number
  text: string
  type: 'assistant' | 'reasoning'
}

export function useDeltaFlush(mutateStream: MutateStream) {
  const queuedDeltasRef = useRef<Map<string, QueuedStreamDelta[]>>(new Map())

  const flushHandleRef = useRef<number | null>(null)

  const lastFlushAtRef = useRef<number>(0)

  // What the previous flush cost on the main thread — drives the adaptive
  // flush floor in scheduleDeltaFlush so multi-stream load yields to input.
  const lastFlushCostRef = useRef<number>(0)

  // The pending commit-cost measurement rAF, so a newer flush (or unmount)
  // can cancel it instead of letting parked callbacks pile up while hidden.
  const measureRafRef = useRef<number | null>(null)

  const flushQueuedDeltas = useCallback(
    (sessionId?: string) => {
      const queue = queuedDeltasRef.current
      const ids = sessionId ? [sessionId] : [...queue.keys()]

      for (const id of ids) {
        const queued = queue.get(id)

        if (!queued) {
          continue
        }

        queue.delete(id)

        const applyQueued = (parts: ChatMessagePart[]) =>
          queued.reduce(
            (next, delta) =>
              delta.type === 'assistant'
                ? dedupeGeneratedImageEchoesInParts(appendAssistantTextPart(next, delta.text, delta.occurredAt))
                : appendReasoningPart(next, delta.text, delta.occurredAt),
            parts
          )

        mutateStream(id, applyQueued, () => applyQueued([]), {}, queued[0]?.occurredAt)
      }
    },
    [mutateStream]
  )

  // Turn-boundary orphan drop (#119543): discard queued bytes without
  // painting them. Used when a new turn starts while no turn is live — the
  // queue can only hold stragglers of the superseded attempt then.
  const dropQueuedDeltas = useCallback((sessionId?: string) => {
    const queue = queuedDeltasRef.current

    if (sessionId) {
      queue.delete(sessionId)

      return
    }

    queue.clear()
  }, [])

  const scheduleDeltaFlush = useCallback(() => {
    if (flushHandleRef.current !== null) {
      return
    }

    if (typeof window === 'undefined') {
      flushQueuedDeltas()

      return
    }

    // Enforce a floor on the gap between two flushes. Without it, an LLM
    // emitting tokens slower than the rAF cadence (~30-80 tok/sec is typical)
    // forces one React commit + Streamdown re-parse per token, and the
    // last-block markdown re-parse cost is roughly linear in current block
    // length. With this floor, slower streams still coalesce ~2 tokens per
    // commit and the synthetic harness shows longtask counts drop from ~5/5s
    // to ~1/5s on big sessions (see scripts/profile-typing-lag.md).
    //
    // ADAPTIVE: the floor scales with what the last flush actually cost.
    // With several sessions streaming at once (split tiles), one flush carries
    // every stream's commit + markdown re-parse; when that work approaches or
    // exceeds the fixed 33ms budget, back-to-back flushes leave the main
    // thread no idle frames and every interaction (typing, resize, hover)
    // stutters even though no render is wasted. Yielding 3x the measured cost
    // keeps the thread ~75% idle for input at any load: cheap flushes stay at
    // 30fps of text growth, expensive multi-stream flushes degrade text fps
    // instead of interactivity — capped so text never updates slower than 4/s.
    // The cost has to include the deferred view-sync frame where the commit
    // actually happens; see runFlush below.
    const sinceLast = performance.now() - lastFlushAtRef.current

    const adaptiveFloor = Math.min(
      Math.max(STREAM_DELTA_FLUSH_MS, lastFlushCostRef.current * 3),
      MAX_STREAM_FLUSH_GAP_MS
    )

    const runFlush = () => {
      flushHandleRef.current = null
      const startedAt = performance.now()
      lastFlushAtRef.current = startedAt
      flushQueuedDeltas()
      // The store write above is only the cheap half of a flush. While a
      // session streams, syncSessionStateToView defers the $messages publish
      // (and with it the React commit + Streamdown re-parse the floor is meant
      // to account for) to its own rAF inside updateSessionState, which runs
      // after this timer task. Stopping the clock here pins lastFlushCostRef
      // near zero and collapses the adaptive floor to 33ms no matter the load.
      // Our rAF is registered after the view-sync one, so it runs in the same
      // frame right after that commit; its timestamp marks frame start, so
      // (now - frameStart) counts only work done inside the frame, not the
      // vsync wait. A hidden renderer never fires rAF, so the write cost
      // stays as the fallback.
      const writeCost = performance.now() - startedAt
      lastFlushCostRef.current = writeCost

      // At most one measurement rAF may be pending: only the newest flush's
      // measurement matters (the guard below discards stale frames), and a
      // hidden renderer parks rAF callbacks — without cancellation a long
      // hidden stream at the floor would accumulate thousands of parked
      // closures that all fire in the first frame on refocus.
      if (measureRafRef.current !== null) {
        window.cancelAnimationFrame(measureRafRef.current)
      }

      measureRafRef.current = window.requestAnimationFrame(frameStart => {
        measureRafRef.current = null

        // A newer flush already started; its own measurement wins.
        if (lastFlushAtRef.current !== startedAt) {
          return
        }

        lastFlushCostRef.current = writeCost + Math.max(0, performance.now() - frameStart)
      })
    }

    // Always a timer, never requestAnimationFrame. Chromium pauses rAF for a
    // renderer it considers hidden, and "hidden" is not something this code can
    // verify: while a turn is in flight the main process unthrottles every chat
    // window (stream-throttle.ts), but that doesn't guarantee frames for a
    // minimized window, a fully off-screen one, or a renderer the compositor
    // has otherwise parked. In those states an rAF-gated flush never runs, so a
    // finished answer sits in this queue until some later input or focus event
    // happens to wake a frame — the reply looks stalled, then arrives all at
    // once on refocus.
    //
    // A timer keeps the same coalescing cadence (that's what the floor above is
    // for) while guaranteeing delivery without user interaction. Timers are
    // clamped in background renderers rather than suspended, and the
    // stream-aware unthrottle lifts even that clamp for the life of the turn;
    // in the worst case (a delta arriving before the unthrottle lands) the
    // clamp only stretches one flush to ~1s in a window nobody can see.
    flushHandleRef.current = window.setTimeout(runFlush, Math.max(0, adaptiveFloor - sinceLast))
  }, [flushQueuedDeltas])

  const queueDelta = useCallback(
    (sessionId: string, key: 'assistant' | 'reasoning', delta: string, occurredAt = Date.now() / 1000) => {
      if (!delta) {
        return
      }

      const queued = queuedDeltasRef.current.get(sessionId) ?? []
      const tail = queued.at(-1)

      if (tail?.type === key) {
        tail.text += delta
      } else {
        queued.push({ occurredAt, text: delta, type: key })
      }

      queuedDeltasRef.current.set(sessionId, queued)
      scheduleDeltaFlush()
    },
    [scheduleDeltaFlush]
  )

  useEffect(
    () => () => {
      if (flushHandleRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(flushHandleRef.current)
      }

      flushHandleRef.current = null

      if (measureRafRef.current !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(measureRafRef.current)
      }

      measureRafRef.current = null
      flushQueuedDeltas()
    },
    [flushQueuedDeltas]
  )

  // Page Visibility does not report every Windows/Linux focus transition.
  // Flush queued deltas on both signals so returning to a chat cannot leave a
  // completed chunk waiting for the next throttled timer.
  // eslint-disable-next-line no-restricted-syntax -- timer-handle clear inside effect, not an atom mirror
  useEffect(() => {
    const flushPendingDeltas = () => {
      if (flushHandleRef.current !== null) {
        window.clearTimeout(flushHandleRef.current)
        flushHandleRef.current = null
      }

      flushQueuedDeltas()
    }

    const flushWhenVisible = () => {
      if (document.visibilityState === 'visible') {
        flushPendingDeltas()
      }
    }

    document.addEventListener('visibilitychange', flushWhenVisible)
    window.addEventListener('focus', flushPendingDeltas)

    return () => {
      document.removeEventListener('visibilitychange', flushWhenVisible)
      window.removeEventListener('focus', flushPendingDeltas)
    }
  }, [flushQueuedDeltas])

  const appendAssistantDelta = useCallback(
    (sessionId: string, delta: string, occurredAt?: number) => {
      if (!delta) {
        return
      }

      queueDelta(sessionId, 'assistant', delta, occurredAt)
    },
    [queueDelta]
  )

  const appendReasoningDelta = useCallback(
    (sessionId: string, delta: string, replace = false, occurredAt = Date.now() / 1000) => {
      if (!delta) {
        return
      }

      if (!replace) {
        queueDelta(sessionId, 'reasoning', delta, occurredAt)

        return
      }

      flushQueuedDeltas(sessionId)

      mutateStream(
        sessionId,
        (parts, message) => {
          if (replace && chatMessageText(message).trim()) {
            return parts
          }

          if (replace) {
            return [...parts.filter(part => part.type !== 'reasoning'), reasoningPart(delta, occurredAt)]
          }

          return appendReasoningPart(parts, delta, occurredAt)
        },
        () => [reasoningPart(delta, occurredAt)],
        {},
        occurredAt
      )
    },
    [flushQueuedDeltas, mutateStream, queueDelta]
  )

  return { flushQueuedDeltas, dropQueuedDeltas, appendAssistantDelta, appendReasoningDelta }
}
