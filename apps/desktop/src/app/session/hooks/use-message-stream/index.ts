import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { translateNow } from '@/i18n'
import {
  appendAssistantTextPart,
  appendReasoningPart,
  assistantTextPart,
  type ChatMessage,
  type ChatMessagePart,
  chatMessageText,
  type GatewayEventPayload,
  mergeFinalAssistantText,
  reasoningPart,
  renderMediaTags,
  sealOpenToolParts,
  upsertToolPart
} from '@/lib/chat-messages'
import {
  dedupeGeneratedImageEchoesInParts,
  generatedImageEchoSources,
  stripGeneratedImageEchoes
} from '@/lib/generated-images'
import { parseTodos } from '@/lib/todos'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { isDiskFullErrorMessage, notifyError } from '@/store/notifications'
import { broadcastSessionsChanged } from '@/store/session-sync'
import { upsertSubagent } from '@/store/subagents'
import { setSessionTodos } from '@/store/todos'

import type { ClientSessionState } from '../../../types'

import { useGatewayEventHandler } from './gateway-event'
import { completionErrorText, delegateTaskPayloads, MAX_STREAM_FLUSH_GAP_MS, STREAM_DELTA_FLUSH_MS } from './utils'

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
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

interface QueuedStreamDeltas {
  assistant: string
  reasoning: string
}

// Date.now() alone can collide when an interim seal and the next segment's
// first delta land in the same millisecond — the new segment would then find
// the sealed bubble by id and append into it instead of starting fresh.
let streamMessageSeq = 0

const nextStreamMessageId = (prefix: string) => `${prefix}-${Date.now()}-${++streamMessageSeq}`

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

  // Patch the in-flight assistant message (or seed it). Centralises the
  // streamId/groupId bookkeeping every event callback would otherwise repeat.
  const mutateStream = useCallback(
    (
      sessionId: string,
      transform: (parts: ChatMessagePart[], message: ChatMessage) => ChatMessagePart[],
      seed: () => ChatMessagePart[],
      opts: {
        pending?: (message: ChatMessage) => boolean
      } = {}
    ) => {
      const apply = () => {
        updateSessionState(sessionId, state => {
          // After a stop, drop any late deltas / tool events for the
          // cancelled turn so they don't keep growing the (now finalized)
          // assistant bubble or, worse, seed a brand-new bubble that
          // appears to belong to the next user message.
          if (state.interrupted) {
            return state
          }

          const streamId = state.streamId ?? nextStreamMessageId('assistant-stream')
          const groupId = state.pendingBranchGroup ?? undefined
          const prev = state.messages
          let nextMessages: ChatMessage[]

          if (!prev.some(m => m.id === streamId)) {
            nextMessages = [
              ...prev,
              {
                id: streamId,
                role: 'assistant',
                parts: seed(),
                pending: true,
                branchGroupId: groupId
              }
            ]
          } else {
            nextMessages = prev.map(m =>
              m.id === streamId
                ? {
                    ...m,
                    parts: transform(m.parts, m),
                    pending: opts.pending ? opts.pending(m) : true
                  }
                : m
            )
          }

          return {
            ...state,
            messages: nextMessages,
            streamId,
            sawAssistantPayload: true,
            awaitingResponse: false
          }
        })
      }

      apply()
    },
    [updateSessionState]
  )

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

  const queuedDeltasRef = useRef<Map<string, QueuedStreamDeltas>>(new Map())
  const flushHandleRef = useRef<number | null>(null)
  const lastFlushAtRef = useRef<number>(0)
  // What the previous flush cost on the main thread — drives the adaptive
  // flush floor in scheduleDeltaFlush so multi-stream load yields to input.
  const lastFlushCostRef = useRef<number>(0)
  // The pending commit-cost measurement rAF, so a newer flush (or unmount)
  // can cancel it instead of letting parked callbacks pile up while hidden.
  const measureRafRef = useRef<number | null>(null)
  const nativeSubagentSessionsRef = useRef<Set<string>>(new Set())
  // Turns that auto-compacted: skip post-turn hydrate so live scrollback survives.
  const compactedTurnRef = useRef<Set<string>>(new Set())
  // Last session we applied a session.info cwd for — lets us tell an agent
  // relocating the SAME session (follow it) from a session switch (don't yank).
  const lastCwdInfoSessionRef = useRef<null | string>(null)

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

        if (queued.assistant) {
          mutateStream(
            id,
            parts => dedupeGeneratedImageEchoesInParts(appendAssistantTextPart(parts, queued.assistant)),
            () => [assistantTextPart(queued.assistant)]
          )
        }

        if (queued.reasoning) {
          mutateStream(
            id,
            parts => appendReasoningPart(parts, queued.reasoning),
            () => [reasoningPart(queued.reasoning)]
          )
        }
      }
    },
    [mutateStream]
  )

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
    (sessionId: string, key: keyof QueuedStreamDeltas, delta: string) => {
      if (!delta) {
        return
      }

      const queued = queuedDeltasRef.current.get(sessionId) ?? { assistant: '', reasoning: '' }
      queued[key] += delta
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
    (sessionId: string, delta: string) => {
      if (!delta) {
        return
      }

      queueDelta(sessionId, 'assistant', delta)
    },
    [queueDelta]
  )

  const appendReasoningDelta = useCallback(
    (sessionId: string, delta: string, replace = false) => {
      if (!delta) {
        return
      }

      if (!replace) {
        queueDelta(sessionId, 'reasoning', delta)

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
            return [...parts.filter(part => part.type !== 'reasoning'), reasoningPart(delta)]
          }

          return appendReasoningPart(parts, delta)
        },
        () => [reasoningPart(delta)]
      )
    },
    [flushQueuedDeltas, mutateStream, queueDelta]
  )

  const upsertToolCall = useCallback(
    (
      sessionId: string,
      payload: GatewayEventPayload | undefined,
      phase: 'running' | 'complete',
      sourceEventType?: string
    ) => {
      // Text deltas flush on a timer but tool events apply now; flush first so
      // a tool part can't jump ahead of the text that preceded it.
      flushQueuedDeltas(sessionId)

      if (sessionInterrupted(sessionId)) {
        return
      }

      // The composer status stack owns todo display now (no inline panel) —
      // mirror every todo state the tool reports into its session store.
      if (payload?.name === 'todo') {
        const todos = parseTodos(payload.todos) ?? parseTodos(payload.result) ?? parseTodos(payload.args)

        if (todos) {
          setSessionTodos(sessionId, todos)
        }
      }

      if (!nativeSubagentSessionsRef.current.has(sessionId)) {
        for (const subagentPayload of delegateTaskPayloads(payload, phase, sourceEventType)) {
          upsertSubagent(
            sessionId,
            subagentPayload,
            true,
            phase === 'complete' ? 'delegate.complete' : 'delegate.running'
          )
        }
      }

      mutateStream(
        sessionId,
        parts => dedupeGeneratedImageEchoesInParts(upsertToolPart(parts, payload, phase)),
        () => upsertToolPart([], payload, phase),
        { pending: m => phase !== 'complete' || (m.pending ?? false) }
      )
    },
    [flushQueuedDeltas, mutateStream, sessionInterrupted]
  )

  const finalizeInterimAssistantMessage = useCallback(
    (sessionId: string, text: string) => {
      updateSessionState(sessionId, state => {
        if (state.interrupted) {
          return state
        }

        const authoritativeText = renderMediaTags(text).trim()

        if (!authoritativeText) {
          return state
        }

        const streamId = state.streamId

        const replaceTextPart = (parts: ChatMessagePart[]) => {
          const visibleText = stripGeneratedImageEchoes(authoritativeText, generatedImageEchoSources(parts)).trim()

          return mergeFinalAssistantText(parts, visibleText)
        }

        let nextMessages = state.messages

        if (streamId && nextMessages.some(m => m.id === streamId)) {
          // Seal the streaming bubble in place, marked interim so it renders
          // without an action footer (see ChatMessage.interim).
          nextMessages = nextMessages.map(m =>
            m.id === streamId ? { ...m, parts: replaceTextPart(m.parts), pending: false, interim: true } : m
          )
        } else {
          // No streaming bubble — create a standalone interim message
          nextMessages = [
            ...nextMessages,
            {
              id: nextStreamMessageId('assistant-interim'),
              role: 'assistant' as const,
              parts: [assistantTextPart(authoritativeText)],
              pending: false,
              interim: true,
              branchGroupId: state.pendingBranchGroup ?? undefined
            }
          ]
        }

        return {
          ...state,
          messages: nextMessages,
          streamId: null,
          interimBoundaryPending: true,
          sawAssistantPayload: state.sawAssistantPayload || Boolean(authoritativeText)
        }
      })
    },
    [updateSessionState]
  )

  const completeAssistantMessage = useCallback(
    (sessionId: string, text: string, responsePreviewed?: boolean, failure?: { error: string; partial: boolean }) => {
      let shouldHydrate = false

      const completedState = updateSessionState(sessionId, state => {
        // Late completion from an already-cancelled turn: cancelRun has
        // already finalized the bubble (kept the partial text, dropped it if
        // empty). Re-running the dedupe below would replace the partial with
        // the just-cancelled full text, so we settle and bail instead.
        if (state.interrupted) {
          return {
            ...state,
            awaitingResponse: false,
            busy: false,
            needsInput: false,
            pendingBranchGroup: null,
            streamId: null,
            turnStartedAt: null
          }
        }

        const streamId = state.streamId
        const finalText = renderMediaTags(text).trim()
        // Structured failure from the terminal frame wins over the legacy text
        // heuristic ("Error: <provider detail>" texts don't match the regexes).
        const completionError = failure?.error ?? completionErrorText(finalText)
        // A partial failure's `text` is streamed output the user should keep,
        // not the error string — settle it like a normal reply AND mark the
        // bubble failed, instead of stripping the text.
        const keepFailedPartialText = Boolean(failure?.partial && finalText)
        const interimBoundaryPending = state.interimBoundaryPending

        const replaceTextPart = (parts: ChatMessagePart[]) => {
          const visibleFinalText = stripGeneratedImageEchoes(finalText, generatedImageEchoSources(parts)).trim()

          return mergeFinalAssistantText(parts, visibleFinalText)
        }

        // Settling the final response onto a bubble makes it the turn's real
        // reply — clear `interim` so it regains the action footer.
        const completeMessage = (message: ChatMessage): ChatMessage => {
          const settled = { ...message, pending: false, interim: false }

          if (completionError && !keepFailedPartialText) {
            return { ...settled, error: completionError, parts: message.parts.filter(part => part.type !== 'text') }
          }

          return {
            ...settled,
            parts: replaceTextPart(message.parts),
            ...(completionError ? { error: completionError } : {})
          }
        }

        const newAssistantFromCompletion = (): ChatMessage => ({
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          parts: completionError && !keepFailedPartialText ? [] : [assistantTextPart(finalText)],
          branchGroupId: state.pendingBranchGroup ?? undefined,
          ...(completionError && { error: completionError })
        })

        const prev = state.messages
        let nextMessages = prev

        if (streamId && prev.some(m => m.id === streamId)) {
          nextMessages = prev.map(m => (m.id === streamId ? completeMessage(m) : m))
        } else {
          const fallbackIndex = [...prev]
            .reverse()
            .findIndex(message => message.role === 'assistant' && !message.hidden)

          if (fallbackIndex >= 0) {
            const index = prev.length - 1 - fallbackIndex
            const existing = prev[index]
            const existingText = chatMessageText(existing).trim()

            // The last assistant row is a sealed interim (a tool-call turn or a
            // verify-on-stop candidate — `message.interim` fires for BOTH, see
            // tui_gateway `_load_interim_assistant_messages`). When the final
            // completion is the SAME turn's reply, settle it onto that interim
            // instead of appending a second bubble. Continuity, not exact
            // equality: streaming can drop characters and the final may add a
            // trailing delta, so treat prefix-either-way as the same message.
            // (mergeFinalAssistantText, via completeMessage, does the real
            // text merge — replaces the interim's text with the full final.)
            const finalContinuesInterim = Boolean(
              existing.interim &&
              finalText &&
              existingText &&
              (finalText === existingText || finalText.startsWith(existingText) || existingText.startsWith(finalText))
            )

            if (existing.pending || (!interimBoundaryPending && finalText && existingText === finalText)) {
              nextMessages = prev.map((message, messageIndex) =>
                messageIndex === index ? completeMessage(message) : message
              )
            } else if ((interimBoundaryPending && responsePreviewed) || finalContinuesInterim) {
              // Settle the interim in place instead of creating a duplicate —
              // the DB has one row, so the live UI must agree. Two distinct
              // settle paths with different boundary requirements:
              //
              // • responsePreviewed covers the verify-on-stop continuation-
              //   budget case, where the final may be a rewrite sharing no
              //   prefix with the interim. Because there is no continuity
              //   guarantee, it must stay gated on the session's
              //   `interimBoundaryPending` flag: after a new `message.start`
              //   resets the flag, a previewed final is a DISTINCT reply and
              //   must append its own bubble, never overwrite the interim
              //   (otherwise interim('old') → message.start →
              //   complete({response_previewed: true, text: 'new'}) would
              //   silently destroy 'old').
              //
              // • finalContinuesInterim (prefix-either-way continuity, same
              //   text or one a prefix of the other) is safe to settle
              //   flag-free: continuity can only hold for the SAME message,
              //   so a `message.start` reset landing between this turn's
              //   `message.interim` and `message.complete` must not force an
              //   append of a duplicate bubble (#74560). This also closes the
              //   non-previewed tool-call gap from #63679.
              nextMessages = prev.map((message, messageIndex) =>
                messageIndex === index ? completeMessage(message) : message
              )
            } else if (finalText) {
              nextMessages = [...prev, newAssistantFromCompletion()]
            }
          } else if (finalText) {
            nextMessages = [...prev, newAssistantFromCompletion()]
          }
        }

        // Turn-settle reconciliation: a `tool.complete` event lost to a
        // degraded websocket leaves its tool row spinning forever. The turn is
        // provably done here — nothing can still be running — so seal any
        // tool-call parts that never saw their completion event.
        nextMessages = sealOpenToolParts(nextMessages)

        const hasInlineError = nextMessages.some(m => m.role === 'assistant' && m.error && !m.hidden)
        const lastVisible = [...nextMessages].reverse().find(m => !m.hidden)
        const unresolvedUserTail = lastVisible?.role === 'user'
        // Having streamed the reply normally means this window owns the whole
        // turn and re-reading stored history would be wasted work. That only
        // holds for a turn it STARTED: an adopted one (resumed onto a session
        // already running elsewhere) arrives reply-first, with no prompt row,
        // so it has to hydrate or the user's own message never shows up.
        shouldHydrate =
          !completionError &&
          !hasInlineError &&
          !unresolvedUserTail &&
          (state.adoptedRunningTurn || !state.sawAssistantPayload || !finalText)

        return {
          ...state,
          messages: nextMessages,
          adoptedRunningTurn: false,
          streamId: null,
          pendingBranchGroup: null,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          interimBoundaryPending: false,
          turnStartedAt: null
        }
      })

      // Persistence / mid-turn disk-full failures land as a terminal frame with
      // an error string, not a rejected prompt.submit. Toast them here so a
      // full disk never looks like a silent no-reply. Only fire on actual
      // failure signals — never on a healthy reply that happens to say
      // "disk full".
      const diskFullSignal = failure?.error || (failure ? text : '')

      if (diskFullSignal && isDiskFullErrorMessage(diskFullSignal)) {
        notifyError(new Error(diskFullSignal), translateNow('notifications.errors.diskFull'))
      }

      scheduleSessionsRefresh()

      if (compactedTurnRef.current.delete(sessionId)) {
        shouldHydrate = false
      }

      if (shouldHydrate) {
        void hydrateFromStoredSession(3, completedState.storedSessionId, sessionId)
      }

      dispatchNativeNotification({
        body: text.slice(0, 140) || translateNow('notifications.native.turnDoneBody'),
        kind: 'turnDone',
        sessionId,
        title: translateNow('notifications.native.turnDoneTitle')
      })
    },
    [hydrateFromStoredSession, scheduleSessionsRefresh, updateSessionState]
  )

  const failAssistantMessage = useCallback(
    (sessionId: string, errorMessage: string) => {
      updateSessionState(sessionId, state => {
        const streamId = state.streamId ?? `assistant-error-${Date.now()}`
        const groupId = state.pendingBranchGroup ?? undefined
        const prev = state.messages
        const error = errorMessage.trim() || 'Hermes reported an error'

        const nextMessages = prev.some(m => m.id === streamId)
          ? prev.map(message =>
              message.id === streamId
                ? {
                    ...message,
                    error,
                    pending: false
                  }
                : message
            )
          : [
              ...prev,
              {
                id: streamId,
                role: 'assistant' as const,
                parts: [],
                error,
                pending: false,
                branchGroupId: groupId
              }
            ]

        return {
          ...state,
          messages: nextMessages,
          streamId: null,
          pendingBranchGroup: null,
          sawAssistantPayload: true,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          interimBoundaryPending: false,
          turnStartedAt: null
        }
      })
    },
    [updateSessionState]
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
    finalizeInterimAssistantMessage,
    queryClient,
    refreshHermesConfig,
    sessionInterrupted,
    sessionStateByRuntimeIdRef,
    updateSessionState,
    upsertToolCall
  })

  return {
    appendAssistantDelta,
    appendReasoningDelta,
    completeAssistantMessage,
    handleGatewayEvent,
    finalizeInterimAssistantMessage,
    upsertToolCall
  }
}
