import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { translateNow } from '@/i18n'
import {
  appendAssistantTextPart,
  appendReasoningPart,
  assistantTextPart,
  type ChatMessage,
  type ChatMessagePart,
  chatMessageText,
  type GatewayEventPayload,
  reasoningPart,
  renderMediaTags,
  textPart,
  upsertToolPart
} from '@/lib/chat-messages'
import { coerceGatewayText, coerceThinkingText, normalizePersonalityValue } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import { gatewayEventRequiresSessionId } from '@/lib/gateway-events'
import {
  dedupeGeneratedImageEchoesInParts,
  generatedImageEchoSources,
  stripGeneratedImageEchoes
} from '@/lib/generated-images'
import { triggerHaptic } from '@/lib/haptics'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { parseTodos } from '@/lib/todos'
import { setClarifyRequest } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { clearAllPrompts, setApprovalRequest, setSecretRequest, setSudoRequest } from '@/store/prompts'
import {
  $compressingStatus,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setSelectedStoredSessionId,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import { broadcastSessionsChanged } from '@/store/session-sync'
import { clearSessionSubagents, pruneDelegateFallbackSubagents, upsertSubagent } from '@/store/subagents'
import { setSessionTodos } from '@/store/todos'
import { recordToolDiff } from '@/store/tool-diffs'
import type { RpcEvent } from '@/types/hermes'

import { sessionRoute } from '../../routes'
import type { ClientSessionState } from '../../types'

interface MessageStreamOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>
  navigate: (path: string) => void
  queryClient: QueryClient
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
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

type SessionRuntimeStatePatch = Partial<
  Pick<
    ClientSessionState,
    'branch' | 'cwd' | 'fast' | 'model' | 'personality' | 'provider' | 'reasoningEffort' | 'serviceTier' | 'yolo'
  >
>

function sessionInfoStatePatch(payload: GatewayEventPayload | undefined): SessionRuntimeStatePatch {
  const patch: SessionRuntimeStatePatch = {}

  if (typeof payload?.model === 'string') {
    patch.model = payload.model || ''
  }

  if (typeof payload?.provider === 'string') {
    patch.provider = payload.provider || ''
  }

  if (typeof payload?.cwd === 'string') {
    patch.cwd = payload.cwd
  }

  if (typeof payload?.branch === 'string') {
    patch.branch = payload.branch
  }

  if (typeof payload?.personality === 'string') {
    patch.personality = normalizePersonalityValue(payload.personality)
  }

  if (typeof payload?.reasoning_effort === 'string') {
    patch.reasoningEffort = payload.reasoning_effort
  }

  if (typeof payload?.service_tier === 'string') {
    patch.serviceTier = payload.service_tier
  }

  if (typeof payload?.fast === 'boolean') {
    patch.fast = payload.fast
  }

  if (typeof payload?.yolo === 'boolean') {
    patch.yolo = payload.yolo
  }

  return patch
}

function hasSessionInfoStatePatch(patch: SessionRuntimeStatePatch): boolean {
  return Object.keys(patch).length > 0
}

// Minimum gap between two assistant-text flushes during a stream. Was 16ms
// (rAF only), which at typical LLM token rates of ~30-80 tok/sec meant every
// token got its own React commit + Streamdown markdown re-parse, scaling
// linearly with the growing last-block length. Bumping to 33ms lets ~2 tokens
// batch into one commit at 60 tok/sec without introducing visible lag on the
// streaming text (still 30 fps of visible text growth). Big perceived
// smoothness win on long messages with big trailing paragraphs; see
// `scripts/profile-typing-lag.md` for the measurement work behind this.
const STREAM_DELTA_FLUSH_MS = 33

// Gateway/provider failures sometimes arrive as message.complete text instead
// of an explicit error event. Treat matches as inline assistant errors so they
// persist like real error events and don't get erased by hydrate fallback.
const COMPLETION_ERROR_PATTERNS = [
  /^API call failed after \d+ retries:/i,
  /^HTTP\s+\d{3}\b/i,
  /^(Provider|Gateway)\s+error:/i
]

function completionErrorText(finalText: string): string | null {
  const text = finalText.trim()

  return text && COMPLETION_ERROR_PATTERNS.some(re => re.test(text)) ? text : null
}

const SUBAGENT_EVENT_TYPES = new Set([
  'subagent.spawn_requested',
  'subagent.start',
  'subagent.thinking',
  'subagent.tool',
  'subagent.progress',
  'subagent.complete'
])

// Anonymous progress events that carry todos but no name still belong to the
// todo stream; named todo events are obviously routed there too.
function toTodoPayload(payload: GatewayEventPayload | undefined): GatewayEventPayload | undefined {
  if (!payload) {
    return undefined
  }

  const isTodo = payload.name === 'todo' || (!payload.name && Object.hasOwn(payload, 'todos'))

  return isTodo ? { ...payload, name: 'todo', tool_id: payload.tool_id || 'todo-live' } : undefined
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function parseMaybeRecord(value: unknown): Record<string, unknown> {
  if (typeof value === 'string') {
    try {
      return asRecord(JSON.parse(value))
    } catch {
      return {}
    }
  }

  return asRecord(value)
}

const firstString = (...candidates: unknown[]): string => {
  for (const v of candidates) {
    if (typeof v === 'string' && v) {
      return v
    }
  }

  return ''
}

function delegateTaskPayloads(
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete',
  sourceEventType?: string
): Record<string, unknown>[] {
  if (payload?.name !== 'delegate_task') {
    return []
  }

  const args = parseMaybeRecord(payload.args ?? payload.input)
  const result = parseMaybeRecord(payload.result)
  const rawTasks = Array.isArray(args.tasks) ? args.tasks : []
  const tasks = rawTasks.length ? rawTasks.map(parseMaybeRecord) : [args]
  const status = phase === 'complete' ? (payload.error ? 'failed' : 'completed') : 'running'
  const toolId = payload.tool_id || payload.tool_call_id || payload.id || 'delegate_task'
  const progressText = firstString(payload.preview, payload.message, payload.context)

  const eventType =
    phase === 'complete'
      ? 'subagent.complete'
      : sourceEventType === 'tool.start'
        ? 'subagent.start'
        : 'subagent.progress'

  return tasks.map((task, index) => {
    const goal = firstString(task.goal, args.goal, payload.context) || 'Delegated task'
    const summary = firstString(result.summary, payload.summary, payload.message)

    return {
      depth: 0,
      duration_seconds: payload.duration_s,
      goal,
      status,
      subagent_id: `delegate-tool:${toolId}:${index}`,
      summary: summary || undefined,
      task_count: tasks.length,
      task_index: index,
      text: eventType === 'subagent.progress' ? progressText || goal : undefined,
      tool_name: eventType === 'subagent.start' ? 'delegate_task' : undefined,
      tool_preview: eventType === 'subagent.start' ? progressText : undefined,
      toolsets: Array.isArray(task.toolsets) ? task.toolsets : Array.isArray(args.toolsets) ? args.toolsets : [],
      event_type: eventType,
      output_tail:
        phase === 'complete' && summary
          ? [{ is_error: Boolean(payload.error), preview: summary, tool: 'delegate_task' }]
          : undefined
    }
  })
}

export function useMessageStream({
  activeSessionIdRef,
  hydrateFromStoredSession,
  navigate,
  queryClient,
  refreshHermesConfig,
  refreshSessions,
  sessionStateByRuntimeIdRef,
  selectedStoredSessionIdRef,
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

          const streamId = state.streamId ?? `assistant-stream-${Date.now()}`
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

  const queuedDeltasRef = useRef<Map<string, QueuedStreamDeltas>>(new Map())
  const flushHandleRef = useRef<number | null>(null)
  const lastFlushAtRef = useRef<number>(0)
  const nativeSubagentSessionsRef = useRef<Set<string>>(new Set())
  // Turns that auto-compacted: skip post-turn hydrate so live scrollback survives.
  const compactedTurnRef = useRef<Set<string>>(new Set())

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
    const sinceLast = performance.now() - lastFlushAtRef.current

    const runFlush = () => {
      flushHandleRef.current = null
      lastFlushAtRef.current = performance.now()
      flushQueuedDeltas()
    }

    if (sinceLast >= STREAM_DELTA_FLUSH_MS && typeof window.requestAnimationFrame === 'function') {
      flushHandleRef.current = window.requestAnimationFrame(runFlush)

      return
    }

    flushHandleRef.current = window.setTimeout(runFlush, Math.max(0, STREAM_DELTA_FLUSH_MS - sinceLast))
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
        if (typeof window.cancelAnimationFrame === 'function') {
          window.cancelAnimationFrame(flushHandleRef.current)
        } else {
          window.clearTimeout(flushHandleRef.current)
        }
      }

      flushHandleRef.current = null
      flushQueuedDeltas()
    },
    [flushQueuedDeltas]
  )

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

  const completeAssistantMessage = useCallback(
    (sessionId: string, text: string) => {
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
        const completionError = completionErrorText(finalText)
        const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()

        const replaceTextPart = (parts: ChatMessagePart[]) => {
          const visibleFinalText = stripGeneratedImageEchoes(finalText, generatedImageEchoSources(parts)).trim()
          const dedupeReference = normalize(visibleFinalText)

          const kept = parts.filter(part => {
            if (part.type === 'text') {
              return false
            }

            if (part.type !== 'reasoning' || !dedupeReference) {
              return true
            }

            const r = normalize(part.text)

            return !(r && (dedupeReference.startsWith(r) || r.startsWith(dedupeReference)))
          })

          return visibleFinalText ? [...kept, assistantTextPart(visibleFinalText)] : kept
        }

        const completeMessage = (message: ChatMessage): ChatMessage =>
          completionError
            ? {
                ...message,
                error: completionError,
                parts: message.parts.filter(part => part.type !== 'text'),
                pending: false
              }
            : {
                ...message,
                parts: replaceTextPart(message.parts),
                pending: false
              }

        const newAssistantFromCompletion = (): ChatMessage => ({
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          parts: completionError ? [] : [assistantTextPart(finalText)],
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

            if (existing.pending || (finalText && existingText === finalText)) {
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

        const hasInlineError = nextMessages.some(m => m.role === 'assistant' && m.error && !m.hidden)
        const lastVisible = [...nextMessages].reverse().find(m => !m.hidden)
        const unresolvedUserTail = lastVisible?.role === 'user'
        shouldHydrate =
          !completionError && !hasInlineError && !unresolvedUserTail && (!state.sawAssistantPayload || !finalText)

        return {
          ...state,
          messages: nextMessages,
          streamId: null,
          pendingBranchGroup: null,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          turnStartedAt: null
        }
      })

      void refreshSessions().catch(() => undefined)
      // Sync the freshly-titled row to other windows (e.g. main, when the turn
      // ran in the pop-out).
      broadcastSessionsChanged()

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
    [hydrateFromStoredSession, refreshSessions, updateSessionState]
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
          turnStartedAt: null
        }
      })
    },
    [updateSessionState]
  )

  const handleGatewayEvent = useCallback(
    (event: RpcEvent) => {
      const payload = event.payload as GatewayEventPayload | undefined
      const explicitSid = event.session_id || ''

      if (!explicitSid && gatewayEventRequiresSessionId(event.type)) {
        return
      }

      const sessionId = explicitSid || activeSessionIdRef.current
      const isActiveEvent = !!sessionId && sessionId === activeSessionIdRef.current

      if (event.type === 'gateway.ready') {
        return
      } else if (event.type === 'session.info') {
        // Apply session-scoped fields when the event targets the active
        // session, OR when it's a global broadcast and we have no session.
        const apply = explicitSid ? isActiveEvent : !activeSessionIdRef.current
        const statePatch = sessionInfoStatePatch(payload)
        const hasStatePatch = hasSessionInfoStatePatch(statePatch)
        const modelChanged = typeof payload?.model === 'string'
        const providerChanged = typeof payload?.provider === 'string'
        const runningChanged = typeof payload?.running === 'boolean'

        // Detect session rotation after /compress. The gateway includes
        // the current stored session id in session.info; when it differs
        // from the desktop's selectedStoredSessionId, the agent has
        // rotated to a new continuation session and we must follow it.
        //
        // For manual /compress the gateway sets payload.new_session_id;
        // for auto-compression during a turn it only updates
        // payload.session_id (the current session key after rotation).
        //
        // IMPORTANT: event.session_id is the runtime hex sid, NOT the
        // stored session key — never use it for rotation detection.
        const rotatedSessionId =
          (typeof payload?.new_session_id === 'string' && payload.new_session_id) ||
          (typeof payload?.session_id === 'string' && payload.session_id !== selectedStoredSessionIdRef.current
            ? payload.session_id
            : '')
        if (
          rotatedSessionId &&
          rotatedSessionId !== selectedStoredSessionIdRef.current &&
          isActiveEvent
        ) {
          setSelectedStoredSessionId(rotatedSessionId)
          selectedStoredSessionIdRef.current = rotatedSessionId
          navigate(sessionRoute(rotatedSessionId))
          void refreshSessions().catch(() => undefined)
        }

        if (apply) {
          if (modelChanged) {
            setCurrentModel(payload!.model || '')
          }

          if (providerChanged) {
            setCurrentProvider(payload!.provider || '')
          }

          if (typeof payload?.cwd === 'string') {
            setCurrentCwd(payload.cwd)
          }

          if (typeof payload?.branch === 'string') {
            setCurrentBranch(payload.branch)
          }

          if (typeof payload?.personality === 'string') {
            setCurrentPersonality(normalizePersonalityValue(payload.personality))
          }

          if (typeof payload?.reasoning_effort === 'string') {
            setCurrentReasoningEffort(payload.reasoning_effort)
          }

          if (typeof payload?.service_tier === 'string') {
            setCurrentServiceTier(payload.service_tier)
          }

          if (typeof payload?.fast === 'boolean') {
            setCurrentFastMode(payload.fast)
          }

          if (typeof payload?.yolo === 'boolean') {
            setYoloActive(payload.yolo)
          }
        }

        if (sessionId && hasStatePatch) {
          updateSessionState(sessionId, state => ({
            ...state,
            ...statePatch,
            branch: statePatch.branch ?? state.branch,
            cwd: statePatch.cwd ?? state.cwd
          }))
        }

        if (apply) {
          if (runningChanged && sessionId) {
            updateSessionState(sessionId, state => {
              const busy = Boolean(payload!.running)

              if (state.busy === busy && (busy || !state.awaitingResponse)) {
                return state
              }

              if (busy) {
                return {
                  ...state,
                  busy,
                  turnStartedAt: state.turnStartedAt ?? Date.now()
                }
              }

              if (state.awaitingResponse && !state.sawAssistantPayload) {
                return state
              }

              return {
                ...state,
                awaitingResponse: false,
                busy,
                pendingBranchGroup: null,
                streamId: null,
                turnStartedAt: null
              }
            })
          }
        }

        if (payload?.usage && (!explicitSid || isActiveEvent)) {
          setCurrentUsage(current => ({ ...current, ...payload.usage }))
        }

        if (typeof payload?.credential_warning === 'string' && payload.credential_warning) {
          requestDesktopOnboarding(payload.credential_warning)
        }

        void refreshHermesConfig()

        if (modelChanged || providerChanged) {
          void queryClient.invalidateQueries({
            queryKey: explicitSid && sessionId ? ['model-options', sessionId] : ['model-options']
          })
        }
      } else if (event.type === 'message.start') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        clearSessionSubagents(sessionId)
        setSessionCompacting(sessionId, false)
        compactedTurnRef.current.delete(sessionId)
        nativeSubagentSessionsRef.current.delete(sessionId)

        if (isActiveEvent) {
          triggerHaptic('streamStart')
        }

        updateSessionState(sessionId, state => ({
          ...state,
          busy: true,
          awaitingResponse: true,
          sawAssistantPayload: false,
          interrupted: false,
          turnStartedAt: Date.now()
        }))

        if (isActiveEvent) {
          setTurnStartedAt(Date.now())
        }
      } else if (event.type === 'message.delta') {
        if (sessionId) {
          appendAssistantDelta(sessionId, coerceGatewayText(payload?.text))
        }
      } else if (event.type === 'thinking.delta') {
        // thinking.delta carries the kawaii spinner status (face + verb from
        // KawaiiSpinner), not real reasoning. The bottom-of-thread loading
        // indicator already covers that UX, so we ignore these events to
        // avoid a duplicative "Thinking" disclosure showing spinner text.
      } else if (event.type === 'reasoning.delta') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text))
        }
      } else if (event.type === 'reasoning.available') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), true)
        }
      } else if (event.type === 'message.complete') {
        if (!sessionId) {
          return
        }

        // Turn ended — drop any blocking prompt still open for THIS session
        // (e.g. interrupted, or the approval already resolved). Scoped to the
        // session so a background turn finishing can't wipe the active chat's
        // prompt, and vice versa.
        clearAllPrompts(sessionId)
        setSessionCompacting(sessionId, false)

        flushQueuedDeltas(sessionId)

        playCompletionSound()

        const finalText = coerceGatewayText(payload?.text) || coerceGatewayText(payload?.rendered)
        completeAssistantMessage(sessionId, finalText)

        if (isActiveEvent) {
          setTurnStartedAt(null)
        }

        if (payload?.usage) {
          setCurrentUsage(current => ({ ...current, ...payload.usage }))
        }
      } else if (event.type === 'tool.start' || event.type === 'tool.progress' || event.type === 'tool.generating') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'running', event.type)
      } else if (event.type === 'tool.complete') {
        if (sessionId) {
          flushQueuedDeltas(sessionId)
          upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'complete', event.type)
          // A pending clarify blocks the turn, so the first tool.complete after
          // one is the clarify resolving — drop the "needs input" flag here so
          // the sidebar indicator clears as soon as it's answered, not only at
          // message.complete.
          updateSessionState(sessionId, state => (state.needsInput ? { ...state, needsInput: false } : state))

          // terminal/process tool calls are the only things that spawn or reap
          // background processes — sync the composer status stack right after.
          if (!sessionInterrupted(sessionId) && (payload?.name === 'terminal' || payload?.name === 'process')) {
            void refreshBackgroundProcesses(sessionId)
          }
        }

        if (typeof payload?.inline_diff === 'string' && payload.inline_diff.trim()) {
          recordToolDiff(payload.tool_id || payload.name || '', payload.inline_diff)
        }
      } else if (SUBAGENT_EVENT_TYPES.has(event.type)) {
        if (sessionId && payload && !sessionInterrupted(sessionId)) {
          if (!nativeSubagentSessionsRef.current.has(sessionId)) {
            pruneDelegateFallbackSubagents(sessionId)
          }

          nativeSubagentSessionsRef.current.add(sessionId)
          upsertSubagent(
            sessionId,
            payload as Record<string, unknown>,
            event.type === 'subagent.spawn_requested' || event.type === 'subagent.start',
            event.type
          )
        }
      } else if (event.type === 'clarify.request') {
        // Surface the clarify tool's overlay. The Python side is blocked on
        // `clarify.respond`, so without this handler the agent would hang
        // forever (see tools/clarify_tool.py + tui_gateway/server.py:_block).
        //
        // Store the request for whichever session raised it — even a background
        // one. clarify.request is a one-shot event; if we dropped it for an
        // unfocused session, that session would block on `clarify.respond`
        // indefinitely and re-focusing it could never recover (the event is
        // gone). Parking it per-session lets the user answer once they switch
        // over; the inline ClarifyTool reads the active session's entry.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
        const question = typeof payload?.question === 'string' ? payload.question : ''

        if (requestId && question) {
          setClarifyRequest({
            requestId,
            question,
            choices: Array.isArray(payload?.choices) ? payload!.choices!.filter(c => typeof c === 'string') : null,
            sessionId: sessionId ?? null
          })

          // The transcript only renders the active session, so a background
          // clarify is otherwise invisible (the row just keeps spinning like
          // it's working). Flag the session so the sidebar shows a persistent
          // "needs input" indicator on its row — works for the active session
          // too, and survives alt-tab / window blur (unlike a toast).
          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: question,
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'approval.request') {
        // Dangerous-command / execute_code approval. The Python side is blocked
        // in _await_gateway_decision() until approval.respond lands; without
        // this the agent stalls until its 5-min timeout and the tool is BLOCKED.
        // Park it per-session (like clarify) so a *background* profile's turn can
        // raise it and wait — the sidebar flags "needs input" and the inline bar
        // surfaces once the user focuses that chat.
        const command = typeof payload?.command === 'string' ? payload.command : ''
        const description = typeof payload?.description === 'string' ? payload.description : 'dangerous command'

        setApprovalRequest({
          // false only when a tirith warning forbids it; backend omits the field otherwise.
          allowPermanent: payload?.allow_permanent !== false,
          command,
          description,
          sessionId: sessionId ?? null
        })

        if (sessionId) {
          updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
        }

        dispatchNativeNotification({
          actions: [
            { id: 'approve', text: translateNow('notifications.native.approveAction') },
            { id: 'reject', text: translateNow('notifications.native.rejectAction') }
          ],
          body: command || description,
          kind: 'approval',
          sessionId,
          title: translateNow('notifications.native.approvalTitle')
        })
      } else if (event.type === 'sudo.request') {
        // Sudo password capture (tools/terminal_tool.py). Blocked on
        // sudo.respond {request_id, password}.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          setSudoRequest({ requestId, sessionId: sessionId ?? null })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'secret.request') {
        // Skill credential capture (tools/skills_tool.py). Blocked on
        // secret.respond {request_id, value}.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const envVar = typeof payload?.env_var === 'string' ? payload.env_var : ''
          const promptText = typeof payload?.prompt === 'string' ? payload.prompt : ''

          setSecretRequest({
            requestId,
            envVar,
            prompt: promptText,
            sessionId: sessionId ?? null
          })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: promptText || envVar || translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'terminal.read.request') {
        // read_terminal tool: serialize the renderer's xterm buffer and answer
        // immediately (Python blocks on the respond). Empty text = no live pane.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const start = typeof payload?.start === 'number' ? payload.start : undefined
          const count = typeof payload?.count === 'number' ? payload.count : undefined
          const result = readActiveTerminal({ start, count })

          void $gateway.get()?.request('terminal.read.respond', {
            request_id: requestId,
            text: result ? JSON.stringify(result) : ''
          })
        }
      } else if (event.type === 'status.update') {
        if (sessionId && payload?.kind === 'compacting') {
          setSessionCompacting(sessionId, true)
          compactedTurnRef.current.add(sessionId)
        } else if (sessionId && payload?.kind === 'process') {
          // The gateway's notification poller announces background process
          // completions / watch matches here — re-sync the status stack.
          void refreshBackgroundProcesses(sessionId)
        } else if (payload?.kind === 'compressing') {
          // Show compress progress in the status bar.
          $compressingStatus.set(payload.text ?? null)
        } else if (payload?.kind === 'ready') {
          // Compression finished or was cancelled — clear the indicator.
          $compressingStatus.set(null)
        }
      } else if (event.type === 'review.summary') {
        // Self-improvement background review saved something to memory/skills
        // and emitted a persistent summary (Python formats it as
        // "💾 Self-improvement review: …"). The CLI prints this via
        // prompt_toolkit and the Ink TUI renders it as a system line; the
        // desktop has neither, so without this handler the skill/memory
        // change happens silently. Surface it as a persistent system message
        // in the transcript so the user is always informed — it must not be a
        // transient toast that can be missed.
        const text = coerceGatewayText(payload?.text).trim()

        if (text && sessionId) {
          flushQueuedDeltas(sessionId)
          updateSessionState(sessionId, state => ({
            ...state,
            messages: [
              ...state.messages,
              {
                id: `review-summary-${Date.now()}`,
                role: 'system',
                parts: [textPart(text)],
                timestamp: Math.floor(Date.now() / 1000)
              }
            ]
          }))
        }
      } else if (event.type === 'error') {
        const errorMessage = payload?.message || 'Hermes reported an error'
        const looksLikeProviderSetup = isProviderSetupErrorMessage(errorMessage)

        // A turn that errors out has also ended — drop any open blocking prompt
        // for this session so an approval/sudo/secret overlay can't linger past
        // the failed turn (same intent as the message.complete clear).
        if (sessionId) {
          clearAllPrompts(sessionId)
          setSessionCompacting(sessionId, false)
          compactedTurnRef.current.delete(sessionId)
        }

        dispatchNativeNotification({
          body: errorMessage,
          kind: 'turnError',
          sessionId,
          title: translateNow('notifications.native.turnErrorTitle')
        })

        if (looksLikeProviderSetup) {
          requestDesktopOnboarding(errorMessage)
        } else {
          // Toast globally, not just when the failing thread is focused: a
          // turn-ending error (e.g. out of funds) blocks every thread, so the
          // inline error alone is too easy to miss. The stable id collapses the
          // same error from multiple blocked threads into one toast.
          notify({
            id: `gateway-error:${errorMessage}`,
            kind: 'error',
            title: 'Hermes error',
            message: errorMessage
          })
        }

        if (sessionId) {
          flushQueuedDeltas(sessionId)
          failAssistantMessage(sessionId, errorMessage)
        }

        if (isActiveEvent) {
          setTurnStartedAt(null)
        }
      }
    },
    [
      appendAssistantDelta,
      appendReasoningDelta,
      activeSessionIdRef,
      completeAssistantMessage,
      failAssistantMessage,
      flushQueuedDeltas,
      navigate,
      queryClient,
      refreshHermesConfig,
      sessionInterrupted,
      selectedStoredSessionIdRef,
      updateSessionState,
      upsertToolCall
    ]
  )

  return {
    appendAssistantDelta,
    appendReasoningDelta,
    completeAssistantMessage,
    handleGatewayEvent,
    upsertToolCall
  }
}
