import { type MutableRefObject, useCallback } from 'react'

import { type GatewayEventPayload, toolCallOwnerMessageId, upsertToolPart } from '@/lib/chat-messages'
import { dedupeGeneratedImageEchoesInParts } from '@/lib/generated-images'
import { isTodoToolName, nextTodosFromToolEvent, parseTodoRevision } from '@/lib/todos'
import { upsertSubagent } from '@/store/subagents'
import { $todosBySession, setSessionTodos } from '@/store/todos'

import type { MutateStream } from './mutate-stream'
import { delegateTaskPayloads } from './utils'

export function useToolUpsert(
  mutateStream: MutateStream,
  flushQueuedDeltas: (sessionId?: string) => void,
  sessionInterrupted: (sessionId: string) => boolean,
  nativeSubagentSessionsRef: MutableRefObject<Set<string>>
) {
  const upsertToolCall = useCallback(
    (
      sessionId: string,
      payload: GatewayEventPayload | undefined,
      phase: 'running' | 'complete',
      sourceEventType?: string,
      occurredAt = Date.now() / 1000
    ) => {
      // Text deltas flush on a timer but tool events apply now; flush first so
      // a tool part can't jump ahead of the text that preceded it.
      flushQueuedDeltas(sessionId)

      if (sessionInterrupted(sessionId)) {
        return
      }

      // The composer status stack owns todo display now (no inline panel) —
      // mirror every todo state the tool reports into its session store.
      if (payload && isTodoToolName(payload.name)) {
        const todos = nextTodosFromToolEvent($todosBySession.get()[sessionId] ?? [], payload)

        if (todos) {
          setSessionTodos(sessionId, todos, parseTodoRevision(payload))
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
        parts => dedupeGeneratedImageEchoesInParts(upsertToolPart(parts, payload, phase, occurredAt)),
        () => upsertToolPart([], payload, phase, occurredAt),
        {
          pending: m => phase !== 'complete' || (m.pending ?? false),
          // A tool event belongs to the bubble that owns the call, not to
          // whatever is streaming now. Long tools (browser scrapes run
          // minutes) outlive the boundary that seals their bubble (interim
          // commentary, a message typed mid-turn, turn settle): without this
          // lookup a completion seeds a new bubble with a duplicate row while
          // the sealed one keeps reading "Result unavailable", and a running
          // event for the same id seeds a second live row with its own timer
          // under the user's message (#113035). Both phases route by id.
          eventTarget: state => toolCallOwnerMessageId(state.messages, payload)
        },
        occurredAt
      )
    },
    [flushQueuedDeltas, mutateStream, nativeSubagentSessionsRef, sessionInterrupted]
  )

  return upsertToolCall
}
