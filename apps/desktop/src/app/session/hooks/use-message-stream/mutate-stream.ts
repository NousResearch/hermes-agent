import { useCallback } from 'react'

import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'

import type { ClientSessionState } from '../../../types'

export type UpdateSessionState = (
  sessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState,
  storedSessionId?: string | null
) => ClientSessionState

// A seal and the next delta can land in the same millisecond.
let streamMessageSeq = 0
export const nextStreamMessageId = (prefix: string) => `${prefix}-${Date.now()}-${++streamMessageSeq}`

export function useMutateStream(updateSessionState: UpdateSessionState) {
  // Patch the in-flight assistant message (or seed it). Centralises the
  // streamId/groupId bookkeeping every event callback would otherwise repeat.
  const mutateStream = useCallback(
    (
      sessionId: string,
      transform: (parts: ChatMessagePart[], message: ChatMessage) => ChatMessagePart[],
      seed: () => ChatMessagePart[],
      opts: {
        pending?: (message: ChatMessage) => boolean
        // Resolve the message an event should mutate by payload identity
        // rather than by the current stream position. A late `tool.complete`
        // that crosses an interim/settle boundary must attach to the bubble
        // that still owns the call, wherever that bubble now sits.
        eventTarget?: (state: ClientSessionState) => string | null
      } = {},
      occurredAt = Date.now() / 1000
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

          const reconciledId = opts.eventTarget?.(state) ?? null
          const streamId = reconciledId ?? state.streamId ?? nextStreamMessageId('assistant-stream')
          // The event landed on a bubble that is NOT the live stream (sealed
          // by interim commentary, a mid-turn user message, or turn settle).
          // It is a patch to history: the bubble keeps its own pending bit
          // and the turn's stream bookkeeping is neither consulted nor changed.
          const patchesSealedBubble = reconciledId !== null && reconciledId !== state.streamId
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
                timestamp: occurredAt,
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
                    pending: patchesSealedBubble ? (m.pending ?? false) : opts.pending ? opts.pending(m) : true
                  }
                : m
            )
          }

          if (patchesSealedBubble) {
            // Later deltas must not append into the bubble the late event
            // just updated, and a late event from an earlier phase must not
            // clear the wait state of the turn now in flight.
            return { ...state, messages: nextMessages }
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

  return mutateStream
}

export type MutateStream = ReturnType<typeof useMutateStream>
