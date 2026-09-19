import type { ClientSessionState } from '@/app/types'
import { toChatMessages } from '@/lib/chat-messages'

import { finalizeInterruptedMessages } from '../../use-prompt-actions/rewind'

import type { GatewayEventContext } from './types'

/** A replayed start must not reset a live stream, even when its input was hidden. */
export function shouldIgnoreMessageStart(ctx: GatewayEventContext): boolean {
  const execution = ctx.event.turn?.id
  const state = ctx.sessionId ? ctx.deps.sessionStateByRuntimeIdRef.current.get(ctx.sessionId) : undefined

  if (state?.interrupted) {
    return true
  }

  if (typeof execution !== 'string' || !execution) {
    return false
  }

  // An unseen start is authoritative even after a missed terminal or a running heartbeat.
  // Opaque execution ids cannot tell an unseen old execution from a new one.
  return !ctx.explicitSid || !!state?.observedStartExecutionIds?.includes(execution)
}

/** Consume only the backend's display projection; references correlate, words do not. */
export function observeMessageStartInput(state: ClientSessionState, ctx: GatewayEventContext): ClientSessionState {
  const execution = ctx.event.turn?.id

  if (!ctx.explicitSid || typeof execution !== 'string' || !execution) {
    return state
  }

  const inputs = Array.isArray(ctx.payload?.inputs)
    ? ctx.payload.inputs.slice(0, 256).filter(item => item && typeof item.id === 'string' && item.id)
    : []

  const ids = inputs.flatMap(item => (item.id ? [item.id] : []))

  const next = {
    ...state,
    observedStartExecutionIds: [...(state.observedStartExecutionIds ?? []), execution].slice(-256),
    observedInputIds: ids
  }

  const input = ctx.payload?.input as { role?: unknown; text?: unknown; display_kind?: unknown } | null

  if (!ids.length || input?.role !== 'user' || typeof input.text !== 'string' || input.display_kind === 'hidden') {
    return next
  }

  // A reference is consumed once. Reusing it for a new RPC occurrence is a new input.
  const own = state.messages.find(
    message => message.role === 'user' && !message.inputIds?.length && inputs.some(item => item.ref === message.id)
  )

  if (own) {
    return {
      ...next,
      messages: state.messages.map(message => (message === own ? { ...message, inputIds: ids } : message))
    }
  }

  const [message] = toChatMessages([
    {
      role: 'user',
      content: input.text,
      display_kind: typeof input.display_kind === 'string' ? input.display_kind : undefined,
      timestamp: ctx.occurredAt
    }
  ])

  return message
    ? {
        ...next,
        messages: [
          ...finalizeInterruptedMessages(state.messages, state.streamId),
          { ...message, id: `input-${execution}-${ids[0]}`, inputIds: ids }
        ],
        streamId: null
      }
    : next
}
