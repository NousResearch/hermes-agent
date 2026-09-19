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

  if (
    (!ids.length && ctx.payload?.inputs_complete !== false) ||
    input?.role !== 'user' ||
    typeof input.text !== 'string' ||
    input.display_kind === 'hidden'
  ) {
    return next
  }

  const [message] = toChatMessages([
    {
      role: 'user',
      content: input.text,
      display_kind: typeof input.display_kind === 'string' ? input.display_kind : undefined,
      timestamp: ctx.occurredAt
    }
  ])

  if (!message) {
    return next
  }

  // One start projects the whole merged input. Binding only one optimistic constituent
  // would hide the other clients' words; replace all matched constituents with that projection.
  const own = state.messages.filter(
    row => row.role === 'user' && !row.inputIds?.length && inputs.some(item => item.ref === row.id)
  )

  if (own.length) {
    const first = own[0]!
    const matched = new Set(own)
    const localAttachments = own.flatMap(row => row.attachmentRefs ?? [])

    const attachmentRefs = [
      ...localAttachments,
      ...(message.attachmentRefs ?? []).filter(ref => !localAttachments.includes(ref))
    ]

    return {
      ...next,
      messages: state.messages.flatMap(row =>
        row === first
          ? [{ ...first, ...message, id: first.id, inputIds: ids, attachmentRefs }]
          : matched.has(row)
            ? []
            : [row]
      )
    }
  }

  return {
    ...next,
    messages: [
      ...finalizeInterruptedMessages(state.messages, state.streamId),
      { ...message, id: `input-${execution}-${ids[0] ?? 'projection'}`, inputIds: ids }
    ],
    streamId: null
  }
}
