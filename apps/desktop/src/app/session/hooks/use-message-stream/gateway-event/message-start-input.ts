import type { ClientSessionState } from '@/app/types'
import { finalizeInterruptedMessages, toChatMessages } from '@/lib/chat-messages'

import type { GatewayEventContext } from './types'

/** A replayed start must not reset a live stream, even when its input was hidden. */
export function shouldIgnoreMessageStart(
  ctx: Pick<GatewayEventContext, 'deps' | 'event' | 'sessionId' | 'explicitSid'>
): boolean {
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
    messages: finalizeInterruptedMessages(state.messages, state.streamId, ctx.occurredAt),
    streamId: null,
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
  const own = next.messages.filter(
    row =>
      row.role === 'user' &&
      (row.inputIds?.some(id => ids.includes(id)) ||
        (!row.inputIds?.length && inputs.some(item => item.ref === row.id)))
  )

  if (own.length) {
    const first = own[0]!
    const matched = new Set(own)
    // Acceptance replaces optimistic image thumbnails with the complete canonical
    // projection. Thumbnail data URLs and server paths are not comparable identities.

    const projected = {
      ...first,
      ...message,
      id: first.id,
      inputIds: ids,
      // A queued durable row is re-placed before dispatch; its old row ID is no longer authoritative.
      rowId: first.inputIds?.length ? undefined : first.rowId,
      attachmentRefs: message.attachmentRefs
    }

    // A queued optimistic input can precede the old execution's final output.
    // Move it past that stream, while leaving hidden regenerate branches in place.
    const boundary = Math.max(
      next.messages.indexOf(first),
      next.messages.findIndex(row => row.id === state.streamId && !row.hidden)
    )

    return {
      ...next,
      messages: next.messages.flatMap((row, index) => {
        const retained = matched.has(row) ? [] : [row]

        return index === boundary ? [...retained, projected] : retained
      })
    }
  }

  return {
    ...next,
    messages: [...next.messages, { ...message, id: `input-${execution}-${ids[0] ?? 'projection'}`, inputIds: ids }]
  }
}
