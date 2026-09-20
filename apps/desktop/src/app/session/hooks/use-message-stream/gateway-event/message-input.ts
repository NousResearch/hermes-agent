import { textPart } from '@/lib/chat-messages'
import { extractImageRefs } from '@/lib/embedded-images'

import { appendMidTurnUserMessage } from '../../use-prompt-actions/rewind'

import type { GatewayEventContext } from './types'

/** Optional v1 input observation. Input identity, never text equality, deduplicates a correction. */
export function handleMessageInputEvent(ctx: GatewayEventContext): boolean {
  if (ctx.event.type !== 'message.input') {
    return false
  }

  const { deps, payload, explicitSid, occurredAt, event } = ctx
  const execution = event.turn?.id
  const input = payload?.input as { role?: unknown; text?: unknown; display_kind?: unknown } | null

  const inputs = Array.isArray(payload?.inputs)
    ? payload.inputs.filter(item => item && typeof item === 'object').slice(0, 256)
    : []

  const ids = inputs.flatMap(item => (typeof item?.id === 'string' && item.id ? [item.id] : []))

  if (
    !explicitSid ||
    !execution ||
    !ids.length ||
    input?.role !== 'user' ||
    input.display_kind !== 'steer' ||
    typeof input.text !== 'string' ||
    !['steer', 'redirect'].includes(payload?.kind ?? '')
  ) {
    return true
  }

  deps.flushQueuedDeltas(explicitSid)
  deps.updateSessionState(explicitSid, state => {
    if (state.interrupted || !state.busy || (state.observedExecutionId && state.observedExecutionId !== execution)) {
      return state
    }

    const seen = state.observedInputIds ?? []

    if (ids.every(id => seen.includes(id))) {
      return state
    }

    const next = { ...state, observedExecutionId: execution, observedInputIds: [...seen, ...ids].slice(-1024) }

    // Own optimistic row already occupies the correct boundary, even if the reply was lost.
    const own = state.messages.find(
      message => !message.inputIds?.length && inputs.some(item => item.ref === message.id)
    )

    if (own) {
      return {
        ...next,
        messages: next.messages.map(message => (message.id === own.id ? { ...message, inputIds: ids } : message))
      }
    }

    const display = extractImageRefs(input.text as string)

    if (!display.cleanedText.trim() && !display.refs.length) {
      return next
    }

    return appendMidTurnUserMessage(
      next,
      {
        id: `input-${ids[0]}`,
        inputIds: ids,
        role: 'user',
        timestamp: occurredAt,
        parts: display.cleanedText ? [textPart(display.cleanedText, occurredAt)] : [],
        attachmentRefs: display.refs.length ? display.refs : undefined
      },
      { interruptTools: payload.kind === 'redirect' }
    )
  })

  return true
}
