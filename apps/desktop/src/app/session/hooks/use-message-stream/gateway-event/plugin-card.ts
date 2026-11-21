import { parsePluginCard, textPart } from '@/lib/chat-messages'
import { coerceGatewayText } from '@/lib/chat-runtime'

import type { GatewayEventContext } from './types'

/**
 * plugin.card: a plugin posted a card (`ctx.post_card`). The gateway already
 * wrote the durable row, so a reload hydrates the same thing this appends,
 * once per card.
 */
export function handlePluginCardEvent({ deps, event, payload, sessionId, occurredAt }: GatewayEventContext): boolean {
  if (event.type !== 'plugin.card') {
    return false
  }

  const text = coerceGatewayText(payload?.text).trim()
  const pluginCard = parsePluginCard(payload)

  if (!text || !pluginCard || !sessionId) {
    return true
  }

  const id = `plugin-card-${pluginCard.pluginId}-${pluginCard.cardId}`

  deps.flushQueuedDeltas(sessionId)
  deps.updateSessionState(sessionId, state =>
    state.messages.some(
      message =>
        message.id === id ||
        (message.pluginCard?.pluginId === pluginCard.pluginId && message.pluginCard.cardId === pluginCard.cardId)
    )
      ? state
      : {
          ...state,
          messages: [
            ...state.messages,
            { id, parts: [textPart(text, occurredAt)], pluginCard, role: 'system', timestamp: occurredAt }
          ]
        }
  )

  return true
}
