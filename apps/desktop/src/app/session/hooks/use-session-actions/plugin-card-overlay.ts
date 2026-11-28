import type { ChatMessage } from '@/lib/chat-messages'

/** A history page is authoritative for an immutable card already received live. */
export function withoutStoredPluginCards(stored: ChatMessage[], live: ChatMessage[]): ChatMessage[] {
  const storedCards = new Set(
    stored.flatMap(message =>
      message.pluginCard ? [JSON.stringify([message.pluginCard.pluginId, message.pluginCard.cardId])] : []
    )
  )

  return live.filter(
    message =>
      !message.pluginCard || !storedCards.has(JSON.stringify([message.pluginCard.pluginId, message.pluginCard.cardId]))
  )
}
