import type { MemoryCard } from './types'

export function indexMemoryCards(cards: readonly MemoryCard[]): Map<string, MemoryCard> {
  const byId = new Map<string, MemoryCard>()

  cards.forEach((card, index) => {
    if (card.entry_id) {
      byId.set(`memory:${card.source}:${card.entry_id}`, card)
      return
    }

    byId.set(`memory:${card.source}:${index}`, card)
    if (card.fingerprint) {
      byId.set(`memory:${card.source}:${index}:${card.fingerprint}`, card)
    }
  })

  return byId
}
