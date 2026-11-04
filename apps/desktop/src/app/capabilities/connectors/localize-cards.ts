/**
 * Catalog-content translations for connector and local MCP server cards. Slugs and server names
 * stay untouched (they drive every action); only `name`/`description` shown to people change,
 * and the original title stays searchable.
 */
import { type ContentDictionary, contentKeys, contentSearchText, lookupContent } from '@/i18n/catalog-content'

import type { ConnectorCardModel } from './types'

export function localizeConnectorCards(
  cards: ConnectorCardModel[],
  locale: string,
  overlay?: ContentDictionary | null
): ConnectorCardModel[] {
  return cards.map(card => {
    const kind = card.residency === 'local' ? 'mcp' : 'connector'
    const text = lookupContent(kind, contentKeys(card.slug, card.plugin), locale, overlay)

    if (!text) {
      return card
    }

    // Hosted connectors are brands (Gmail, GitHub…): the brand stays the title, the purpose is Russian.
    return {
      ...card,
      name: card.residency === 'hosted' ? card.name : text.name,
      description: text.description ?? card.description,
      originalName: card.name,
      search: `${contentSearchText(text)} ${(card.description ?? '').toLowerCase()}`
    }
  })
}
