import type { Translations } from '@/i18n/types'

import type { ConnectorCardModel } from './types'

export function connectorKindWord(card: ConnectorCardModel, copy: Translations['connectorsPage']['card']): string {
  if (card.plugin) {
    return copy.kindPlugin(card.plugin)
  }

  return card.ways.inCatalog === true ? copy.kindCatalog : copy.kindCustom
}

export function showsCatalogMark(card: ConnectorCardModel): boolean {
  return card.inCatalog && card.plugin === undefined
}
