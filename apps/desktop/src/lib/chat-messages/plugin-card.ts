import type { PluginCard } from './types'

const isString = (value: unknown): value is string => typeof value === 'string' && value !== ''

/** A stored or live plugin card (`display_metadata.plugin_card`, or the `plugin.card`
 *  event payload) in renderer shape; undefined for anything malformed. The gateway
 *  validates on admission, but a row is read back from storage, so read it defensively. */
export function parsePluginCard(raw: unknown): PluginCard | undefined {
  if (!raw || typeof raw !== 'object') {
    return undefined
  }

  const { card_id: cardId, directive, plugin_id: pluginId } = raw as Record<string, unknown>

  if (!isString(pluginId) || !isString(cardId)) {
    return undefined
  }

  const card: PluginCard = { cardId, pluginId }

  if (directive && typeof directive === 'object') {
    const { attrs, name } = directive as Record<string, unknown>

    if (isString(name)) {
      const strings = Object.entries(attrs && typeof attrs === 'object' ? attrs : {}).filter(
        (entry): entry is [string, string] => typeof entry[1] === 'string'
      )

      card.directive = { attrs: Object.fromEntries(strings), name }
    }
  }

  return card
}
