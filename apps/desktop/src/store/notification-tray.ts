/**
 * The notifications tray's state: which items are listed and whether any has news the person has
 * not seen (the bell's dot).
 *
 * Items are data, registered once in a table (`app/discovery/tray-items.tsx`); the tray never names
 * one. Each item publishes a state atom: `null` keeps it out of the tray, otherwise `news` is a
 * token for what it currently announces (an update's target version, say) or null when it is
 * listed with nothing new. Opening the tray sees every listed item's news; a new token (the next
 * update) lights the bell again. What was seen is kept in this browser only.
 */

import { computed, type ReadableAtom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

export interface TrayItemState {
  /** Identifies what the item announces; a different token is news again. Null: nothing new. */
  news: null | string
}

export interface TrayItemSource {
  /** Stable and unique among items: it namespaces the seen tokens. */
  id: string
  $state: ReadableAtom<null | TrayItemState>
}

const SEEN_KEY = 'hermes.desktop.notificationTray.seen.v1'

export const $traySeen = persistentAtom<string[]>(SEEN_KEY, [], Codecs.stringArray)

const seenToken = (id: string, news: string) => JSON.stringify([id, news])

export interface NotificationTray<T extends TrayItemSource> {
  /** The listed items in table order. */
  $listed: ReadableAtom<T[]>
  /** Listed items whose news is not seen yet, by id. */
  $unseen: ReadableAtom<ReadonlySet<string>>
  /** Everything listed now counts as seen. Tokens of news no longer announced are dropped. */
  markAllSeen: () => void
}

export function createNotificationTray<T extends TrayItemSource>(items: readonly T[]): NotificationTray<T> {
  const ids = new Set<string>()

  for (const item of items) {
    if (ids.has(item.id)) {
      throw new Error(`notification tray: duplicate item id ${item.id}`)
    }

    ids.add(item.id)
  }

  const $states = computed(
    items.map(item => item.$state),
    (...states) => states
  )

  const $listed = computed($states, states => items.filter((_, index) => states[index] !== null))

  const currentTokens = (states: readonly (null | TrayItemState)[]) =>
    items.flatMap((item, index) => {
      const news = states[index]?.news

      return news ? [{ id: item.id, token: seenToken(item.id, news) }] : []
    })

  const $unseen = computed([$states, $traySeen], (states, seen) => {
    const known = new Set(seen)

    return new Set(currentTokens(states).flatMap(({ id, token }) => (known.has(token) ? [] : [id])))
  })

  return {
    $listed,
    $unseen,
    markAllSeen: () => {
      const tokens = currentTokens($states.get()).map(({ token }) => token)

      if (tokens.length !== $traySeen.get().length || tokens.some(token => !$traySeen.get().includes(token))) {
        $traySeen.set(tokens)
      }
    }
  }
}
