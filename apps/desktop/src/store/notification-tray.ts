/**
 * The notifications tray's state: which items are listed and whether any has news the person has
 * not seen (the bell's dot).
 *
 * Items are data, registered once in a table (`app/discovery/tray-items.tsx`); the tray never names
 * one. Each item publishes a state atom: `null` keeps it out of the tray, otherwise `news` is a
 * token for what it currently announces (an update's target version, say) or null when it is
 * listed with nothing new. Opening the tray sees every listed item's news; a new token (the next
 * update) lights the bell again. Clearing an item takes it out of the tray until it announces
 * something else (the next update). What was seen or cleared is kept in this browser only.
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

const CLEARED_KEY = 'hermes.desktop.notificationTray.cleared.v1'

/** What the person cleared, as the same tokens: an item with nothing new is cleared as `''`. */
export const $trayCleared = persistentAtom<string[]>(CLEARED_KEY, [], Codecs.stringArray)

const seenToken = (id: string, news: string) => JSON.stringify([id, news])

export interface NotificationTray<T extends TrayItemSource> {
  /** The listed items in table order. */
  $listed: ReadableAtom<T[]>
  /** Listed items whose news is not seen yet, by id. */
  $unseen: ReadableAtom<ReadonlySet<string>>
  /** Everything listed now counts as seen. Tokens of news no longer announced are dropped. */
  markAllSeen: () => void
  /** Takes the item out of the tray until it announces something else. */
  clear: (id: string) => void
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

  /** What each item says now, as a token: null while it has nothing to list. */
  const announced = (states: readonly (null | TrayItemState)[], index: number) =>
    states[index] ? seenToken(items[index].id, states[index].news ?? '') : null

  const $listed = computed([$states, $trayCleared], (states, cleared) =>
    items.filter((_, index) => {
      const token = announced(states, index)

      return token !== null && !cleared.includes(token)
    })
  )

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
    },
    clear: id => {
      const states = $states.get()
      const token = announced(states, items.findIndex(item => item.id === id))

      if (token === null) {
        return
      }

      // Only what is still announced is worth remembering: the rest can never match again.
      const live = new Set(items.map((_, index) => announced(states, index)))

      $trayCleared.set([...$trayCleared.get().filter(kept => live.has(kept) && kept !== token), token])
    }
  }
}
