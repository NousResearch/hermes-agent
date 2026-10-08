import { atom, computed } from 'nanostores'
import { type ReactNode, useEffect } from 'react'

import { Codecs, persistentAtom } from '@/lib/persisted'

export type ButterbarTone = 'accent' | 'danger' | 'info' | 'neutral' | 'success' | 'warn'

export interface ButterbarItem {
  id: string
  node: ReactNode
  /** Shows a close button. Implied by `persistKey`. */
  closeable?: boolean
  /** Remembers the close across launches under this key. Without it a close
   *  lasts for this run only. Bump the key to show a notice again. */
  persistKey?: string
  tone?: ButterbarTone
  /** Any CSS color. Fill, text, links and dots all derive from it with
   *  color-mix, so one value restyles the whole bar. Wins over `tone`. */
  color?: string
  /** `soft` tints the chrome (default); `solid` paints the color itself. */
  variant?: 'soft' | 'solid'
  icon?: ReactNode
  /** Higher sorts first; ties keep registration order. */
  priority?: number
  /** Autoplay dwell for this slide when the bar holds several. */
  durationMs?: number
}

const sanitizeKeys = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((key): key is string => typeof key === 'string' && key.length > 0) : []

const $registered = atom<readonly ButterbarItem[]>([])
const $closedThisRun = atom<readonly string[]>([])

export const $butterbarDismissed = persistentAtom<string[]>(
  'hermes.desktop.butterbar.dismissed',
  [],
  Codecs.json<string[]>(sanitizeKeys)
)

export const $butterbarItems = computed([$registered, $closedThisRun, $butterbarDismissed], (items, closed, dismissed) =>
  items
    .filter(item => !closed.includes(item.id) && !(item.persistKey && dismissed.includes(item.persistKey)))
    .map((item, order) => ({ item, order }))
    .sort((a, b) => (b.item.priority ?? 0) - (a.item.priority ?? 0) || a.order - b.order)
    .map(({ item }) => item)
)

/** Put a notice in the bar from anywhere. Re-registering an id replaces it in
 *  place. Returns the unregister. */
export function registerButterbar(item: ButterbarItem): () => void {
  const current = $registered.get()
  const at = current.findIndex(entry => entry.id === item.id)

  $registered.set(at < 0 ? [...current, item] : current.map((entry, i) => (i === at ? item : entry)))

  return () => {
    if ($registered.get().includes(item)) {
      $registered.set($registered.get().filter(entry => entry !== item))
    }
  }
}

export function dismissButterbar(item: ButterbarItem) {
  if (item.persistKey) {
    const dismissed = $butterbarDismissed.get()

    if (!dismissed.includes(item.persistKey)) {
      $butterbarDismissed.set([...dismissed, item.persistKey])
    }

    return
  }

  $closedThisRun.set([...$closedThisRun.get(), item.id])
}

export function isButterbarCloseable(item: ButterbarItem): boolean {
  return item.closeable ?? Boolean(item.persistKey)
}

/** Component-scoped registration: shows while mounted (and `item` is set). */
export function useButterbar(item: ButterbarItem | null) {
  useEffect(() => (item ? registerButterbar(item) : undefined), [item])
}
