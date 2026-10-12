import { Codecs, persistentAtom } from '@/lib/persisted'

const STORAGE_KEY = 'hermes.desktop.sidebarAlphabetical'

/**
 * Opt-in: list the sidebar's Projects overview, Pinned section and the
 * "Move to project" menu A→Z instead of by activity / hand-picked drag order.
 * Off (the default) keeps every stock ordering, drag-to-reorder included.
 * Device-local presentation only, so it lives in this window's storage.
 */
export const $sidebarAlphabetical = persistentAtom<boolean>(STORAGE_KEY, false, Codecs.bool)

export function setSidebarAlphabetical(on: boolean) {
  $sidebarAlphabetical.set(on)
}
