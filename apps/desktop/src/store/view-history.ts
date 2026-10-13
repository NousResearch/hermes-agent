import { atom, computed } from 'nanostores'

import type { AppView } from '@/app/routes'

/**
 * The views this window has visited — the app's own history, not the router's.
 *
 * Entries are PATHNAMES, so an overlay tab change (`/settings?tab=providers`)
 * never becomes an entry: Back returns to the previous VIEW, which is what the
 * affordance promises, instead of walking a route's query strings.
 *
 * Cursor semantics, like a browser: navigations from a point behind the cursor
 * drop what was ahead of it, and Back MOVES the cursor rather than pushing —
 * which is also why the route change Back itself causes is a no-op here (the
 * entry under the cursor is already that route).
 */
export interface ViewHistory {
  entries: readonly string[]
  index: number
}

const MAX_ENTRIES = 32

export const $viewHistory = atom<ViewHistory>({ entries: [], index: -1 })

export const $canGoBack = computed($viewHistory, history => history.index > 0)

/** Record the route the window is now on. Called once per navigation. */
export function noteViewChange(pathname: string): void {
  const { entries, index } = $viewHistory.get()

  if (entries[index] === pathname) {
    return
  }

  // A step back ONTO the entry behind the cursor — a `replace` close returning
  // to the view an overlay was opened from — moves the cursor instead of
  // pushing, so the view the user just dismissed cannot be reached again by
  // Back. Same shape as the browser: forward navigation from here drops what
  // was ahead.
  if (index > 0 && entries[index - 1] === pathname) {
    $viewHistory.set({ entries, index: index - 1 })

    return
  }

  const kept = [...entries.slice(0, index + 1), pathname]
  const overflow = Math.max(0, kept.length - MAX_ENTRIES)

  $viewHistory.set({ entries: kept.slice(overflow), index: kept.length - 1 - overflow })
}

/**
 * Step the cursor back one view and return the route to open, or null when
 * there is nothing behind this one. The cursor moves BEFORE the navigation
 * lands, so `noteViewChange` sees the route it is already on and ignores it.
 */
export function backViewRoute(): string | null {
  const { entries, index } = $viewHistory.get()

  if (index <= 0) {
    return null
  }

  $viewHistory.set({ entries, index: index - 1 })

  return entries[index - 1] ?? null
}

/** Open the previous view. False when there is nothing behind this one. */
export function goBackToPreviousView(navigate: (to: string) => void): boolean {
  const target = backViewRoute()

  if (target === null) {
    return false
  }

  navigate(target)

  return true
}

/**
 * The back affordance's visibility rule — the one place it is written. The
 * button and the titlebar's width reservation both read it, so a visible button
 * is always reserved space and a reserved slot always has a button.
 */
export function backVisible(view: AppView, canGoBack: boolean): boolean {
  return view !== 'chat' && canGoBack
}
