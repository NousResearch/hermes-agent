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

/** The tab a view had FRONTED when the window last left it, keyed by the view's
 *  pathname. A route cannot carry the tab: fronting a session tile beside the
 *  workspace is not a navigation, and opening a page fronts the workspace (so
 *  it can render the page) without the route remembering the tile it displaced.
 *  Back therefore restores the route AND this tab, or it lands on the workspace
 *  — the first session tab — and abandons the one the user was reading. */
const paneByPath = new Map<string, string>()

/** Stamp the tab the view the window is on had fronted. Idempotent, and a no-op
 *  before the first route is recorded.
 *
 *  Written where the tab is FRONTED — on every layout commit
 *  (`components/pane-shell/tree/store.ts`), and again when a view is entered
 *  (`app/contrib/wiring.tsx`, for a restored tab that was fronted before the
 *  route was known) — never where a route merely LANDS: a nav affordance fronts
 *  the workspace for a page in the click's own tick
 *  (`routes.navigateToWorkspacePage`), so anything keyed on the new location
 *  reads only the workspace and loses the session tile the user was reading. The
 *  page borrow hands the tab back (`routes.revealWorkspacePane`). */
export function noteActiveViewPane(paneId: string): void {
  const { entries, index } = $viewHistory.get()
  const pathname = entries[index]

  if (!pathname) {
    return
  }

  if (paneByPath.get(pathname) !== paneId) {
    paneByPath.set(pathname, paneId)
  }
}

/** The tab Back should re-front for `pathname`, or null when none was recorded. */
export function viewPaneFor(pathname: string): null | string {
  return paneByPath.get(pathname) ?? null
}

/** Drop remembered tabs for views the capped history has dropped, so the map
 *  cannot grow past the entries it mirrors. */
function prunePaneMemory(entries: readonly string[]): void {
  if (paneByPath.size <= entries.length) {
    return
  }

  const live = new Set(entries)

  for (const pathname of paneByPath.keys()) {
    if (!live.has(pathname)) {
      paneByPath.delete(pathname)
    }
  }
}

/** Test-only: forget every remembered tab. */
export function resetViewPaneMemory(): void {
  paneByPath.clear()
}

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
  const nextEntries = kept.slice(overflow)

  $viewHistory.set({ entries: nextEntries, index: nextEntries.length - 1 })
  prunePaneMemory(nextEntries)
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

/**
 * Open the previous view, and restore the tab it had fronted. False when there
 * is nothing behind this one.
 *
 * The route change alone lands on the workspace — the first session tab — so a
 * session tile the previous view was reading would be abandoned. `restorePane`
 * re-fronts the remembered tab; pass it where the panel can reveal a tree pane.
 */
export function goBackToPreviousView(navigate: (to: string) => void, restorePane?: (paneId: string) => void): boolean {
  const target = backViewRoute()

  if (target === null) {
    return false
  }

  navigate(target)

  const pane = viewPaneFor(target)

  if (pane !== null && restorePane) {
    restorePane(pane)
  }

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
