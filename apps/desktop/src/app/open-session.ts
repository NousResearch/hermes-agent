/**
 * One door for "open this session" — every surface (sidebar, ⌘K, notifications,
 * session switcher, refs, cron/artifacts) goes through here so a chat that's
 * already a tile (or the main tab) is JUMPED TO instead of yanked into main.
 *
 * Intents:
 *   - `in-place` (sidebar click / Enter) — focus existing tile/main if on
 *     screen; else load into main (same as the left sessions sidebar).
 *   - `stack` (⌘K, notifications — anything that opens a chat from outside the
 *     workspace) — like `tab`, but may spend main or an open blank draft tab
 *     when either is empty.
 *   - `tab` (⌘/⌃-click / ⌘-Enter / session refs) — focus if already on screen,
 *     else open as a stacked session tab (never steals main from under you).
 *   - `window` (⇧⌘-click) — pop into its own window; falls back to `tab` when
 *     the bridge has no session-window support.
 */
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import {
  focusedSessionNeedsRoute,
  focusOpenSession,
  openSessionTile,
  reuseBlankDraftTile
} from '@/store/session-states'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

import { $workspaceIsPage, sessionRoute } from './routes'

export type OpenSessionIntent = 'in-place' | 'stack' | 'tab' | 'window'

export type OpenSessionNavigate = (to: string, options?: { replace?: boolean }) => void

/**
 * Is the main tab holding a conversation worth preserving?
 *
 * A loaded chat may still be mid-turn, so replacing it with something else
 * throws away work the user can see. A blank draft has nothing to lose, which
 * is what lets the sidebar "+" and a `stack` open take the cheaper main path
 * instead of stacking a tab nobody asked for.
 */
export function mainChatOccupied(activeSessionId: null | string, selectedStoredSessionId: null | string): boolean {
  return Boolean(activeSessionId || selectedStoredSessionId)
}

/** Read modifiers the way session rows do — meta OR ctrl for tab, +shift for
 *  window. `base` is what an unmodified select means for the caller: the
 *  sidebar spends main (`in-place`), a palette-style open doesn't (`stack`). */
export function openSessionIntentFromModifiers(
  event?: null | { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean },
  base: OpenSessionIntent = 'in-place'
): OpenSessionIntent {
  if (!event) {
    return base
  }

  const mod = Boolean(event.metaKey || event.ctrlKey)

  if (mod && event.shiftKey) {
    return 'window'
  }

  if (mod) {
    return 'tab'
  }

  return base
}

/**
 * @param navigate Required for `in-place` (route into main when not on screen).
 *   `tab` / `window` ignore it — pass a no-op when you don't have a router handle.
 */
export function openSession(
  storedSessionId: string,
  navigate: OpenSessionNavigate,
  intent: OpenSessionIntent = 'in-place'
): void {
  if (!storedSessionId) {
    return
  }

  let resolved: OpenSessionIntent = intent

  if (resolved === 'window') {
    if (canOpenSessionWindow()) {
      void openSessionInNewWindow(storedSessionId)

      return
    }

    // No pop-out support → treat like a new tab.
    resolved = 'tab'
  }

  // A `stack` open arrives from outside the workspace, so unlike a sidebar
  // click it can't assume main is spendable: it behaves like `tab`, except main
  // IS fair game while it's only a blank draft, and an already-open blank draft
  // tab is spent before a new one is stacked.
  let spendBlankDraft = false

  if (resolved === 'stack') {
    spendBlankDraft = mainChatOccupied($activeSessionId.get(), $selectedStoredSessionId.get())
    resolved = spendBlankDraft ? 'tab' : 'in-place'
  }

  if (resolved === 'tab') {
    // Already on screen? Front it. openSessionTile would no-op on main without
    // focusing, or try to relocate an existing tile — neither is right for a
    // soft "open beside" link.
    if (focusOpenSession(storedSessionId)) {
      return
    }

    // Nothing to jump to, but an open tab may still be an empty "New session" —
    // that's the tab the user would have typed into, so spend it rather than
    // stacking a second blank one beside it.
    if (spendBlankDraft && reuseBlankDraftTile(storedSessionId)) {
      return
    }

    openSessionTile(storedSessionId, 'center')

    return
  }

  // Already on screen (open tile, or the main session)? Jump to its tab;
  // otherwise load it into main. From a full page (artifacts, skills, …) a
  // `'main'` hit still has to route back: fronting the workspace tab alone
  // leaves the page showing.
  if (focusedSessionNeedsRoute(focusOpenSession(storedSessionId), $workspaceIsPage.get())) {
    navigate(sessionRoute(storedSessionId))
  }
}
