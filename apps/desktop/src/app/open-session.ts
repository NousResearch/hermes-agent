/**
 * One door for "open this session" — every surface (sidebar, ⌘K, notifications,
 * session switcher, refs, cron/artifacts) goes through here so a chat that's
 * already a tile (or the main tab) is JUMPED TO instead of yanked into main.
 *
 * Intents:
 *   - `in-place` (click / Enter) — focus existing tile/main if on screen; else
 *     load into main (same as the left sessions sidebar).
 *   - `tab` (⌘/⌃-click / ⌘-Enter / session refs) — focus if already on screen,
 *     else open as a stacked session tab (never steals main from under you).
 *   - `window` (⇧⌘-click) — pop into its own window; falls back to `tab` when
 *     the bridge has no session-window support.
 */
import { focusedSessionNeedsRoute, focusOpenSession, openSessionTile } from '@/store/session-states'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

import { $workspaceIsPage, sessionRoute } from './routes'

export type OpenSessionIntent = 'in-place' | 'tab' | 'window'

export type OpenSessionNavigate = (to: string, options?: { replace?: boolean }) => void

/** Read modifiers the way session rows do — meta OR ctrl for tab, +shift for window. */
export function openSessionIntentFromModifiers(
  event?: null | { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }
): OpenSessionIntent {
  if (!event) {
    return 'in-place'
  }

  const mod = Boolean(event.metaKey || event.ctrlKey)

  if (mod && event.shiftKey) {
    return 'window'
  }

  if (mod) {
    return 'tab'
  }

  return 'in-place'
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

  if (resolved === 'tab') {
    // Already on screen? Front it. openSessionTile would no-op on main without
    // focusing, or try to relocate an existing tile — neither is right for a
    // soft "open beside" link.
    if (focusOpenSession(storedSessionId)) {
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
