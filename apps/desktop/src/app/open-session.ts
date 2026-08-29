/**
 * One door for "open this session" — every surface (sidebar, ⌘K, notifications,
 * session switcher, refs, cron/artifacts) goes through here so a chat that's
 * already a tile (or the main tab) is JUMPED TO instead of yanked into main.
 *
 * Intents:
 *   - `in-place` (default non-sidebar open) — focus existing tile/main if on
 *     screen; else load into main.
 *   - `stack` (⌘K, notifications — anything that opens a chat from outside the
 *     workspace) — like `tab`, but may spend main or an open blank draft tab
 *     when either is empty.
 *   - `tab` (⌘/⌃-click / ⌘-Enter / session refs) — focus if already on screen,
 *     else open as a stacked session tab (never steals main from under you).
 *   - `window` (⇧⌘-click) — pop into its own window; falls back to `tab` when
 *     the bridge has no session-window support.
 */
import type { WorkspaceMode } from '@/contrib/types'
import { $activeSessionId, $selectedStoredSessionId, markSessionRead } from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import {
  focusedSessionNeedsRoute,
  focusOpenSession,
  openSessionTile,
  reuseBlankDraftTile,
  setSessionTileWorkspaceScope
} from '@/store/session-states'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

import { $workspaceIsPage, sessionRoute } from './routes'

export type OpenSessionIntent = 'in-place' | 'main' | 'stack' | 'tab' | 'window'

export type OpenSessionNavigate = (to: string, options?: { replace?: boolean }) => void

export interface OpenSessionWorkspaceScope {
  ownerRoute?: SessionProfileRoute
  workspaceMode: WorkspaceMode
  workspaceOwnerKey?: string
  workspaceTabTitle?: string
}

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
 *  an inline open spends main (`in-place`), a palette-style open doesn't (`stack`). */
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
  intent: OpenSessionIntent = 'in-place',
  workspaceScope: OpenSessionWorkspaceScope = { workspaceMode: 'sessions' }
): void {
  if (!storedSessionId) {
    return
  }

  // Any explicit open/focus means the user has seen the finished-turn marker.
  // Must run BEFORE the focus short-circuits below: clicking a session that is
  // already on screen (open tile, or the main session) would otherwise return
  // at focusOpenSession and never clear its unread dot.
  markSessionRead(storedSessionId)
  setSessionTileWorkspaceScope(storedSessionId, workspaceScope)
  const botWorkspaceScope = workspaceScope.workspaceMode === 'bots' ? workspaceScope : undefined
  const routedWorkspaceScope = workspaceScope.ownerRoute ? workspaceScope : botWorkspaceScope

  let resolved: OpenSessionIntent = intent

  if (resolved === 'window') {
    if (canOpenSessionWindow()) {
      void openSessionInNewWindow(storedSessionId)

      return
    }

    // No pop-out support → treat like a new tab.
    resolved = 'tab'
  }

  if (resolved === 'main') {
    // Canonical relationship chats explicitly own the main workspace. Route
    // even when the session is already open as a tile; resumeSession removes
    // that redundant tile when the main surface binds.
    navigate(sessionRoute(storedSessionId))

    return
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
    const focused = focusOpenSession(storedSessionId, workspaceScope)

    if (focused) {
      if (focusedSessionNeedsRoute(focused, $workspaceIsPage.get())) {
        navigate(sessionRoute(storedSessionId))
      }

      return
    }

    // Nothing to jump to, but an open tab may still be an empty "New session" —
    // that's the tab the user would have typed into, so spend it rather than
    // stacking a second blank one beside it.
    if (
      spendBlankDraft &&
      (botWorkspaceScope
        ? reuseBlankDraftTile(storedSessionId, botWorkspaceScope)
        : reuseBlankDraftTile(storedSessionId))
    ) {
      return
    }

    if (routedWorkspaceScope) {
      openSessionTile(storedSessionId, 'center', undefined, undefined, routedWorkspaceScope)
    } else {
      openSessionTile(storedSessionId, 'center')
    }

    focusOpenSession(storedSessionId, workspaceScope)

    return
  }

  // Already on screen? Front it. If the main session is hidden behind a full
  // page, route back to the workspace; a tile hit remains front-only for the
  // default intent used by non-sidebar callers.
  const focused = focusOpenSession(storedSessionId, workspaceScope)

  if (focusedSessionNeedsRoute(focused, $workspaceIsPage.get())) {
    navigate(sessionRoute(storedSessionId))
  }
}
