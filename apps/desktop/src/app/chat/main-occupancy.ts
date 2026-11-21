/**
 * Is the main tab holding a conversation worth preserving?
 *
 * A loaded chat may still be mid-turn, so replacing it with something else
 * throws away work the user can see. A blank draft has nothing to lose, which
 * is what lets the sidebar "+" and a `stack` open take the cheaper main path
 * instead of stacking a tab nobody asked for.
 *
 * Lives beside — not inside — the open door because the tab strip and the
 * draft-tab helpers read it too, and the door reads THEM when a click has to
 * choose a destination.
 */
export function mainChatOccupied(activeSessionId: null | string, selectedStoredSessionId: null | string): boolean {
  return Boolean(activeSessionId || selectedStoredSessionId)
}
