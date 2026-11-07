import {
  pinSession,
  setPinnedSessionOrder,
  setSidebarSessionOrderIds,
  setSidebarSessionOrderManual,
  unpinSession
} from '@/store/layout'
import { $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'
import { setSessionColorOverride } from '@/store/session-color'

/** Pins and colours are keyed by the DURABLE (lineage-root) id so they survive
 *  compression's session-id rotation; a row's live id resolves through
 *  `$sessions` (the app's own lineage matcher), and an id that resolves to
 *  nothing is passed through as-is (the stores tolerate ids for rows this
 *  window hasn't loaded). */
function durableSessionPinId(storedSessionId: string): string {
  const session = $sessions.get().find(s => sessionMatchesStoredId(s, storedSessionId))

  return session ? sessionPinId(session) : storedSessionId
}

/** The Recents order store is keyed by the LIVE id (`session.id`) — the drag
 *  path persists `reorderableRowIds` and the sidebar's reconcile effect keeps
 *  only ids present in `unpinnedAgentSessions.map(s => s.id)`. A plugin holds
 *  the durable id from the row slot, so map it back to the loaded row's live id
 *  before writing; a durable id written verbatim would be dropped by the next
 *  reconcile and, if nothing else survived, flip the manual flag off. */
function liveSessionId(storedSessionId: string): string {
  const session = $sessions.get().find(s => sessionMatchesStoredId(s, storedSessionId))

  return session ? session.id : storedSessionId
}

/** Session-list mutations a plugin may perform on the user's behalf. Every
 *  method writes the SAME stores the app's own controls write, so a plugin
 *  action and a hand click can never disagree — the sidebar and the tab
 *  strip re-render from those stores immediately. Ids are stored (durable)
 *  session ids as a sidebar row slot carries them; a live id is resolved to
 *  the same row through the app's lineage matcher. */
export const sessionsHost = {
  /** Pin or unpin a session — the row's ⇧-click / context-menu action. A
   *  pinned session moves into the Pinned section on the next render.
   *  `index` slots the pin at that position in the Pinned list (a drop
   *  target between two pins); omitted = append, like the ⇧-click. */
  pin: (storedSessionId: string, pinned = true, index?: number): void => {
    const id = durableSessionPinId(storedSessionId)

    if (pinned) {
      pinSession(id, index)
    } else {
      unpinSession(id)
    }
  },

  /** Replace the manual Recents order with `ids` (what a drag persists).
   *  Ids the window hasn't loaded reconcile on the next render, exactly
   *  like the app's own reorder. An EMPTY list clears the manual order and
   *  returns Recents to the default sort — the sidebar's own reconcile
   *  effect reaches that state one render after a drag empties the list;
   *  the verb states it directly so a plugin reset never depends on a
   *  mounted effect. */
  reorder: (ids: string[]): void => {
    setSidebarSessionOrderManual(ids.length > 0)
    setSidebarSessionOrderIds(ids.map(liveSessionId))
  },

  /** Permute the Pinned section — the sidebar's own pinned-drag path. Only
   *  pins the list names move; a pin it omits (row not loaded) keeps its slot. */
  reorderPinned: (ids: string[]): void => {
    setPinnedSessionOrder(ids.map(durableSessionPinId))
  },

  /** Set a session's colour override (or clear it with `null`) — the same
   *  per-session colour the app's own picker writes, so a plugin swatch and
   *  a hand-picked colour are one value. */
  setColor: (storedSessionId: string, color: null | string): void => {
    setSessionColorOverride(durableSessionPinId(storedSessionId), color)
  }
}
