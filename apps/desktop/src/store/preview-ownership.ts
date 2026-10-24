/**
 * PREVIEW TAB OWNERSHIP — who a preview tab belongs to.
 *
 * A tab is owned by the stored session that opened it (resolved through every
 * compression rotation), or — before that session's stored id exists — by the
 * runtime that opened it (a pending tab) or by the draft on screen (a true
 * draft tab). The drawer, agent tools and pop-out scoping all answer "whose is
 * this tab?" here; `preview.ts` owns the tab list and its storage.
 */

import { atom, computed } from 'nanostores'

import type { PreviewTab } from './preview'
import { normalizeProfileKey } from './profile'
import { $activeSessionId, $sessions, sessionMatchesStoredId } from './session'
import { $focusedStoredSessionId } from './session-focus'

// An ownerless tab opened while a live chat was on screen whose stored id had
// not arrived yet (the agent's open_preview right after the first send): the
// runtime it was opened under. Memory-only — runtime ids do not survive a
// relaunch. A draft opened before any send has no runtime, so its tabs carry
// no stamp and only adoptDraftPreviewTabs can take them.
export const $pendingRuntimeByTab = atom<ReadonlyMap<string, string>>(new Map())

// Auto-compression rotates a conversation's stored id (tip → next tip). The
// tabs are rekeyed onto the new tip, and the old id stays an alias so the
// window between the rotation and the focus following it cannot hide them —
// a hidden Browser's pane leaves the tree and would lose its page.
export const $rotatedSessionIds = atom<ReadonlyMap<string, string>>(new Map())

export function latestSessionId(
  sessionId: null | string | undefined,
  rotated: ReadonlyMap<string, string>
): null | string {
  let id = sessionId ?? null

  // Acyclic by construction (`rekeyPreviewTabsSession` only links one tip to
  // a different tip), so no chain is longer than the map.
  for (let hops = 0; id && rotated.has(id) && hops < rotated.size; hops++) {
    id = rotated.get(id)!
  }

  return id
}

/** `sessionId` resolved through every compression rotation seen so far. */
export function currentSessionId(sessionId: null | string | undefined): null | string {
  return latestSessionId(sessionId, $rotatedSessionIds.get())
}

/** True when `sessionId`'s drawer shows `tab`: its own tabs plus every pin.
 *  With no session (a fresh draft) the true draft tabs are its own — never a
 *  runtime's pending tab, which only that runtime's drawer shows. */
function tabVisibleTo(
  tab: PreviewTab,
  sessionId: null | string,
  rotated: ReadonlyMap<string, string>,
  pending: ReadonlyMap<string, string>
): boolean {
  if (tab.pinned) {
    return true
  }

  if (tab.sessionId == null && pending.has(tab.id)) {
    return false
  }

  return latestSessionId(tab.sessionId, rotated) === latestSessionId(sessionId, rotated)
}

/** Who an agent tool acts for: a stored id (null = none bound yet), or the
 *  full identity of the runtime asking — its stored id, the runtime itself
 *  (the tabs it opened before that id bound are its own) and the profile it
 *  belongs to, whose bucket is the only one whose pins it may use. A bare id
 *  (no profile) is a request from the chat on screen: the viewed bucket's
 *  pins. */
export type PreviewOwner =
  null | string | { profile?: null | string; runtimeId: null | string; sessionId: null | string }

export interface OwnerIdentity {
  /** Normalized profile whose pins the owner shares; null = the view's. */
  profile: null | string
  runtimeId: null | string
  sessionId: null | string
}

export function ownerIdentity(owner: PreviewOwner): OwnerIdentity {
  if (owner === null || typeof owner !== 'object') {
    return { profile: null, runtimeId: null, sessionId: owner }
  }

  return {
    profile: owner.profile ? normalizeProfileKey(owner.profile) || 'default' : null,
    runtimeId: owner.runtimeId,
    sessionId: owner.sessionId
  }
}

/** True when `tab` is `who`'s own — pins aside. A stored owner matches by id
 *  through every compression rotation. An ownerless tab is one of two things,
 *  never a shared pool: a RUNTIME's pending tab (opened before its stored id
 *  bound), which only that runtime owns; or a true DRAFT tab (opened in a
 *  fresh draft before any runtime existed), which only the draft on screen
 *  owns — the primary's runtime while it has no stored id, or a request with
 *  no runtime at all. `inView`: a draft tab outside the viewed bucket belongs
 *  to no draft on screen. */
function ownsTab(tab: PreviewTab, who: OwnerIdentity, inView: boolean): boolean {
  if (tab.sessionId != null) {
    const rotated = $rotatedSessionIds.get()

    return who.sessionId != null && latestSessionId(tab.sessionId, rotated) === latestSessionId(who.sessionId, rotated)
  }

  const draftRuntime = $activeSessionId.get()
  // A request naming no runtime and no session speaks for the draft on screen.
  const runtime = who.runtimeId ?? (who.sessionId == null ? draftRuntime : null)
  const pendingRuntime = $pendingRuntimeByTab.get().get(tab.id)

  if (pendingRuntime !== undefined) {
    return runtime === pendingRuntime
  }

  return inView && who.sessionId == null && runtime === draftRuntime
}

/** `who`'s tabs in bucket `key`: its own, plus the bucket's pins when it is
 *  `who`'s profile. `viewKey` is the bucket on screen. */
export function bucketTabsFor(
  tabs: readonly PreviewTab[],
  who: OwnerIdentity,
  key: string,
  viewKey: string
): PreviewTab[] {
  const sharesPins = key === (who.profile ?? viewKey)

  return tabs.filter(tab => (tab.pinned && sharesPins) || ownsTab(tab, who, key === viewKey))
}

/** The primary's selection names a session the list already holds — one being
 *  resumed — rather than a stored id no row carries yet, which can only be the
 *  active runtime's own id arriving. */
export const $selectionIsListed = computed([$focusedStoredSessionId, $sessions], (sessionId, sessions) =>
  Boolean(sessionId && sessions.some(session => sessionMatchesStoredId(session, sessionId)))
)

/** What the drawer is showing for: the focused session and the runtime on
 *  screen (see `$visiblePreviewTabs`). */
export interface DrawerView {
  activeRuntime: null | string
  focusedIsTile: boolean
  pending: ReadonlyMap<string, string>
  rotated: ReadonlyMap<string, string>
  selectionIsListed: boolean
  sessionId: null | string
}

/** True when the focused session's drawer shows `tab`: its own tabs and pins,
 *  plus — for the primary, never under a listed session — the tabs its runtime
 *  opened before its stored id arrived. */
export function drawerShowsTab(tab: PreviewTab, view: DrawerView): boolean {
  return (
    tabVisibleTo(tab, view.sessionId, view.rotated, view.pending) ||
    (!view.focusedIsTile &&
      !view.selectionIsListed &&
      tab.sessionId == null &&
      view.activeRuntime !== null &&
      view.pending.get(tab.id) === view.activeRuntime)
  )
}

export function setPendingRuntime(tabId: string, runtimeId: null | string): void {
  const current = $pendingRuntimeByTab.get()

  if ((current.get(tabId) ?? null) === runtimeId) {
    return
  }

  const next = new Map(current)

  if (runtimeId) {
    next.set(tabId, runtimeId)
  } else {
    next.delete(tabId)
  }

  $pendingRuntimeByTab.set(next)
}

/** `runtimeId`'s session state was dropped before its stored id bound: its
 *  tabs are plain ownerless tabs again, which the next draft's first send
 *  adopts. Omitted = every runtime (all session states were dropped). */
export function forgetPendingRuntimeTabs(runtimeId?: string): void {
  const pending = $pendingRuntimeByTab.get()

  if ([...pending.values()].some(runtime => runtimeId === undefined || runtime === runtimeId)) {
    $pendingRuntimeByTab.set(
      new Map(runtimeId === undefined ? [] : [...pending].filter(([, runtime]) => runtime !== runtimeId))
    )
  }
}
