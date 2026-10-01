/**
 * DETACHED SESSIONS — live chats that belong to a surface instead of a pane.
 *
 * The app has had exactly two ways to show a conversation: the workspace pane
 * (the primary session) and a session tile. A tile is welded to the layout
 * tree — it IS a pane, with a tab in the strip, a dock direction and a place in
 * the close/promote order — which makes it the wrong shape for a chat that
 * lives INSIDE another surface, like the composer on the Workflows canvas.
 * Registering one as a tile would put a tab in the user's workspace for a
 * conversation that has no business being there.
 *
 * A detached session is the third shape: a real session, streamed and rendered
 * by the real chat stack, owned by whoever mounted it. No pane, no tab, not
 * persisted here — the owner persists its own id and re-binds on mount.
 *
 * The registry exists for ONE reason: `$sessionStates` is a weighted LRU whose
 * unreferenced settled transcripts get evicted under pressure, and "referenced"
 * meant active, selected, or a tile. A detached session is none of those, so
 * without this its transcript would be collected out from under a surface still
 * showing it. Being in here is what makes it referenced.
 */

import { atom, computed } from 'nanostores'

/** Stored session id → the runtime id it is currently bound to. */
export const $detachedSessions = atom<Record<string, string>>({})

const $detachedRuntimeIds = computed($detachedSessions, byStored => new Set(Object.values(byStored)))

/** Claim a stored session as detached, bound to `runtimeId`. Re-binding after
 *  a resume is the same call — the stored id is the durable identity. */
export function bindDetachedSession(storedSessionId: string, runtimeId: string): void {
  const current = $detachedSessions.get()

  if (current[storedSessionId] === runtimeId) {
    return
  }

  $detachedSessions.set({ ...current, [storedSessionId]: runtimeId })
}

/** Give up the claim. The transcript becomes evictable like any other. */
export function releaseDetachedSession(storedSessionId: string): void {
  const current = $detachedSessions.get()

  if (!(storedSessionId in current)) {
    return
  }

  const next = { ...current }
  delete next[storedSessionId]
  $detachedSessions.set(next)
}

/** Is this runtime (or its stored session) held open by a detached surface? */
export function isDetachedSession(runtimeId: string, storedSessionId?: null | string): boolean {
  return $detachedRuntimeIds.get().has(runtimeId) || (!!storedSessionId && storedSessionId in $detachedSessions.get())
}
