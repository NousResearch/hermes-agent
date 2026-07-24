/**
 * Mirror the sidebar's localStorage pins into the backend "keep" flag.
 *
 * Pins live in `$pinnedSessionIds` (localStorage) and drive the sidebar UI.
 * The `sessions.auto_archive` sweep, however, runs backend-side and is blind to
 * localStorage — so without this bridge it could hide a pinned chat. This
 * watcher PATCHes `pinned` on the session REST endpoint whenever the pinned set
 * changes, and re-asserts the whole current set at boot, which transparently
 * migrates pre-existing pins (no flag, no user action — the sweep just starts
 * honouring them). It never touches the sidebar's own display; localStorage
 * stays the source of truth there.
 */

import { setSessionPinnedRemote } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions, sessionMatchesStoredId } from '@/store/session'

// pin ids we've successfully PATCHed pinned=true this session.
const mirrored = new Set<string>()
// pin ids awaiting their row so we can resolve the owning profile before PATCH.
const pending = new Set<string>()

function profileFor(pinId: string): null | string | undefined {
  return $sessions.get().find(row => sessionMatchesStoredId(row, pinId))?.profile
}

function reconcile(): void {
  // Config/session REST is only reachable through the Electron bridge.
  if (!window.hermesDesktop) {
    return
  }

  const current = new Set($pinnedSessionIds.get())

  // Unpinned: anything we were tracking that's no longer in the set.
  for (const id of [...mirrored, ...pending]) {
    if (!current.has(id)) {
      mirrored.delete(id)
      pending.delete(id)
      void setSessionPinnedRemote(id, false, profileFor(id)).catch(() => {})
    }
  }

  // Newly pinned: hold until we can resolve the row (for its profile).
  for (const id of current) {
    if (!mirrored.has(id)) {
      pending.add(id)
    }
  }

  // Flush whatever we can resolve now; unresolved ids (row not loaded yet)
  // retry on the next $sessions change.
  for (const id of [...pending]) {
    const row = $sessions.get().find(entry => sessionMatchesStoredId(entry, id))

    if (!row) {
      continue
    }

    pending.delete(id)
    mirrored.add(id)
    void setSessionPinnedRemote(id, true, row.profile).catch(() => {
      // Let a later reconcile retry the mirror.
      mirrored.delete(id)
      pending.add(id)
    })
  }
}

// Sync once, then re-sync on pin-set and session-list changes. Call once per app.
export function watchSessionPins(): void {
  reconcile()
  $pinnedSessionIds.listen(reconcile)
  $sessions.listen(reconcile)
}
