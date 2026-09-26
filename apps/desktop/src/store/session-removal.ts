import { atom } from 'nanostores'

// Client-side cache eviction (Apollo-style optimistic layer): ids the user just
// deleted/archived. The backend tree is a snapshot that still lists them until
// its next refresh, so the render-time overlay strips these so the tree matches
// the live `$sessions` cache exactly — same as the flat Recents list. Pruned on
// refresh once the server snapshot has caught up.
//
// This lives beside the session stores rather than in `projects.ts` so the
// resume path can consult it without importing the project tree (which itself
// reads `store/session`).
export const $removedSessionIds = atom<Set<string>>(new Set())

/**
 * Per-id tombstone lifecycle generations (#85163). Membership alone cannot
 * distinguish "unchanged" from an add → remove ABA cycle while a by-id
 * resolve is in flight: an archive that begins AND rolls back (failed RPC)
 * mid-request leaves membership looking untouched, yet the response raced a
 * doomed row. Immutable snapshots let async publishers reject any response
 * whose target changed, without blocking unrelated ids or a later explicit
 * resume that starts after the lifecycle has settled.
 */
export type SessionTombstoneGenerationSnapshot = ReadonlyMap<string, number>
let tombstoneGenerations: SessionTombstoneGenerationSnapshot = new Map()

function setRemovedSessionIds(next: Set<string>): void {
  const current = $removedSessionIds.get()
  const changed = new Set<string>()

  for (const id of current) {
    if (!next.has(id)) {
      changed.add(id)
    }
  }

  for (const id of next) {
    if (!current.has(id)) {
      changed.add(id)
    }
  }

  if (!changed.size) {
    return
  }

  const generations = new Map(tombstoneGenerations)

  for (const id of changed) {
    generations.set(id, (generations.get(id) ?? 0) + 1)
  }

  // Publish the generation FIRST: a subscriber reacting to membership must
  // already observe the lifecycle change when it starts a by-id lookup.
  tombstoneGenerations = generations
  $removedSessionIds.set(next)
}

/** Generation snapshot to compare against later (see `tombstoneLifecycleChanged`). */
export function captureSessionTombstoneGenerations(): SessionTombstoneGenerationSnapshot {
  return tombstoneGenerations
}

/** True when any id's tombstone lifecycle moved since `snapshot` (ABA-safe). */
export function tombstoneLifecycleChanged(
  snapshot: SessionTombstoneGenerationSnapshot,
  ids: Array<null | string | undefined>
): boolean {
  return ids.some(id => {
    const target = id?.trim()

    if (!target) {
      return false
    }

    return snapshot.get(target) !== tombstoneGenerations.get(target)
  })
}

export function tombstoneSessions(ids: Array<null | string | undefined>): void {
  const next = new Set($removedSessionIds.get())

  for (const id of ids) {
    const trimmed = id?.trim()

    if (trimmed) {
      next.add(trimmed)
    }
  }

  setRemovedSessionIds(next)
}

export function untombstoneSessions(ids: Array<null | string | undefined>): void {
  const current = $removedSessionIds.get()

  if (!current.size) {
    return
  }

  const next = new Set(current)

  for (const id of ids) {
    const trimmed = id?.trim()

    if (trimmed) {
      next.delete(trimmed)
    }
  }

  setRemovedSessionIds(next)
}

// Ids whose delete/archive RPC is still in flight. Their tombstones are pinned
// against the projects.tree prune: a refresh whose snapshot predates the
// mutation completing must NOT drop the tombstone, or the row flashes back until
// the backend catches up. Keyed by id, so concurrent deletes stay independent.
export const $sessionMutationsInFlight = atom<Set<string>>(new Set())

function mutateInFlight(ids: Array<null | string | undefined>, add: boolean): void {
  const current = $sessionMutationsInFlight.get()
  const next = new Set(current)

  for (const id of ids) {
    const trimmed = id?.trim()

    if (trimmed) {
      add ? next.add(trimmed) : next.delete(trimmed)
    }
  }

  if (next.size !== current.size) {
    $sessionMutationsInFlight.set(next)
  }
}

export const beginSessionMutation = (ids: Array<null | string | undefined>): void => mutateInFlight(ids, true)
export const endSessionMutation = (ids: Array<null | string | undefined>): void => mutateInFlight(ids, false)

/** The session is on its way out: already tombstoned, or its delete/archive RPC
 *  is still in flight. Either way the durable row is doomed, so nothing may
 *  resume it — a resume would 404 and toast "Resume failed / Session not found"
 *  for a chat the user deliberately removed.
 *
 *  Deletion tombstones synchronously and only untombstones if the RPC fails
 *  (which restores the row and the route), so this predicate is the single
 *  answer every resume actuator asks. */
export function isSessionRemovalPending(sessionId: null | string | undefined): boolean {
  const id = sessionId?.trim()

  if (!id) {
    return false
  }

  return $removedSessionIds.get().has(id) || $sessionMutationsInFlight.get().has(id)
}
