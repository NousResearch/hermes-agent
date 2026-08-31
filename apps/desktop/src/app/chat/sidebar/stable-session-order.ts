import { sessionMatchesStoredId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

const profileKey = (session: Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>): string =>
  (session.profile ?? '').trim() || 'default'

/** Stable presentation identity: compression may rotate the live tip id, but
 * the row is still the same conversation and must not jump as a "new" item. */
export const stableSessionKey = (session: Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>): string =>
  `${profileKey(session)}::${session._lineage_root_id ?? session.id}`

export interface StableSessionOrder {
  keys: string[]
  sessions: SessionInfo[]
}

/**
 * Reconcile a fresh recency-sorted backend page without moving rows the person
 * is already looking at. Background tool/session writes update `last_active`
 * frequently; treating each timestamp as a reorder instruction makes the rail
 * shuffle on a timer. Existing keys therefore retain their relative order.
 *
 * New conversations still enter at their authoritative recency position, and
 * an explicit foreground selection is the one user-owned operation allowed to
 * promote an existing row. Backend fields remain untouched; this owns only the
 * renderer's presentation order.
 */
export function stabilizeSessionOrder(
  previousKeys: readonly string[],
  recencySorted: readonly SessionInfo[],
  promotedStoredSessionId?: null | string
): StableSessionOrder {
  const rowsByKey = new Map(recencySorted.map(session => [stableSessionKey(session), session]))
  const candidateKeys = recencySorted.map(stableSessionKey)
  const keys = previousKeys.filter(key => rowsByKey.has(key))
  const known = new Set(keys)

  // Insert each genuinely new conversation relative to the next row the
  // authoritative recency page and the stable order both know. Existing rows
  // never move relative to one another.
  for (let candidateIndex = 0; candidateIndex < candidateKeys.length; candidateIndex += 1) {
    const key = candidateKeys[candidateIndex]!

    if (known.has(key)) {
      continue
    }

    let insertionIndex = keys.length

    for (let nextIndex = candidateIndex + 1; nextIndex < candidateKeys.length; nextIndex += 1) {
      const nextKnownIndex = keys.indexOf(candidateKeys[nextIndex]!)

      if (nextKnownIndex >= 0) {
        insertionIndex = nextKnownIndex

        break
      }
    }

    keys.splice(insertionIndex, 0, key)
    known.add(key)
  }

  if (promotedStoredSessionId) {
    const promoted = recencySorted.find(session => sessionMatchesStoredId(session, promotedStoredSessionId))
    const promotedKey = promoted ? stableSessionKey(promoted) : null
    const index = promotedKey ? keys.indexOf(promotedKey) : -1

    if (index > 0 && promotedKey) {
      keys.splice(index, 1)
      keys.unshift(promotedKey)
    }
  }

  return {
    keys,
    sessions: keys.flatMap(key => {
      const session = rowsByKey.get(key)

      return session ? [session] : []
    })
  }
}
