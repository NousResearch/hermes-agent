import { atom, computed } from 'nanostores'

import { listAllProfileSessions, type SessionInfo } from '@/hermes'

import { $sessions } from './session'

// Archived rows are excluded from the sessions query, so the Archived view has
// to fetch its own set. Capped: it's a lookup surface, not a feed.
const ARCHIVED_FETCH_LIMIT = 200

export const $archivedSessions = atom<SessionInfo[]>([])
export const $archivedSessionsLoading = atom(false)
let loadGeneration = 0
let loadingScope: boolean | null = null

export async function loadArchivedSessions(allConnections = false): Promise<void> {
  if ($archivedSessionsLoading.get() && loadingScope === allConnections) {
    return
  }

  const generation = ++loadGeneration
  loadingScope = allConnections
  $archivedSessionsLoading.set(true)

  try {
    const result = await listAllProfileSessions(ARCHIVED_FETCH_LIMIT, 0, 'only', 'recent', 'all', {}, allConnections)

    if (generation === loadGeneration) {
      $archivedSessions.set(result.sessions)
    }
  } catch {
    if (generation === loadGeneration) {
      $archivedSessions.set([])
    }
  } finally {
    if (generation === loadGeneration) {
      loadingScope = null
      $archivedSessionsLoading.set(false)
    }
  }
}

/** Spend on a session — provider-reported price when we have one, our own
 *  estimate otherwise. */
export const sessionCostUsd = (session: SessionInfo): number =>
  session.actual_cost_usd || session.estimated_cost_usd || 0

/** Whether ANY loaded session reports spend. Subscription auth never quotes a
 *  price, so for those users a cost sort would rank a list of zeroes — the
 *  menu hides the option instead of offering a dead one. */
export const $sessionsHaveCost = computed($sessions, sessions => sessions.some(session => sessionCostUsd(session) > 0))
