import { atom, computed } from 'nanostores'

import { listAllProfileSessions } from '@/api/sessions'
import type { SessionInfo } from '@/types/hermes'

import { $sessions } from './session'

const ARCHIVED_FETCH_PAGE_SIZE = 200

export async function listEveryArchivedSession(): Promise<SessionInfo[]> {
  const sessions: SessionInfo[] = []

  while (true) {
    const page = await listAllProfileSessions(
      ARCHIVED_FETCH_PAGE_SIZE,
      0,
      'only',
      'recent',
      'all',
      {},
      sessions.length
    )

    sessions.push(...page.sessions)

    if (page.sessions.length === 0 || sessions.length >= page.total) {
      return sessions
    }
  }
}

export const $archivedSessions = atom<SessionInfo[]>([])
export const $archivedSessionsLoading = atom(false)

export async function loadArchivedSessions(): Promise<void> {
  if ($archivedSessionsLoading.get()) {
    return
  }

  $archivedSessionsLoading.set(true)

  try {
    $archivedSessions.set(await listEveryArchivedSession())
  } catch {
    $archivedSessions.set([])
  } finally {
    $archivedSessionsLoading.set(false)
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
