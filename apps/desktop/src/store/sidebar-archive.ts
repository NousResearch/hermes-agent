import { atom, computed } from 'nanostores'

import type { ProfileScope } from '@/api/client'
import { listAllProfileSessions } from '@/api/sessions'
import type { SessionInfo } from '@/types/hermes'

import { $sessions } from './session'

const ARCHIVED_FETCH_PAGE_SIZE = 200

/**
 * Pages through `archived=only` until every archived row is loaded.
 *
 * The offset advances by the page size only — never by the number of rows
 * received. The list endpoints deliberately back-fill pinned conversations
 * past their LIMIT (and the pin back-fill ignores the archived filter, so a
 * non-archived pin rides along on an archived page), which means a page can
 * return more rows than `limit`. Using the row count as the next offset would
 * skip archived rows that the back-fill displaced; the page size is the only
 * cursor the server's pagination contract guarantees.
 *
 * Back-filled pins can repeat across pages (a pinned row past the window is
 * re-appended by every page whose window covers it), so rows dedupe by id on
 * the way in. Termination: `page.total` counts only archived rows (the pin
 * back-fill can't inflate it), so once the cursor has consumed `total`
 * positions every archived row has been requested. An empty page also stops
 * the loop — the server returns one only past the end (or when profiles
 * changed under us mid-pagination), never mid-list.
 */
export async function listEveryArchivedSession(scope?: ProfileScope): Promise<SessionInfo[]> {
  const sessions: SessionInfo[] = []
  const seen = new Set<string>()

  let offset = 0

  while (true) {
    const page =
      scope === undefined
        ? await listAllProfileSessions(ARCHIVED_FETCH_PAGE_SIZE, 0, 'only', 'recent', 'all', {}, offset)
        : await listAllProfileSessions(
            ARCHIVED_FETCH_PAGE_SIZE,
            0,
            'only',
            'recent',
            'all',
            {},
            offset,
            scope
          )

    for (const session of page.sessions) {
      if (!seen.has(session.id)) {
        seen.add(session.id)
        sessions.push(session)
      }
    }

    offset += ARCHIVED_FETCH_PAGE_SIZE

    if (page.sessions.length === 0 || offset >= page.total) {
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
    // A background refresh must not turn a usable Archived view into an empty
    // one when the backend is temporarily unavailable. Keep the last good set.
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
