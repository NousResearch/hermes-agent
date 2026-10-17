import { backendScopeKey } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { $apiRequestScope } from '@/api/client'
import { listAllProfileSessions } from '@/api/sessions'
import type { SessionInfo } from '@/types/hermes'

import { $sidebarShowArchived } from './layout'
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
 * re-appended by every page whose window covers it), so rows dedupe by owner
 * and id on the way in. Termination: `page.total` counts only archived rows, so stop
 * after that many distinct archived rows have arrived. Older backends can
 * let a live pin displace an archived row from a page; cursor >= total would
 * leave that final row unread. An empty page also stops the loop. Live pins
 * can ride along even past the end, so once the cursor has covered `total`,
 * a page with no new archived row stops it too: a row archived or reordered
 * mid-load can leave the distinct count short for good.
 */
export async function listEveryArchivedSession(): Promise<SessionInfo[]> {
  const sessions: SessionInfo[] = []
  const seen = new Set<string>()
  let archivedCount = 0

  let offset = 0

  while (true) {
    const page = await listAllProfileSessions(ARCHIVED_FETCH_PAGE_SIZE, 0, 'only', 'recent', 'all', {}, offset)
    const archivedBefore = archivedCount

    for (const session of page.sessions) {
      const identity = `${session.connection_id || ''}\0${session.profile || 'default'}\0${session.id}`

      if (!seen.has(identity)) {
        seen.add(identity)
        sessions.push(session)

        if (session.archived) {
          archivedCount += 1
        }
      }
    }

    offset += ARCHIVED_FETCH_PAGE_SIZE

    if (
      page.sessions.length === 0 ||
      archivedCount >= page.total ||
      (offset >= page.total && archivedCount === archivedBefore)
    ) {
      return sessions
    }
  }
}

export const $archivedSessions = atom<SessionInfo[]>([])
export const $archivedSessionsLoading = atom(false)

// Each load takes a generation; only the newest may publish or clear the
// flag. The scope key pins it to the connection/profile it was issued on, so
// a switch mid-load starts a fresh fetch and strands the previous one.
let archiveGeneration = 0
let loadingScope: null | string = null

function archiveScopeKey(): string {
  const { connectionId, profile } = $apiRequestScope.get()

  return backendScopeKey(connectionId, profile)
}

export async function loadArchivedSessions(): Promise<void> {
  const scope = archiveScopeKey()

  if ($archivedSessionsLoading.get() && loadingScope === scope) {
    return
  }

  const generation = ++archiveGeneration
  loadingScope = scope

  $archivedSessionsLoading.set(true)

  try {
    const sessions = await listEveryArchivedSession()

    if (generation === archiveGeneration && archiveScopeKey() === scope) {
      $archivedSessions.set(sessions)
    }
  } catch {
    // A background refresh must not turn a usable Archived view into an empty
    // one when the backend is temporarily unavailable. Keep the last good set.
  } finally {
    if (generation === archiveGeneration) {
      $archivedSessionsLoading.set(false)
    }
  }
}

// Rows from another connection/profile must not linger (or be acted on), and
// nothing else reloads on a switch: the sidebar effect watches only the
// Archived toggle and background sync waits for a backend sessions.changed.
// Clear on every real scope change; reload (superseding a load stranded on
// the old scope) while Archived is showing. The keep-last-good fallback in
// loadArchivedSessions is therefore per scope.
let observedScope = archiveScopeKey()

$apiRequestScope.listen(() => {
  const scope = archiveScopeKey()

  if (scope === observedScope) {
    return
  }

  observedScope = scope
  $archivedSessions.set([])

  if ($sidebarShowArchived.get()) {
    void loadArchivedSessions()
  }
})

/** Spend on a session — provider-reported price when we have one, our own
 *  estimate otherwise. */
export const sessionCostUsd = (session: SessionInfo): number =>
  session.actual_cost_usd || session.estimated_cost_usd || 0

/** Whether ANY loaded session reports spend. Subscription auth never quotes a
 *  price, so for those users a cost sort would rank a list of zeroes — the
 *  menu hides the option instead of offering a dead one. */
export const $sessionsHaveCost = computed($sessions, sessions => sessions.some(session => sessionCostUsd(session) > 0))
