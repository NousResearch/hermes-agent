import { sessionTitle } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

/**
 * index sessions by every id a pin might be stored under.
 *
 * the sidebar fetches three independent slices - recents, cron, and messaging
 * - and renders the latter two in self-managed sections. any of them can be
 * pinned, so all three must be indexed here or the pinned section cannot
 * resolve the pin to a row. a pinned session is also filtered out of its own
 * section, so failing to index it does not merely misplace the row: it removes
 * the session from the sidebar entirely.
 *
 * each session is keyed under both its live id and its lineage root, so a pin
 * stored before an auto-compression still resolves to the live continuation
 * tip. recents are indexed last and win a direct id collision.
 */
export function buildSessionByAnyId(
  visibleSessions: SessionInfo[],
  cronSessions: SessionInfo[],
  messagingSessions: SessionInfo[]
): Map<string, SessionInfo> {
  const map = new Map<string, SessionInfo>()

  for (const session of [...cronSessions, ...messagingSessions, ...visibleSessions]) {
    map.set(session.id, session)

    if (session._lineage_root_id && !map.has(session._lineage_root_id)) {
      map.set(session._lineage_root_id, session)
    }
  }

  return map
}

/**
 * compare session rows alphabetically by displayed title.
 *
 * uses case-insensitive, accent-insensitive natural ordering (portuguese/multilingual-aware,
 * numeric: true so "session 2" precedes "session 10"). ties fall back to case-sensitive natural
 * ordering, and finally the session id for a deterministic total ordering.
 */
export function compareSessionTitles(a: SessionInfo, b: SessionInfo): number {
  const titleA = sessionTitle(a)
  const titleB = sessionTitle(b)

  const primary = titleA.localeCompare(titleB, undefined, { numeric: true, sensitivity: 'base' })
  if (primary !== 0) {
    return primary
  }

  const secondary = titleA.localeCompare(titleB, undefined, { numeric: true })
  if (secondary !== 0) {
    return secondary
  }

  return a.id.localeCompare(b.id)
}

/**
 * resolve the pinned section's rows: collects locally stored pin ids and any row the server flags
 * pinned that the local set does not know about yet, filtered by in-flight pin writes, and sorts
 * them automatically by displayed conversation title.
 */
export function resolvePinnedSessions(
  pinnedSessionIds: readonly string[],
  sessionByAnyId: Map<string, SessionInfo>,
  allSessions: readonly SessionInfo[],
  unconfirmedPinWrites: ReadonlySet<string>
): SessionInfo[] {
  const seen = new Set<string>()
  const out: SessionInfo[] = []

  for (const pinId of pinnedSessionIds) {
    const session = sessionByAnyId.get(pinId)

    if (session && !seen.has(session.id)) {
      seen.add(session.id)
      out.push(session)
    }
  }

  for (const session of allSessions) {
    if (session.pinned !== true || seen.has(session.id)) {
      continue
    }

    // a pin write of ours the row predates - under either identity, since the
    // fence is keyed on the durable id and the row may surface as its tip.
    if (
      unconfirmedPinWrites.has(session.id) ||
      (session._lineage_root_id != null && unconfirmedPinWrites.has(session._lineage_root_id))
    ) {
      continue
    }

    seen.add(session.id)
    out.push(session)
  }

  return out.sort(compareSessionTitles)
}
