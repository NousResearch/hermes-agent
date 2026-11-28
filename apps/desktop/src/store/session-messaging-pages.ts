import { normalizeSessionSource } from '@/lib/session-source'
import type { SessionInfo } from '@/types/hermes'

import { normalizeProfileKey } from './profile'

/** Reconcile the background messaging seed with rows the user paged in.
 *
 * Every refresh re-reads ONE recency page across all messaging platforms,
 * while "Load more" pages a single platform deeper. A seed that hit its cap
 * is therefore not authoritative for any one platform: a row older than the
 * seed's oldest row was simply past the page, so its absence proves nothing
 * and it stays (up to that platform's previously loaded count, so displaced
 * rows never grow a section). A missing row newer than the seed's oldest row
 * really left (deleted, archived, moved elsewhere), so it goes. A seed
 * shorter than the cap is exhaustive and replaces everything.
 *
 * `scopeProfile` is the profile the seed was fetched for (`'all'` for the
 * unified view): rows of another profile left over from before a profile
 * switch are never carried into this one. */
export function reconcileMessagingRefresh(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  truncated: boolean,
  scopeProfile = 'all'
): SessionInfo[] {
  if (!truncated) {
    return incoming
  }

  const inScope = (session: SessionInfo) =>
    scopeProfile === 'all' || normalizeProfileKey(session.profile) === scopeProfile

  const oldestSeed = incoming.reduce((oldest, session) => Math.min(oldest, session.last_active ?? 0), Infinity)
  const preserveSlots = new Map<string | null, number>()
  const incomingIds = new Set(incoming.map(session => session.id))
  const incomingLineages = new Set(incoming.map(session => session._lineage_root_id ?? session.id))
  const paged: SessionInfo[] = []

  for (const session of previous) {
    if (inScope(session)) {
      const source = normalizeSessionSource(session.source)
      preserveSlots.set(source, (preserveSlots.get(source) ?? 0) + 1)
    }
  }

  for (const session of incoming) {
    const source = normalizeSessionSource(session.source)
    preserveSlots.set(source, Math.max(0, (preserveSlots.get(source) ?? 0) - 1))
  }

  for (const session of previous) {
    const pastThePage = (session.last_active ?? 0) <= oldestSeed

    if (
      !pastThePage ||
      !inScope(session) ||
      incomingIds.has(session.id) ||
      incomingLineages.has(session._lineage_root_id ?? session.id)
    ) {
      continue
    }

    const source = normalizeSessionSource(session.source)
    const slots = preserveSlots.get(source) ?? 0

    if (slots > 0) {
      paged.push(session)
      preserveSlots.set(source, slots - 1)
    }
  }

  return paged.length ? [...incoming, ...paged] : incoming
}
