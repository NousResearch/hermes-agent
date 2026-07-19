import { normalizeSessionSource } from '@/lib/session-source'
import type { SessionInfo } from '@/types/hermes'

/** Reconcile the background messaging seed with rows the user paged in.
 * A seed that hit its aggregate cap is not authoritative for any one platform,
 * so it may replace rows but cannot reduce that platform's loaded count. A
 * shorter seed is exhaustive and may drop deleted or archived rows. */
export function reconcileMessagingRefresh(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  truncated: boolean
): SessionInfo[] {
  if (!truncated) {
    return incoming
  }

  const preserveSlots = new Map<string | null, number>()
  const incomingIds = new Set(incoming.map(session => session.id))
  const incomingLineages = new Set(incoming.map(session => session._lineage_root_id ?? session.id))
  const paged: SessionInfo[] = []

  for (const session of previous) {
    const source = normalizeSessionSource(session.source)
    preserveSlots.set(source, (preserveSlots.get(source) ?? 0) + 1)
  }

  for (const session of incoming) {
    const source = normalizeSessionSource(session.source)
    preserveSlots.set(source, Math.max(0, (preserveSlots.get(source) ?? 0) - 1))
  }

  for (const session of previous) {
    if (incomingIds.has(session.id) || incomingLineages.has(session._lineage_root_id ?? session.id)) {
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
