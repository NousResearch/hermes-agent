import type { SessionInfo } from '@/hermes'
import { sessionPinId } from '@/store/session'

import { sessionRowKey } from './simplified-session-identity'

export function ambiguousSessionIds(rows: readonly SessionInfo[]): Set<string> {
  const seen = new Map<string, string>()
  const ambiguous = new Set<string>()

  for (const row of rows) {
    const ownerKey = sessionRowKey(row)

    if (seen.has(row.id) && seen.get(row.id) !== ownerKey) {
      ambiguous.add(row.id)
    }

    seen.set(row.id, ownerKey)
  }

  return ambiguous
}

/** Backend pins are authoritative; a legacy bare-id hint cannot choose between owners. */
export function simplifiedPins(
  rows: readonly SessionInfo[],
  pinIds: readonly string[],
  pending: ReadonlySet<string>,
  pinOverride?: (row: SessionInfo) => boolean | undefined
) {
  const ambiguous = ambiguousSessionIds(rows)

  const rank = (row: SessionInfo) => {
    const index = pinIds.indexOf(sessionPinId(row))

    return index < 0 ? pinIds.length : index
  }

  return rows
    .filter(row => {
      const override = pinOverride?.(row)

      if (override !== undefined) {
        return override
      }

      const local = pinIds.includes(sessionPinId(row)) && !ambiguous.has(row.id)

      if (pending.has(row.id) || pending.has(sessionPinId(row))) {
        return local
      }

      return row.pinned === true || (row.pinned === undefined && local)
    })
    .sort((a, b) => rank(a) - rank(b))
}
