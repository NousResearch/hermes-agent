import type { SessionInfo } from '@/hermes'
import { sessionPinId } from '@/store/session'

export function ambiguousSessionIds(rows: readonly SessionInfo[]): Set<string> {
  const seen = new Set<string>()
  const ambiguous = new Set<string>()

  for (const row of rows) {
    if (seen.has(row.id)) {
      ambiguous.add(row.id)
    }

    seen.add(row.id)
  }

  return ambiguous
}

/** Backend pins are authoritative; a legacy bare-id hint cannot choose between owners. */
export function simplifiedPins(rows: readonly SessionInfo[], pinIds: readonly string[], pending: ReadonlySet<string>) {
  const ambiguous = ambiguousSessionIds(rows)

  const rank = (row: SessionInfo) => {
    const index = pinIds.indexOf(sessionPinId(row))

    return index < 0 ? pinIds.length : index
  }

  return rows
    .filter(row => {
      const local = pinIds.includes(sessionPinId(row)) && !ambiguous.has(row.id)

      if (pending.has(row.id) || pending.has(sessionPinId(row))) {
        return local
      }

      return row.pinned === true || (row.pinned === undefined && local)
    })
    .sort((a, b) => rank(a) - rank(b))
}
