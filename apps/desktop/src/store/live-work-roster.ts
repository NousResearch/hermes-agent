import { atom } from 'nanostores'

import { lineageAliases } from './session'

// App-wide answer to "which conversations still have work running outside
// their turn?" from the backend's `subagent.roster` read. The event-fed stores
// ($subagentsBySession, $backgroundStatusBySession) only hear about sessions
// whose tile is mounted, and their `subagent.list` / `process.list` snapshots
// need the session attached to this transport — so a row that is not on
// screen, and every row right after a reload or socket reconnect, read idle
// while children ran. Keyed by STORED session id (the backend reports the
// durable conversation id), one bucket per backend source so a failing source
// never wipes another's rows.

export interface LiveWorkCounts {
  background: number
  subagents: number
}

export type LiveWorkRoster = Record<string, Record<string, LiveWorkCounts>>

export const $liveWorkRoster = atom<LiveWorkRoster>({})

export function setLiveWorkRosterSource(source: string, entries: readonly unknown[] | null) {
  const current = $liveWorkRoster.get()

  if (!entries) {
    if (!(source in current)) {
      return
    }

    const { [source]: _dropped, ...rest } = current
    $liveWorkRoster.set(rest)

    return
  }

  const bucket: Record<string, LiveWorkCounts> = {}

  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    const row = entry as Record<string, unknown>
    const sid = typeof row.session_id === 'string' ? row.session_id : ''

    if (!sid) {
      continue
    }

    bucket[sid] = { background: Number(row.background) || 0, subagents: Number(row.subagents) || 0 }
  }

  $liveWorkRoster.set({ ...current, [source]: bucket })
}

/** Stored ids (all lineage aliases) the roster says have `kind` running,
 *  minus conversations a live, event-fed entry already speaks for: those
 *  entries carry richer per-child state and stay authoritative. */
export function rosterLiveIds(
  roster: LiveWorkRoster,
  kind: keyof LiveWorkCounts,
  liveCovered: ReadonlySet<string>,
  sessions: Parameters<typeof lineageAliases>[1]
): string[] {
  const ids = new Set<string>()

  for (const bucket of Object.values(roster)) {
    for (const [storedId, counts] of Object.entries(bucket)) {
      if (counts[kind] <= 0) {
        continue
      }

      const aliases = lineageAliases(storedId, sessions)

      if (aliases.some(alias => liveCovered.has(alias))) {
        continue
      }

      for (const alias of aliases) {
        ids.add(alias)
      }
    }
  }

  return [...ids]
}
