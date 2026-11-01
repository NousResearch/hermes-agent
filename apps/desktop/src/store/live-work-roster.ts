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
// never wipes another's rows. Each bucket carries the time its read was SENT,
// so an event-fed entry is authoritative only when it is newer than that read.

export interface LiveWorkCounts {
  background: number
  subagents: number
}

export interface LiveWorkBucket {
  at: number
  sessions: Record<string, LiveWorkCounts>
}

export type LiveWorkRoster = Record<string, LiveWorkBucket>

export const $liveWorkRoster = atom<LiveWorkRoster>({})

export function setLiveWorkRosterSource(source: string, entries: readonly unknown[] | null, at = Date.now()) {
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

  $liveWorkRoster.set({ ...current, [source]: { at, sessions: bucket } })
}

/** Stored ids (all lineage aliases) the roster says have `kind` running.
 *  `eventFedAt` maps each alias an event-fed entry speaks for to when that
 *  entry last heard from the backend: an entry NEWER than the roster read stays
 *  authoritative (a child that finished after the read was sent), while an older
 *  one (retained history from a pane that stopped listening) never hides work
 *  the roster saw start since. */
export function rosterLiveIds(
  roster: LiveWorkRoster,
  kind: keyof LiveWorkCounts,
  eventFedAt: ReadonlyMap<string, number>,
  sessions: Parameters<typeof lineageAliases>[1]
): string[] {
  const ids = new Set<string>()

  for (const { at, sessions: bucket } of Object.values(roster)) {
    for (const [storedId, counts] of Object.entries(bucket)) {
      if (counts[kind] <= 0) {
        continue
      }

      const aliases = lineageAliases(storedId, sessions)

      if (aliases.some(alias => (eventFedAt.get(alias) ?? -Infinity) >= at)) {
        continue
      }

      for (const alias of aliases) {
        ids.add(alias)
      }
    }
  }

  return [...ids]
}

/** Record `at` for every alias, keeping the freshest stamp per alias. */
export function noteEventFed(eventFedAt: Map<string, number>, aliases: readonly string[], at: number) {
  for (const alias of aliases) {
    eventFedAt.set(alias, Math.max(at, eventFedAt.get(alias) ?? -Infinity))
  }
}

/** True when the source's last read reported any live work. */
export const rosterSourceHasWork = (roster: LiveWorkRoster, source: string) =>
  Object.values(roster[source]?.sessions ?? {}).some(c => c.subagents > 0 || c.background > 0)
