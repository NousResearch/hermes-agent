/**
 * DERIVED SESSION STATUS SETS — pure projections of `$sessionStates` (which
 * holds `busy`/`needsInput` per runtime), keeping the data flow one-directional:
 * gateway event → cache → $sessionStates → computed views.
 *
 * Split out of `session-states.ts` (which owns the producer side: the states
 * themselves plus the status-version accounting in `noteStatusWrite`). This
 * module only READS `$sessionStates`, so the dependency runs one way and the
 * version watchdog there is always registered before these computeds exist.
 */

import { computed } from 'nanostores'

import type { ClientSessionState } from '@/app/types'
import { stableArray } from '@/lib/stable-array'
import type { SessionInfo } from '@/types/hermes'

import { $sessions, lineageAliases } from './session'
import { $sessionStates, statusSetVersions } from './session-states'

// Perf: `$sessionStates` is republished on EVERY message delta (tens/sec during
// a turn), but these sets only change on busy/needsInput edges. `stableArray`
// keeps the prior reference when membership is unchanged so `computed` skips the
// emit — otherwise the whole sidebar + every row re-renders per token. The
// status-version gate goes one step further: on status-quiet tokens the walk
// itself is skipped and the cached array is returned directly.
// Published under every id the conversation answers to, not just its current
// tip: consumers hold whichever id they were created with, and compression
// rotates the tip out from under them (see lineageAliases).
//
// A conversation that has not been persisted yet has no stored id at all, and
// dropping it here is what left the FIRST turn of a new chat with no running
// indicator anywhere — no dot, no row arc — for as long as it took the backend
// to hand one back. Its runtime id is the right fallback because until a stored
// id exists the two are the same value (submit.ts: "an unpersisted
// conversation's queue key IS its runtime id"), so the row matches; once a
// session is persisted its runtime id is nobody's key and the fallback is inert.
const storedIds = (
  states: Record<string, ClientSessionState>,
  sessions: readonly SessionInfo[],
  pred: (s: ClientSessionState) => boolean
) => {
  const ids = new Set<string>()

  for (const [runtimeId, state] of Object.entries(states)) {
    if (!pred(state)) {
      continue
    }

    for (const alias of lineageAliases(state.storedSessionId ?? runtimeId, sessions)) {
      ids.add(alias)
    }
  }

  return [...ids]
}

// Session-row lookup by any id a conversation answers to (live id, lineage
// root, or a rotated tip), rebuilt when the $sessions list ref changes — the
// list is replaced wholesale, never mutated, so the ref is a valid cache key.
// First row wins, matching `Array.find` semantics. Replaces the per-state
// `sessions.find(sessionMatchesStoredId)` scan: O(states × sessions) per token.
let rowByIdSessions: readonly SessionInfo[] | null = null
let rowById = new Map<string, SessionInfo>()

function sessionRowByAnyId(sessions: readonly SessionInfo[], storedId: string): SessionInfo | undefined {
  if (rowByIdSessions !== sessions) {
    rowByIdSessions = sessions
    const next = new Map<string, SessionInfo>()

    for (const row of sessions) {
      if (!next.has(row.id)) {
        next.set(row.id, row)
      }

      const root = row._lineage_root_id

      if (root && !next.has(root)) {
        next.set(root, row)
      }

      for (const lineageId of row._lineage_ids ?? []) {
        if (!next.has(lineageId)) {
          next.set(lineageId, row)
        }
      }
    }

    rowById = next
  }

  return rowById.get(storedId)
}

let workingIds: readonly string[] = []
let workingSessions: readonly SessionInfo[] | null = null
let workingVersion = -1
export const $workingSessionIds = computed([$sessionStates, $sessions], (states, sessions) => {
  // Status-quiet publish (transcript delta only): the membership inputs are
  // unchanged — return the cached set without walking runtimes.
  const version = statusSetVersions().busy

  if (workingSessions === sessions && workingVersion === version) {
    return workingIds
  }

  workingSessions = sessions
  workingVersion = version

  return (workingIds = stableArray(workingIds, storedIds(states, sessions, s => s.busy)))
})

let attentionIds: readonly string[] = []
let attentionSessions: readonly SessionInfo[] | null = null
let attentionVersion = -1
export const $attentionSessionIds = computed([$sessionStates, $sessions], (states, sessions) => {
  const version = statusSetVersions().busy

  if (attentionSessions === sessions && attentionVersion === version) {
    return attentionIds
  }

  attentionSessions = sessions
  attentionVersion = version

  return (attentionIds = stableArray(attentionIds, storedIds(states, sessions, s => s.needsInput)))
})

// An open session nothing has ever been sent to — the ⌘T tab whose backend
// session exists but is unlisted, or a tile still waiting on its first send.
// `blankDraftTile`'s predicate, read as a status rather than as a slot to spend.
//
// The row's own `message_count` is the tiebreaker, and it is load-bearing: a
// session RESUMING also holds an empty message list for the moment between
// binding its runtime and loading its transcript, and calling that a draft
// would flash the wrong mark on a conversation with years of history in it.
let draftIds: readonly string[] = []
let draftSessions: readonly SessionInfo[] | null = null
let draftVersion = -1
export const $draftSessionIds = computed([$sessionStates, $sessions], (states, sessions) => {
  const version = statusSetVersions().draft

  if (draftSessions === sessions && draftVersion === version) {
    return draftIds
  }

  draftSessions = sessions
  draftVersion = version

  const unsent = (state: ClientSessionState) => {
    if (state.busy || state.messages.length > 0) {
      return false
    }

    const storedId = state.storedSessionId

    // No stored id is the ⌘T tab that hasn't reached the backend yet: a draft
    // by definition, and no row to consult. Asking anyway would match a row on
    // an empty lineage root.
    if (!storedId) {
      return true
    }

    const row = sessionRowByAnyId(sessions, storedId)

    return !row || row.message_count === 0
  }

  return (draftIds = stableArray(draftIds, storedIds(states, sessions, unsent)))
})
