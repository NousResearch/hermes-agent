/**
 * PROFILE DOT STATE — the profile rail's per-profile rollup of the shared
 * session dot state ($sessionDotStateById), so a profile that finished (or
 * blocked, or is still working) while another profile was selected shows an
 * indicator on its rail square and in the condensed profile dropdown (#91710).
 *
 * The session-level layers already exist and stay authoritative: the live
 * busy→idle edge, the persisted per-profile unread markers
 * ($unreadFinishedMarkers, which survive restarts), and the backend's derived
 * row.unread watermark — all folded by session-dot-state.ts into ONE state per
 * session. This module only re-buckets those resolved states by the profile
 * that owns each session, keyed by the same composite (connection, profile)
 * scope the gateway registry uses (backendScopeKey), so identically named
 * profiles on different gateways never cross-mark.
 *
 * Three passes, weakest claim first (a later pass overwrites on priority, the
 * same rule session-dot-state applies per session):
 *
 * 1. LOADED ROWS — every listed chat/messaging row claims its own scope from
 *    its (connection_id, profile) tags. Cron and kanban rows are deliberately
 *    EXCLUDED: those runs finish unwatched by design and counting them would
 *    turn every profile's square into a permanently lit run counter (#93552).
 * 2. UNLISTED RUNTIMES — a live (busy / needs-input) session whose row is not
 *    loaded (an inactive profile's background work, a hidden Bot Chat) claims
 *    the scope its socket proved (runtimeSessionOwner: the exact registry
 *    (connection, profile), or the legacy profile-only pool's bare name).
 *    No proven owner → no claim: no dot rather than a wrong one.
 * 3. PERSISTED UNREAD MARKERS — the cross-profile, cross-restart half. A
 *    marker with no loaded row is an unread session in a profile whose rows
 *    are not on screen. Markers are bucketed by BARE profile name (see
 *    session-unread.ts), so they are attributed to a (connection, profile)
 *    scope only when the gateway that owns that profile name is unambiguous —
 *    loaded rows' connections, the active gateway, and the fleet roster must
 *    all agree. Two gateways exposing the same profile name leave that bucket
 *    unpainted rather than guessing (#91710: identically named profiles stay
 *    independently scoped).
 */

import { backendScopeKey, LOCAL_CONNECTION_ID } from '@hermes/shared'
import { computed } from 'nanostores'

import { stableRecord } from '@/lib/stable-array'
import type { SessionInfo } from '@/types/hermes'

import { $activeConnectionId } from './connections'
import { $fleetRoster } from './fleet-roster'
import { $profiles, normalizeProfileKey } from './profile'
import {
  $cronSessions,
  $kanbanSessions,
  $messagingSessions,
  $sessions,
  sessionMatchesStoredId,
  sessionPinId
} from './session'
import { $sessionDotStateById } from './session-dot-state'
import { $sessionStates, runtimeSessionOwner } from './session-states'
import { $unreadFinishedMarkers } from './session-unread'

/** The summary one profile square paints: its loudest state plus the counts
 *  behind it. Priority matches the session-level rank: a profile that needs
 *  your answer outranks one that is working, which outranks unread. */
export type ProfileDotState = 'needs-input' | 'unread' | 'working'

export interface ProfileDotSummary {
  readonly state: ProfileDotState
  readonly needsInputCount: number
  readonly workingCount: number
  readonly unreadCount: number
}

/** The canonical scope key for a rail square or a session row: the same
 *  composite the registry uses, so a row and its square can never disagree
 *  about which (gateway, profile) a session belongs to. */
export const profileDotScopeKey = (
  connectionId: null | string | undefined,
  profile: null | string | undefined
): string => backendScopeKey(connectionId?.trim() || null, profile)

const profileKeyOf = (row: Pick<SessionInfo, 'profile'>): string => normalizeProfileKey(row.profile)

const rowScopeOf = (row: Pick<SessionInfo, 'connection_id' | 'profile'>): string =>
  profileDotScopeKey(row.connection_id, profileKeyOf(row))

// Interned so `useStoreSelector` consumers bail out when their square's
// counts did not change, and so stableRecord's per-value reference compare
// gates the computed's emit.
const summaryCache = new Map<string, ProfileDotSummary>()

function internSummary(needsInputCount: number, workingCount: number, unreadCount: number): ProfileDotSummary {
  const key = `${needsInputCount}\u0000${workingCount}\u0000${unreadCount}`
  let summary = summaryCache.get(key)

  if (!summary) {
    summary = Object.freeze({
      state: needsInputCount > 0 ? ('needs-input' as const) : workingCount > 0 ? ('working' as const) : ('unread' as const),
      needsInputCount,
      workingCount,
      unreadCount
    })
    summaryCache.set(key, summary)
  }

  return summary
}

interface Tally {
  needsInput: number
  unread: number
  working: number
}

let prev: Readonly<Record<string, ProfileDotSummary>> = {}

/** Profile-scope key → the summary that scope's rail square paints. Scopes
 *  with no claims are absent. */
export const $profileDotStateByScope = computed(
  [
    $sessionDotStateById,
    $sessionStates,
    $sessions,
    $messagingSessions,
    $cronSessions,
    $kanbanSessions,
    $unreadFinishedMarkers,
    $profiles,
    $fleetRoster,
    $activeConnectionId
  ],
  (dots, states, sessions, messaging, cron, kanban, markers, profiles, roster, activeConnectionId) => {
    const tallies = new Map<string, Tally>()

    const tallyFor = (scope: string): Tally => {
      let tally = tallies.get(scope)

      if (!tally) {
        tally = { needsInput: 0, unread: 0, working: 0 }
        tallies.set(scope, tally)
      }

      return tally
    }

    const claimRow = (scope: string, state: 'needs-input' | 'unread' | 'working') => {
      const tally = tallyFor(scope)

      if (state === 'needs-input') {
        tally.needsInput++
      } else if (state === 'working') {
        tally.working++
      } else {
        tally.unread++
      }
    }

    // Pass 1: loaded rows. The dot state is already resolved per session;
    // stalled/background fold into working exactly like the sidebar's status
    // buckets do.
    const chatRows: readonly SessionInfo[] = [...sessions, ...messaging]

    for (const row of chatRows) {
      if (row.archived) {
        continue
      }

      const dot = dots[row.id]

      if (dot === 'needs-input' || dot === 'working' || dot === 'unread') {
        claimRow(rowScopeOf(row), dot)
      } else if (dot === 'stalled' || dot === 'background') {
        claimRow(rowScopeOf(row), 'working')
      }
    }

    // Pass 2: unlisted live runtimes. A listed conversation is already claimed
    // by its row (under every alias the dot state publishes), so this pass only
    // decides scope for work the row lists cannot see.
    const listedIds = new Set<string>()

    for (const row of [...chatRows, ...cron, ...kanban]) {
      listedIds.add(row.id)

      if (row._lineage_root_id) {
        listedIds.add(row._lineage_root_id)
      }

      if (row._lineage_ids) {
        for (const id of row._lineage_ids) {
          listedIds.add(id)
        }
      }
    }

    for (const [runtimeId, state] of Object.entries(states)) {
      if (!state || (!state.busy && !state.needsInput)) {
        continue
      }

      const storedId = state.storedSessionId ?? runtimeId

      if (listedIds.has(storedId)) {
        continue
      }

      const owner = runtimeSessionOwner(runtimeId)

      if (!owner) {
        continue
      }

      const scope =
        typeof owner === 'string'
          ? profileDotScopeKey(null, owner)
          : profileDotScopeKey(owner.connectionId, owner.profile)

      claimRow(scope, state.needsInput ? 'needs-input' : 'working')
    }

    // Pass 3: persisted unread markers with no loaded row — the sessions of a
    // profile whose rows are not on screen (inactive profile, cold start).
    // The marker bucket is a BARE profile name; attribute it to a gateway scope
    // only when every source that names this profile agrees on which gateway
    // that is.
    const allRows: readonly SessionInfo[] = [...chatRows, ...cron, ...kanban]

    for (const [profile, ids] of Object.entries(markers)) {
      const rowsInProfile = allRows.filter(row => profileKeyOf(row) === profile)

      const unmatched = ids.filter(
        id => !rowsInProfile.some(row => sessionPinId(row) === id || sessionMatchesStoredId(row, id))
      )

      if (!unmatched.length) {
        continue
      }

      const connections = new Set<null | string>()

      for (const row of rowsInProfile) {
        connections.add(row.connection_id?.trim() || null)
      }

      if (profiles.some(entry => normalizeProfileKey(entry.name) === profile)) {
        connections.add(
          activeConnectionId && activeConnectionId !== LOCAL_CONNECTION_ID ? activeConnectionId : null
        )
      }

      for (const agent of roster?.agents ?? []) {
        if (normalizeProfileKey(agent.profile) === profile) {
          connections.add(agent.connectionId)
        }
      }

      if (connections.size !== 1) {
        continue
      }

      const [connectionId] = connections

      tallyFor(profileDotScopeKey(connectionId, profile)).unread += unmatched.length
    }

    const next: Record<string, ProfileDotSummary> = {}

    for (const [scope, tally] of tallies) {
      if (tally.needsInput || tally.working || tally.unread) {
        next[scope] = internSummary(tally.needsInput, tally.working, tally.unread)
      }
    }

    return (prev = stableRecord(prev, next))
  }
)

/** The summary for one rail square's (connection, profile). Absent scope →
 *  undefined → the square paints no indicator. */
export const profileDotSummaryFor = (
  byScope: Readonly<Record<string, ProfileDotSummary>>,
  connectionId: null | string | undefined,
  profile: null | string | undefined
): ProfileDotSummary | undefined => byScope[profileDotScopeKey(connectionId, profile)]
