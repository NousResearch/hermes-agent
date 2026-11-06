import type { DesktopAgentRoster, DesktopAgentWork, DesktopConnectionKind, DesktopRegistryConnection } from '@/global'
import { sortConnectionsForDisplay } from '@/lib/connection-display'
import { fleetWorkState } from '@/lib/fleet-work'
import { sortByProfileOrder } from '@/lib/profile-order'

// Pure grouping for the fleet profile rail: which gateways sit "at rest"
// beside the active one, and which agents each of them carries. Kept free of
// React and stores so the ordering/collapse rules are unit-testable.

export interface FleetAgent {
  connectionId: string
  connectionKind: DesktopConnectionKind
  connectionLabel: string
  /** Profile name as the owning gateway knows it. */
  profile: string
  /** Pre-computed @name-device mention handle from the roster. */
  handle: string
  isDefault: boolean
  /** Freshest open worker session, when the roster carried one. The rail
   *  decides liveness from `lastActive`, never from the fetch time. */
  work?: DesktopAgentWork
}

export interface FleetGroup {
  connectionId: string
  kind: DesktopConnectionKind
  label: string
  reachable: boolean
  error?: string
  needsSignIn?: boolean
  /** The gateway's default profile — every Hermes home has one, so a group
   *  always carries it even before the roster has been enumerated. */
  defaultAgent: FleetAgent
  /** Named (non-default) profiles in the user's rail order, like the active strip. */
  named: FleetAgent[]
}

export const DEFAULT_PROFILE = 'default'

export function fleetRouteKey(connectionId: string, profile: string): string {
  return `${connectionId}::${profile}`
}

/**
 * Groups for every registered gateway EXCEPT the active one, in the same order
 * the connection switcher lists them (This device first, then by label), so the
 * rail and the readout agree. Positions never depend on which gateway is
 * active — a square must not move under the pointer when it is clicked.
 *
 * - No roster yet → each gateway still shows its default square, so the strip
 *   is complete on first paint and only gains named squares later.
 * - A gateway the roster neither lists as a source nor attributes agents to
 *   was collapsed into another registration of the same backend (install_id
 *   match) → skipped, never shown twice.
 */
export function buildRestGroups({
  activeConnectionId,
  connections,
  order = [],
  roster
}: {
  activeConnectionId: null | string
  connections: readonly DesktopRegistryConnection[]
  order?: readonly string[]
  roster: DesktopAgentRoster | null
}): FleetGroup[] {
  const groups: FleetGroup[] = []

  for (const connection of sortConnectionsForDisplay(connections)) {
    if (connection.id === activeConnectionId) {
      continue
    }

    const source = roster?.sources.find(candidate => candidate.connectionId === connection.id)
    const rows = roster?.agents.filter(agent => agent.connectionId === connection.id) ?? []

    if (roster && !source && rows.length === 0) {
      continue
    }

    const toAgent = (profile: string, handle?: string, work?: DesktopAgentWork): FleetAgent => ({
      connectionId: connection.id,
      connectionKind: connection.kind,
      connectionLabel: connection.label,
      profile,
      handle: handle ?? profile,
      isDefault: profile === DEFAULT_PROFILE,
      ...(work ? { work } : {})
    })

    const defaultRow = rows.find(row => row.profile === DEFAULT_PROFILE)

    const named = sortByProfileOrder(
      rows.filter(row => row.profile !== DEFAULT_PROFILE).map(row => toAgent(row.profile, row.handle, row.work)),
      order,
      agent => agent.profile
    )

    groups.push({
      connectionId: connection.id,
      kind: connection.kind,
      label: connection.label,
      reachable: source?.reachable ?? true,
      ...(source?.error && source.error !== 'connect-on-demand' ? { error: source.error } : {}),
      ...(source?.needsSignIn ? { needsSignIn: true } : {}),
      defaultAgent: toAgent(DEFAULT_PROFILE, defaultRow?.handle, defaultRow?.work),
      named
    })
  }

  return groups
}

/** Every square on the rest side, for the condensed-menu threshold. */
export function countRestAgents(groups: readonly FleetGroup[]): number {
  return groups.reduce((total, group) => total + 1 + group.named.length, 0)
}

export interface FleetWorkingCount {
  working: number
  /** True when every live claim rests on a roster page past the stale window. */
  stale: boolean
}

/** How many of these agents are working right now, for the group header's
 *  "N working". Idle agents are not counted; stale-only claims are counted but
 *  flagged so the header can mute them. */
export function countWorkingAgents(
  agents: readonly { work?: DesktopAgentWork }[],
  rosterFetchedAt: number,
  nowMs: number
): FleetWorkingCount {
  let working = 0
  let stale = 0

  for (const agent of agents) {
    const state = fleetWorkState(agent.work, rosterFetchedAt, nowMs)

    if (state === 'idle') {
      continue
    }

    working += 1

    if (state === 'stale') {
      stale += 1
    }
  }

  return { working, stale: working > 0 && stale === working }
}
