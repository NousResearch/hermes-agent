import type { DesktopAgentRoster, DesktopConnectionKind, DesktopRegistryConnection } from '@/global'
import { sortConnectionsForDisplay } from '@/lib/connection-display'
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
}

export interface FleetGroup {
  connectionId: string
  kind: DesktopConnectionKind
  label: string
  reachable: boolean
  /** The roster deliberately did not dial this gateway rather than failing to:
   *  an ssh connection before its first use, or the local runtime while the
   *  primary route is remote (enumerating it would spawn a backend the user
   *  never asked for). Clicking a square starts it — so it must not be painted
   *  as a fault. */
  onDemand: boolean
  /** The gateway's default profile — every Hermes home has one, so a group
   *  always carries it even before the roster has been enumerated. */
  defaultAgent: FleetAgent
  /** Named (non-default) profiles in the user's rail order, like the active strip. */
  named: FleetAgent[]
}

export const DEFAULT_PROFILE = 'default'

/** The three conditions a gateway can be in on the rail. `reachable` alone
 *  cannot say them apart: enumeration returns no profiles both when a dial
 *  FAILED and when it was deliberately skipped, and reporting the second as
 *  "unreachable" accuses a healthy machine (Bot Mode already draws this
 *  distinction; the rail did not). One resolver so the rail, the condensed
 *  menu and the gateway dropdown never drift. */
export type FleetGatewayCondition = 'down' | 'on-demand' | 'ready'

export function fleetGatewayCondition(group: Pick<FleetGroup, 'onDemand' | 'reachable'>): FleetGatewayCondition {
  if (group.reachable) {
    return 'ready'
  }

  return group.onDemand ? 'on-demand' : 'down'
}

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

    const toAgent = (profile: string, handle?: string): FleetAgent => ({
      connectionId: connection.id,
      connectionKind: connection.kind,
      connectionLabel: connection.label,
      profile,
      handle: handle ?? profile,
      isDefault: profile === DEFAULT_PROFILE
    })

    const defaultRow = rows.find(row => row.profile === DEFAULT_PROFILE)

    const named = sortByProfileOrder(
      rows.filter(row => row.profile !== DEFAULT_PROFILE).map(row => toAgent(row.profile, row.handle)),
      order,
      agent => agent.profile
    )

    groups.push({
      connectionId: connection.id,
      kind: connection.kind,
      label: connection.label,
      reachable: source?.reachable ?? true,
      onDemand: source?.error === 'connect-on-demand',
      defaultAgent: toAgent(DEFAULT_PROFILE, defaultRow?.handle),
      named
    })
  }

  return groups
}

/** Every square on the rest side, for the condensed-menu threshold. */
export function countRestAgents(groups: readonly FleetGroup[]): number {
  return groups.reduce((total, group) => total + 1 + group.named.length, 0)
}
