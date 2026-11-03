import { gatewayActivationEpoch, host, useGatewayActivationEpoch, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { canonicalGroupRequest } from './canonical-groups'
import { resolveBotConnectionRoute } from './routing'
import type { RosterRow } from './types'

export type GroupExecutionMode = 'canonical' | 'legacy' | 'unavailable'

export function groupExecutionMode(value: unknown): GroupExecutionMode {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return 'unavailable'
  }

  const capability = value as Record<string, unknown>
  const { driver, persistent_process, features, methods } = capability

  if (features !== undefined && (!Array.isArray(features) || features.some(feature => typeof feature !== 'string'))) {
    return 'unavailable'
  }

  if (methods !== undefined && (!Array.isArray(methods) || methods.some(method => typeof method !== 'string'))) {
    return 'unavailable'
  }

  if (driver === true) {return 'canonical'}

  // App-managed hosted owners can be nonpersistent, even when their driver stops.
  const ownsRooms = Object.hasOwn(capability, 'protocol_version') || Object.hasOwn(capability, 'authority_gateway_id')
    || methods?.includes('groups.create') || features?.includes('canonical_session_owner')

  if (driver === false && persistent_process === false && !ownsRooms) {
    return 'legacy'
  }

  return 'unavailable'
}

/** The profiles `groups.create` will seat for THIS caller, as `groups.capabilities`
 *  reports them (the server's member rules: parked, retiring and unaddressable bots
 *  are out, and only a host-scoped socket seats served siblings). Null when the
 *  gateway predates the field, so callers keep their connection-only filter. */
export function groupSeatableProfiles(value: unknown): null | Set<string> {
  const seatable = value && typeof value === 'object' ? (value as Record<string, unknown>).seatable_profiles : undefined

  if (!Array.isArray(seatable)) {
    return null
  }

  return new Set(seatable.filter((name): name is string => typeof name === 'string'))
}

/** Fence a group-creation flow to the source that approved it. Pass the epoch the
 *  capability was read under (default: now); the predicate turns false once the
 *  connection, profile, socket or activation epoch moves, so a late result is
 *  neither acted on nor published. */
export function groupCreationSource(route: { connectionId: string; profile: string },
  activationEpoch = gatewayActivationEpoch()) {
  return () => gatewayActivationEpoch() === activationEpoch &&
    route.connectionId === host.state.connectionId.get() &&
    route.profile === host.state.profile.get() && host.state.gateway.get() === 'open'
}

export interface GroupCapabilityState {
  activationEpoch: number
  mode: GroupExecutionMode
  seatable: null | Set<string>
  source: string
}

/** `groups.capabilities` for the active source while `active`, keyed by that
 *  source so an answer from a source the window has left is never used. */
export function useGroupCapability(active: boolean) {
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const activationEpoch = useGatewayActivationEpoch()
  const source = JSON.stringify([connectionId, profile, gateway, activationEpoch])
  const [capability, setCapability] = useState<GroupCapabilityState | null>(null)

  useEffect(() => {
    let cancelled = false
    setCapability(null)

    if (!active || gateway !== 'open' || !connectionId) {
      return
    }

    const route = { connectionId, profile }
    const sourceCurrent = groupCreationSource(route, activationEpoch)

    const settle = (result: unknown, failed: boolean) => {
      if (!cancelled && sourceCurrent()) {
        setCapability({
          activationEpoch,
          mode: failed ? 'unavailable' : groupExecutionMode(result),
          seatable: failed ? null : groupSeatableProfiles(result),
          source
        })
      }
    }

    void canonicalGroupRequest<unknown>(route, 'groups.capabilities').then(
      result => settle(result, false),
      () => settle(null, true)
    )

    return () => { cancelled = true }
  }, [activationEpoch, active, connectionId, gateway, profile, source])

  const current = capability?.source === source ? capability : null

  return {
    capability,
    connectionId,
    mode: current ? current.mode : ('checking' as const),
    observedRoute: connectionId ? { connectionId, profile } : null,
    seatable: current ? current.seatable : null,
    source
  }
}

/** True when `bot` lives on `connectionId`: its row and its route (when it
 *  carries one) both name that connection. A row whose route disagrees with its
 *  connection is on neither side. */
export function botOnConnection(bot: RosterRow, connectionId: null | string): boolean {
  const connection = String(connectionId || '').trim()

  if (!connection) {
    return false
  }

  const routeAgrees = !bot.route?.connectionId || bot.route.connectionId === connection

  return bot.connectionId ? bot.connectionId === connection && routeAgrees : !bot.remoteSource && routeAgrees
}

/** A bot on another connection the Desktop can carry a room turn to: a live row
 *  whose route resolves to the same connection the row names. */
function courierReachable(bot: RosterRow): boolean {
  if (bot.sourceMissing || bot.sourceReachable === false) {
    return false
  }

  const { route } = resolveBotConnectionRoute(bot)

  return Boolean(route && (!bot.connectionId || route.connectionId === bot.connectionId))
}

function memberConnection(bot: RosterRow, connectionId: null | string): string {
  return String(bot.connectionId || bot.route?.connectionId || connectionId || '').trim()
}

/** True when `members` live on two or more connections. No single gateway can
 *  own such a room, so it runs on this Desktop, which carries each member's turn
 *  to its own gateway. A member with no recorded connection counts as being on
 *  `connectionId`. */
export function groupNeedsDesktopCourier(members: RosterRow[], connectionId: null | string): boolean {
  return new Set(members.map(bot => memberConnection(bot, connectionId))).size > 1
}

/** True when every bot in `selection` lives on ONE connection other than
 *  `connectionId`. The gateway here cannot own that room and no Desktop courier
 *  is needed between machines, so creation from here is refused: the user adds a
 *  bot from this connection or switches to that one. */
export function groupSelectionElsewhere(selection: RosterRow[], connectionId: null | string): boolean {
  return selection.length > 0 && !groupNeedsDesktopCourier(selection, connectionId) &&
    !botOnConnection(selection[0], connectionId)
}

/** The roster rows New Group Chat may seat: never an outage placeholder. Bots on
 *  the current connection must be ones the authority's seatable roster names
 *  (#141178); bots on another connection are offered when the Desktop can reach
 *  them, and a selection spanning connections becomes a Desktop-driven room. The
 *  ONE predicate behind both the dialog's picker and the menu item that opens it. */
export function seatableGroupBots(
  roster: RosterRow[],
  { connectionId, mode, seatable }: { connectionId: null | string; mode: GroupExecutionMode | 'checking'; seatable: null | Set<string> }
): RosterRow[] {
  return roster.filter(bot => {
    if (bot?.ghost) {
      return false
    }

    if (mode === 'legacy') {
      return true
    }

    if (!botOnConnection(bot, connectionId)) {
      return courierReachable(bot)
    }

    // The profile is the one createCanonicalGroup will send.
    return !seatable || seatable.has(bot.route?.targetProfile ?? bot.targetProfile ?? bot.name)
  })
}
