import { gatewayActivationEpoch, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { canonicalGroupRequest } from './canonical-groups'
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

  if (driver === true) {
    return 'canonical'
  }

  // App-managed hosted owners can be nonpersistent, even when their driver stops.
  const ownsRooms =
    Object.hasOwn(capability, 'protocol_version') ||
    Object.hasOwn(capability, 'authority_gateway_id') ||
    methods?.includes('groups.create') ||
    features?.includes('canonical_session_owner')

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
export function groupCreationSource(
  route: { connectionId: string; profile: string },
  activationEpoch = gatewayActivationEpoch()
) {
  return () =>
    gatewayActivationEpoch() === activationEpoch &&
    route.connectionId === host.state.connectionId.get() &&
    route.profile === host.state.profile.get() &&
    host.state.gateway.get() === 'open'
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
  const activationEpoch = gatewayActivationEpoch()
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

    return () => {
      cancelled = true
    }
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

/** The roster rows New Group Chat may seat: never an outage placeholder; in
 *  legacy mode every bot, otherwise only bots on the current connection that
 *  the authority's seatable roster names (#141178). The ONE predicate behind
 *  both the dialog's picker and the menu item that opens it. */
export function seatableGroupBots(
  roster: RosterRow[],
  {
    connectionId,
    mode,
    seatable
  }: { connectionId: null | string; mode: GroupExecutionMode | 'checking'; seatable: null | Set<string> }
): RosterRow[] {
  const currentConnection = String(connectionId || '').trim()

  const belongsToCurrentConnection = (bot: RosterRow) => {
    if (!currentConnection) {
      return false
    }

    if (bot.connectionId) {
      return (
        bot.connectionId === currentConnection &&
        (!bot.route?.connectionId || bot.route.connectionId === currentConnection)
      )
    }

    return !bot.remoteSource && (!bot.route?.connectionId || bot.route.connectionId === currentConnection)
  }

  return roster.filter(bot => {
    if (bot?.ghost) {
      return false
    }

    if (mode === 'legacy') {
      return true
    }

    // The profile is the one createCanonicalGroup will send.
    return (
      belongsToCurrentConnection(bot) &&
      (!seatable || seatable.has(bot.route?.targetProfile ?? bot.targetProfile ?? bot.name))
    )
  })
}
