import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'

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
