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
