export type GroupExecutionMode = 'canonical' | 'legacy' | 'unavailable'

export function groupExecutionMode(value: unknown): GroupExecutionMode {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return 'unavailable'
  }

  const { driver, persistent_process, features } = value as Record<string, unknown>

  if (features !== undefined && (!Array.isArray(features) || features.some(feature => typeof feature !== 'string'))) {
    return 'unavailable'
  }

  if (driver === true) {return 'canonical'}

  // Canonical owners remain persistent even while their group driver is unavailable.
  if (driver === false && persistent_process === false && !features?.includes('canonical_session_owner')) {
    return 'legacy'
  }

  return 'unavailable'
}
