import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'

export type GroupExecutionMode = 'canonical' | 'legacy' | 'unavailable'

export function groupExecutionMode(value: unknown, error?: unknown, previous?: GroupExecutionMode): GroupExecutionMode {
  if (error !== undefined) {
    return (typeof error === 'object' && error !== null && 'code' in error && error.code === -32601) || previous === 'legacy'
      ? 'legacy' : 'unavailable'
  }

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return 'unavailable'
  }

  const { driver, methods } = value as Record<string, unknown>

  if (!Array.isArray(methods) || methods.some(method => typeof method !== 'string')) {
    return 'unavailable'
  }

  return methods.includes('groups.discard') ? driver === true ? 'canonical' : 'unavailable' : 'legacy'
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
