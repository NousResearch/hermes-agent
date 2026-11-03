import type { ProfileScope } from '@/hermes'

import type { GatewayRequest } from './agent-plugins'
import { registryConnectionKind } from './connection-registry-state'
import { requestGatewayForAgent } from './gateway'

/** The registry connection a scope pins; null for a legacy profile label or the ambient backend. */
export function scopeConnectionId(scope: ProfileScope): null | string {
  return (scope && typeof scope === 'object' && scope.connectionId?.trim()) || null
}

/** Electron reconciles desktop halves only on this machine, so a remote owner never gets one. */
export function isRemotePluginOwner(scope: ProfileScope, foregroundRemote: boolean): boolean {
  const connectionId = scopeConnectionId(scope)

  return connectionId ? connectionId !== 'local' && registryConnectionKind(connectionId) !== 'local' : foregroundRemote
}

/** Pin all plugin RPCs to the selected backend and profile, not the foreground chat. */
export function pluginOwnerRequest(scope: ProfileScope, ambient: GatewayRequest): GatewayRequest {
  const connectionId = scopeConnectionId(scope)

  if (!connectionId) {
    return ambient
  }

  const profile = scope && typeof scope === 'object' ? scope.profile : null

  return <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) =>
    requestGatewayForAgent<T>(connectionId, String(params?.profile ?? profile ?? 'default'), method, params, timeoutMs)
}
