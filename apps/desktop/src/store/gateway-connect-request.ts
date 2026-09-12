import { atom } from 'nanostores'

import type { GatewayConnectRequest } from '@/lib/gateway-connect-link'

export const $gatewayConnectRequest = atom<GatewayConnectRequest | null>(null)

export function requestGatewayConnect(request: GatewayConnectRequest): void {
  if (!$gatewayConnectRequest.get()) {
    $gatewayConnectRequest.set(request)
  }
}

export function closeGatewayConnect(): void {
  $gatewayConnectRequest.set(null)
}
