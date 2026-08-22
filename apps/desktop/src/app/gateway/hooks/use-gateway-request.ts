import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'

import type { HermesGateway } from '@/hermes'
import {
  $gateway,
  acquireGatewayRequestLease,
  ensureActiveGatewayOpen,
  isActivePrimary,
  reconnectPrimaryGateway
} from '@/store/gateway'
import { setConnection } from '@/store/session'

export function useGatewayRequest() {
  // Reactive companion to `gatewayRef`. The ref exists so `requestGateway`
  // keeps a stable identity and always reaches the live socket, but it is only
  // populated by the subscription effect below — i.e. AFTER the first render.
  // A component that reads `gatewayRef.current` while rendering therefore sees
  // null on mount, and if the connection state doesn't happen to flip
  // afterwards it never re-renders to pick the instance up. Anything that needs
  // the gateway as a render-time VALUE (props, memo deps) must use this.
  const gateway = useStore($gateway) as HermesGateway | null
  const gatewayRef = useRef<HermesGateway | null>(null)

  const connectionRef = useRef<Awaited<ReturnType<NonNullable<typeof window.hermesDesktop>['getConnection']>> | null>(
    null
  )

  // Track the active gateway (primary or a background profile's socket) so
  // outbound requests and overlay props always target the focused profile.
  useEffect(
    () =>
      $gateway.subscribe(gateway => {
        gatewayRef.current = gateway as HermesGateway | null
      }),
    []
  )

  const ensureGatewayOpen = useCallback(async () => {
    const existing = gatewayRef.current

    if (!existing) {
      return null
    }

    if (existing.connectionState === 'open') {
      return existing
    }

    const conn = await reconnectPrimaryGateway(existing)

    if (!conn) {
      connectionRef.current = null
      setConnection(null)

      return null
    }

    connectionRef.current = conn
    setConnection(conn)

    return existing
  }, [])

  const requestGateway = useCallback(
    async <T>(method: string, params: Record<string, unknown> = {}, timeoutMs?: number, signal?: AbortSignal) => {
      const gateway = gatewayRef.current

      if (!gateway) {
        throw new Error('Hermes gateway unavailable')
      }

      try {
        return await gateway.request<T>(method, params, timeoutMs, signal)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)

        if (gateway.connectionState === 'open' || !/not connected|connection closed/i.test(message)) {
          throw error
        }

        // Primary keeps the OAuth-aware reconnect (remote gateways re-mint a
        // single-use ticket); background profiles are always local pool
        // backends, so the registry handles their reconnect with no reauth.
        const recovered = isActivePrimary() ? await ensureGatewayOpen() : await ensureActiveGatewayOpen()

        if (!recovered) {
          throw error
        }

        return recovered.request<T>(method, params, timeoutMs, signal)
      }
    },
    [ensureGatewayOpen]
  )

  const bindGatewayRequest = useCallback(
    (gateway: HermesGateway, profile: string) => acquireGatewayRequestLease(gateway, profile),
    []
  )

  return { bindGatewayRequest, connectionRef, gateway, gatewayRef, requestGateway }
}
