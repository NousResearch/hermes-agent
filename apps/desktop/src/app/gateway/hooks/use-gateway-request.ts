import { isGatewayReauthRequired } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'

import type { HermesGateway } from '@/hermes'
import { resolveDesktopGatewayWsUrl } from '@/lib/gateway-ws-url'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { $gateway, activeGateway, ensureActiveGatewayOpen, isActivePrimary } from '@/store/gateway'
import { $gatewayState, setConnection } from '@/store/session'

export function useGatewayRequest() {
  const gatewayState = useStore($gatewayState)
  const gateway = useStore($gateway) as HermesGateway | null
  const gatewayRef = useRef<HermesGateway | null>(null)
  const connectionRef = useRef<Awaited<ReturnType<NonNullable<typeof window.hermesDesktop>['getConnection']>> | null>(
    null
  )
  const gatewayStateRef = useRef(gatewayState)
  const reconnectingRef = useRef<Promise<HermesGateway | null> | null>(null)
  const reauthErrorRef = useRef<unknown>(null)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    gatewayStateRef.current = gatewayState
  }, [gatewayState])

  useEffect(
    () =>
      $gateway.subscribe(gateway => {
        gatewayRef.current = gateway as HermesGateway | null
      }),
    []
  )

  const ensureGatewayOpen = useCallback(async () => {
    const existing = gatewayRef.current ?? activeGateway()

    if (!existing) {
      return null
    }

    if (gatewayStateRef.current === 'open' && existing.connectionState === 'open') {
      return existing
    }

    if (reconnectingRef.current) {
      return reconnectingRef.current
    }

    reconnectingRef.current = (async () => {
      const desktop = window.hermesDesktop

      if (!desktop) {
        return null
      }

      reauthErrorRef.current = null

      try {
        const conn = await withTimeout(
          desktop.getConnection(),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out reconnecting to Hermes backend'
        )

        connectionRef.current = conn
        setConnection(conn)

        const wsUrl = await withTimeout(
          resolveDesktopGatewayWsUrl(desktop, conn),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out re-minting the gateway WebSocket URL'
        )

        await existing.connect(wsUrl)

        return existing
      } catch (error) {
        if (isGatewayReauthRequired(error)) {
          reauthErrorRef.current = error
        }

        connectionRef.current = null
        setConnection(null)

        return null
      } finally {
        reconnectingRef.current = null
      }
    })()

    return reconnectingRef.current
  }, [])

  const requestGateway = useCallback(
    async <T>(method: string, params: Record<string, unknown> = {}, timeoutMs?: number, signal?: AbortSignal) => {
      const gateway = gatewayRef.current ?? activeGateway()

      if (!gateway) {
        throw new Error('Hermes gateway unavailable')
      }

      try {
        return await gateway.request<T>(method, params, timeoutMs, signal)
      } catch (error) {
        if (!isGatewayTransportError(error)) {
          throw error
        }

        const recovered = isActivePrimary() ? await ensureGatewayOpen() : await ensureActiveGatewayOpen()

        if (!recovered) {
          const reauthError = reauthErrorRef.current
          reauthErrorRef.current = null

          if (reauthError) {
            throw reauthError
          }

          throw error
        }

        return recovered.request<T>(method, params, timeoutMs, signal)
      }
    },
    [ensureGatewayOpen]
  )

  return { connectionRef, gateway, gatewayRef, requestGateway }
}

const GATEWAY_TRANSPORT_ERROR_CODES = new Set([
  'ECONNABORTED',
  'ECONNREFUSED',
  'ECONNRESET',
  'EHOSTUNREACH',
  'ENETUNREACH',
  'ENOTFOUND',
  'EPIPE',
  'ETIMEDOUT',
  'ERR_NETWORK',
  'ERR_SOCKET_CLOSED'
])

function errorCode(value: unknown): string | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const code = (value as { code?: unknown }).code

  return typeof code === 'string' ? code.toUpperCase() : null
}

function isGatewayTransportError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  if (/not connected|connection closed|connection reset|ECONNRESET/i.test(message)) {
    return true
  }

  const cause = typeof error === 'object' && error !== null ? (error as { cause?: unknown }).cause : undefined

  return [error, cause].some(value => {
    const code = errorCode(value)

    return code !== null && GATEWAY_TRANSPORT_ERROR_CODES.has(code)
  })
}
