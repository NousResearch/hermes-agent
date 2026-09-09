import { isGatewayReauthRequired, resolveGatewayWsUrl } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'

import type { HermesGateway } from '@/hermes'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { $gateway, ensureActiveGatewayOpen, isActivePrimary } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState, setConnection } from '@/store/session'

export function useGatewayRequest() {
  const gatewayState = useStore($gatewayState)
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

  const gatewayStateRef = useRef(gatewayState)
  const reconnectingRef = useRef<Promise<HermesGateway | null> | null>(null)
  // Holds the reauth error from the most recent failed reconnect so
  // requestGateway can surface the gateway's "session expired, sign in again"
  // message instead of the opaque "connection closed" that triggered the retry.
  const reauthErrorRef = useRef<unknown>(null)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    gatewayStateRef.current = gatewayState
  }, [gatewayState])

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

    if (gatewayStateRef.current === 'open') {
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
        // Reconnect to whichever profile the gateway is currently routed to (not
        // always the primary), so a sleep/wake reconnect keeps the user on the
        // profile they were chatting in. Both awaits below are IPC round-trips
        // into the main process with no timeout of their own (#93454) — a
        // wedged main-process round-trip otherwise hangs this await forever,
        // latching reconnectingRef.current so every later requestGateway() call
        // returns the same never-settling promise. Bound the same way
        // use-gateway-boot.ts bounds the primary boot/soft-switch equivalents.
        const conn = await withTimeout(
          desktop.getConnection($activeGatewayProfile.get()),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out reconnecting to Hermes backend'
        )

        connectionRef.current = conn
        setConnection(conn)

        // Re-mint the WS URL before reconnecting. OAuth tickets are single-use
        // and short-lived, so the cached conn.wsUrl ticket is dead here;
        // resolveGatewayWsUrl() never connects with a stale ticket. An explicit
        // auth rejection becomes a reauth error; transport failures remain
        // retryable. Stash only the former so requestGateway can show the
        // actionable "sign in again" message.
        const wsUrl = await withTimeout(
          resolveGatewayWsUrl(desktop, conn),
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
      const gateway = gatewayRef.current

      if (!gateway) {
        throw new Error('Hermes gateway unavailable')
      }

      try {
        return await gateway.request<T>(method, params, timeoutMs, signal)
      } catch (error) {
        if (!isGatewayTransportError(error)) {
          throw error
        }

        // Primary keeps the OAuth-aware reconnect (remote gateways re-mint a
        // single-use ticket). Background profiles stay on the registry's
        // connection-owned reconnect path, including composite remote/SSH
        // sources.
        const recovered = isActivePrimary() ? await ensureGatewayOpen() : await ensureActiveGatewayOpen()

        if (!recovered) {
          // Prefer the reauth error from the failed reconnect (OAuth session
          // expired) over the generic transport error that triggered the retry.
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
