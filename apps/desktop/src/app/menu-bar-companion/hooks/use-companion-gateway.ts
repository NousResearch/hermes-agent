import { resolveGatewayWsUrl } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import * as React from 'react'

import { HermesGateway } from '@/hermes'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

type GatewayRequest = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/**
 * Lightweight gateway socket for the menu-bar companion window.
 * Reuses the same Desktop backend connection as the main app — no second runtime.
 */
export function useCompanionGateway() {
  const profile = normalizeProfileKey(useStore($activeGatewayProfile))
  const [ready, setReady] = React.useState(false)
  const [error, setError] = React.useState('')
  const gatewayRef = React.useRef<HermesGateway | null>(null)

  React.useEffect(() => {
    let cancelled = false
    const gateway = new HermesGateway()
    gatewayRef.current = gateway

    const boot = async () => {
      try {
        const desktop = window.hermesDesktop

        if (!desktop) {
          throw new Error('Desktop bridge unavailable')
        }

        const conn = await desktop.getConnection(profile)
        const wsUrl = await resolveGatewayWsUrl(desktop, conn)
        await gateway.connect(wsUrl)

        if (!cancelled) {
          setReady(true)
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setReady(false)
          setError(err instanceof Error ? err.message : String(err))
        }
      }
    }

    void boot()

    return () => {
      cancelled = true
      gateway.close()
      gatewayRef.current = null
    }
  }, [profile])

  const request = React.useCallback<GatewayRequest>(
    async (method, params = {}) => {
      const gateway = gatewayRef.current

      if (!gateway || gateway.connectionState !== 'open') {
        throw new Error(error || 'Gateway not connected')
      }

      return gateway.request(method, params) as Promise<never>
    },
    [error]
  )

  const retry = React.useCallback(() => {
    setError('')
    setReady(false)
    const gateway = gatewayRef.current

    if (!gateway) {
      return
    }
    void (async () => {
      try {
        const desktop = window.hermesDesktop

        if (!desktop) {
          throw new Error('Desktop bridge unavailable')
        }

        const conn = await desktop.getConnection(profile)
        const wsUrl = await resolveGatewayWsUrl(desktop, conn)
        await gateway.connect(wsUrl)
        setReady(true)
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    })()
  }, [profile])

  return { ready, error, request, retry }
}
