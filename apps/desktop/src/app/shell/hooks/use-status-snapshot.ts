import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { getStatus } from '@/hermes'
import { evaluateRuntimeReadiness, requiresProfileIdentity, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import type { StatusResponse } from '@/types/hermes'

const REFRESH_MS = 15_000

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export function useStatusSnapshot(gatewayState: string | undefined, requestGateway: GatewayRequester) {
  const profile = useStore($activeGatewayProfile)
  const connection = useStore($connection)
  const [statusSnapshot, setStatusSnapshot] = useState<StatusResponse | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<RuntimeReadinessResult | null>(null)

  useEffect(() => {
    let cancelled = false

    const refresh = async () => {
      try {
        const [next, inference] = await Promise.all([
          getStatus(),
          gatewayState === 'open'
            ? evaluateRuntimeReadiness(requestGateway, {
                profile,
                requireProfileIdentity: requiresProfileIdentity(connection)
              }).catch(error => ({
                checksDisagree: false,
                ready: false,
                reason: error instanceof Error ? error.message : String(error),
                source: 'fallback' as const
              }))
            : Promise.resolve(null)
        ])

        if (cancelled) {
          return
        }

        setStatusSnapshot(next)
        setInferenceStatus(inference)
      } catch {
        // Keep last snapshot through transient gateway flaps.
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), REFRESH_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [connection, gatewayState, profile, requestGateway])

  return { inferenceStatus, statusSnapshot }
}
