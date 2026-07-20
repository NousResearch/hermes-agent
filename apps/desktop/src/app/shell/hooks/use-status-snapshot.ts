import { useEffect, useState } from 'react'

import { getStatus } from '@/hermes'
import { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import type { StatusResponse } from '@/types/hermes'

const REFRESH_MS = 15_000

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export function useStatusSnapshot(gatewayState: string | undefined, requestGateway: GatewayRequester) {
  const [statusSnapshot, setStatusSnapshot] = useState<StatusResponse | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<RuntimeReadinessResult | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    // A closed/connecting gateway cannot have an authoritative live-runtime
    // result. Clear readiness before starting the REST status leg so a hung
    // getStatus() cannot leave a stale "ready" state visible after disconnect.
    if (gatewayState !== 'open') {
      setInferenceStatus(null)
    }

    const scheduleRefresh = () => {
      if (!cancelled) {
        timer = window.setTimeout(() => void refresh(), REFRESH_MS)
      }
    }

    const refresh = async () => {
      try {
        // Wait for both legs before scheduling the next refresh. setInterval
        // allowed a slow runtime check to overlap with later polls, which
        // multiplied load on an already-busy gateway and let stale failures
        // race newer healthy results.
        const [statusResult, inferenceResult] = await Promise.allSettled([
          getStatus(),
          gatewayState === 'open' ? evaluateRuntimeReadiness(requestGateway) : Promise.resolve(null)
        ])

        if (cancelled) {
          return
        }

        if (statusResult.status === 'fulfilled') {
          setStatusSnapshot(statusResult.value)
        }

        if (inferenceResult.status === 'fulfilled') {
          const inference = inferenceResult.value

          if (inference === null) {
            setInferenceStatus(null)
          } else if (inference.source !== 'fallback') {
            // runtime_check/setup_status returned an authoritative boolean.
            // A fallback means both RPCs failed or returned no boolean, so it
            // is a transient/unknown transport state, not proof that inference
            // became unconfigured. Keep the last authoritative result instead
            // of flashing "Inference not ready" during a gateway flap.
            setInferenceStatus(inference)
          }
        }
      } finally {
        scheduleRefresh()
      }
    }

    void refresh()

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [gatewayState, requestGateway])

  return { inferenceStatus, statusSnapshot }
}
