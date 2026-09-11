import { useEffect, useState } from 'react'

import { getStatus } from '@/hermes'
import { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { refreshFreeTierStatus, setFreeTierRoute } from '@/store/free-tier'
import { $setupReadyTick } from '@/store/live-sync'
import { dismissNotification, notify } from '@/store/notifications'
import type { StatusResponse } from '@/types/hermes'

// Statusbar health is ambient chrome, not live data — nothing the user acts on
// within seconds. 60s + an actively-viewed check keeps traffic low; focus and
// visibility listeners refresh immediately on return.
const REFRESH_MS = 60_000

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export function useStatusSnapshot(
  gatewayState: string | undefined,
  requestGateway: GatewayRequester,
  gatewayScope = ''
) {
  const [statusSnapshot, setStatusSnapshot] = useState<StatusResponse | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<RuntimeReadinessResult | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    let sharedProfileWarning: string = ''
    let sharedProfileNoticeId: string | undefined

    // Status and inference readiness belong to one backend. A source switch
    // can keep gatewayState="open" throughout, so clear the previous source's
    // snapshot and start a fresh scoped request explicitly.
    setStatusSnapshot(null)
    setInferenceStatus(null)

    // A closed/connecting gateway cannot have an authoritative live-runtime
    // result. Clear readiness before starting the REST status leg so a hung
    // getStatus() cannot leave a stale "ready" state visible after disconnect.
    if (gatewayState !== 'open') {
      setInferenceStatus(null)
    }

    const scheduleRefresh = () => {
      if (!cancelled) {
        timer = window.setTimeout(() => void refresh({ readiness: false }), REFRESH_MS)
      }
    }

    const isViewed = () =>
      // macOS commonly leaves an occluded BrowserWindow `visible`; focus is
      // the missing signal that prevents status + readiness RPCs while the
      // user is working in another app.
      document.visibilityState === 'visible' && document.hasFocus()

    // Inference readiness + the free-tier verdict. Not on the periodic tick:
    // both change only at seams the backend announces (`setup.ready` at boot)
    // or that this window crosses (open, return from another app), so they
    // run once per seam instead of every 60s.
    const refreshReadiness = async () => {
      if (gatewayState !== 'open') {
        return
      }

      // The free-tier verdict is a local, zero-network read that writes
      // straight to its own store and swallows its failures — nothing here
      // waits on it or reads the result.
      const [inferenceResult] = await Promise.allSettled([
        evaluateRuntimeReadiness(requestGateway),
        refreshFreeTierStatus(requestGateway)
      ])

      if (cancelled || inferenceResult.status !== 'fulfilled') {
        return
      }

      const inference = inferenceResult.value

      if (inference.source !== 'fallback') {
        // runtime_check/setup_status returned an authoritative boolean.
        // A fallback means both RPCs failed or returned no boolean, so it
        // is a transient/unknown transport state, not proof that inference
        // became unconfigured. Keep the last authoritative result instead
        // of flashing "Inference not ready" during a gateway flap.
        setInferenceStatus(inference)
        setFreeTierRoute(inference.freeTier)
      }
    }

    const refresh = async ({ readiness }: { readiness: boolean }) => {
      if (!isViewed()) {
        scheduleRefresh()

        return
      }

      try {
        // Wait for every leg before scheduling the next refresh. setInterval
        // allowed a slow runtime check to overlap with later polls, which
        // multiplied load on an already-busy gateway and let stale failures
        // race newer healthy results.
        const [statusResult] = await Promise.allSettled([
          getStatus(),
          readiness ? refreshReadiness() : Promise.resolve()
        ])

        if (cancelled) {
          return
        }

        if (statusResult.status === 'fulfilled') {
          setStatusSnapshot(statusResult.value)
          const warning: string = statusResult.value.shared_profile_warning || ''

          // Keep dismissal until the conflict clears. A new overlap can warn again.
          if (warning !== sharedProfileWarning) {
            if (sharedProfileNoticeId) {
              dismissNotification(sharedProfileNoticeId)
            }

            sharedProfileWarning = warning
            sharedProfileNoticeId = warning ? notify({ kind: 'warning', message: warning }) : undefined
          }
        }
      } finally {
        scheduleRefresh()
      }
    }

    const onReturn = () => {
      if (isViewed() && !cancelled) {
        if (timer !== undefined) {
          window.clearTimeout(timer)
        }

        void refresh({ readiness: true })
      }
    }

    // `setup.ready` (routed by the gateway-event lifecycle handler for the
    // active source only) says the boot bootstrap just settled the route: one
    // readiness round now, so the chip/strip/onboarding move at once. Rides
    // outside the status tick so it neither resets nor waits on the timer.
    const unsubscribeSetupReady = $setupReadyTick.listen(() => void refreshReadiness())

    document.addEventListener('visibilitychange', onReturn)
    window.addEventListener('focus', onReturn)
    void refresh({ readiness: true })

    return () => {
      cancelled = true
      unsubscribeSetupReady()
      document.removeEventListener('visibilitychange', onReturn)
      window.removeEventListener('focus', onReturn)

      if (sharedProfileNoticeId) {
        dismissNotification(sharedProfileNoticeId)
      }

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [gatewayScope, gatewayState, requestGateway])

  return { inferenceStatus, statusSnapshot }
}
