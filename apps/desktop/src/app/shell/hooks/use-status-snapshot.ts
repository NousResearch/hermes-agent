import { useEffect, useRef, useState } from 'react'

import { getStatus } from '@/hermes'
import { type I18nContextValue, useI18n } from '@/i18n'
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

interface ScopedStatus {
  inferenceStatus: RuntimeReadinessResult | null
  statusSnapshot: StatusResponse | null
}

export function useStatusSnapshot(
  gatewayState: string | undefined,
  requestGateway: GatewayRequester,
  gatewayScope: string = ''
): { inferenceStatus: RuntimeReadinessResult | null; statusSnapshot: StatusResponse | null } {
  const { t }: I18nContextValue = useI18n()
  const warningMessage: string = t.notifications.sharedProfileWarning
  const [statusSnapshot, setStatusSnapshot] = useState<StatusResponse | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<RuntimeReadinessResult | null>(null)
  const cacheRef = useRef(new Map<string, ScopedStatus>())

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    let sharedProfileWarning: boolean = false
    let sharedProfileNoticeId: string | undefined

    const cache = cacheRef.current

    // A disconnect invalidates every scope, including requests still in flight.
    if (gatewayState !== 'open') {
      cache.clear()
    }

    const cached = cache.get(gatewayScope)

    const scopedStatus: ScopedStatus = {
      inferenceStatus: cached?.inferenceStatus ?? null,
      statusSnapshot: cached?.statusSnapshot ?? null
    }

    cache.set(gatewayScope, scopedStatus)

    // Revalidate this scope's own last answers without blanking healthy chrome
    // on a chat switch. An unseen scope still starts with no trusted answers.
    setStatusSnapshot(scopedStatus.statusSnapshot)
    setInferenceStatus(scopedStatus.inferenceStatus)

    const remember = (patch: Partial<ScopedStatus>): boolean => {
      // Entry identity is a run token: a late answer can warm its owner after
      // a switch, but cannot overwrite a newer run or survive a disconnect.
      if (cache.get(gatewayScope) !== scopedStatus) {
        return false
      }

      Object.assign(scopedStatus, patch)

      return true
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

      if (inferenceResult.status !== 'fulfilled') {
        return
      }

      const inference = inferenceResult.value

      // Transport fallback is unknown, not proof that inference lost its
      // credentials. Keep the last authoritative result in both view and cache.
      if (inference.source === 'fallback' || !remember({ inferenceStatus: inference }) || cancelled) {
        return
      }

      setInferenceStatus(inference)
      setFreeTierRoute(inference.freeTier)
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

        if (statusResult.status === 'fulfilled') {
          const next = statusResult.value
          const previous = scopedStatus.statusSnapshot
          // Preserve reference identity on a no-op: the 60s tick re-reads a
          // usually-unchanged snapshot, and a fresh object for the same content
          // re-renders every consumer for nothing.
          const value = JSON.stringify(previous) === JSON.stringify(next) ? previous : next

          if (!remember({ statusSnapshot: value }) || cancelled) {
            return
          }

          setStatusSnapshot(value)
          const warning: boolean = Boolean(statusResult.value.shared_profile_warning)

          // Keep dismissal until the conflict clears. A new overlap can warn again.
          if (warning !== sharedProfileWarning) {
            if (sharedProfileNoticeId) {
              dismissNotification(sharedProfileNoticeId)
            }

            sharedProfileWarning = warning
            sharedProfileNoticeId = warning ? notify({ kind: 'warning', message: warningMessage }) : undefined
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
  }, [gatewayScope, gatewayState, requestGateway, warningMessage])

  return { inferenceStatus, statusSnapshot }
}
