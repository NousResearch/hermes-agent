import { useEffect, useState } from 'react'

import { getStatus } from '@/hermes'
import { type I18nContextValue, useI18n } from '@/i18n'
import { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { $freeTierStatus, refreshFreeTierStatus, setFreeTierRoute } from '@/store/free-tier'
import { $setupReadyTick } from '@/store/live-sync'
import { dismissNotification, notify } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'
import type { StatusResponse } from '@/types/hermes'

// Statusbar health is ambient chrome, not live data — nothing the user acts on
// within seconds. 60s + an actively-viewed check keeps traffic low; focus and
// visibility listeners refresh immediately on return.
const REFRESH_MS = 60_000

// The scope the cached free-tier verdict was read under. Module-level because
// $freeTierStatus is one app-wide atom, not per hook instance.
let freeTierScope: string | undefined

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export function useStatusSnapshot(
  gatewayState: string | undefined,
  requestGateway: GatewayRequester,
  gatewayScope: string = ''
): { inferenceStatus: RuntimeReadinessResult | null; statusSnapshot: StatusResponse | null } {
  const { t }: I18nContextValue = useI18n()
  const warningMessage: string = t.notifications.sharedProfileWarning
  const [statusSnapshot, setStatusSnapshot] = useState<StatusResponse | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<RuntimeReadinessResult | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    let sharedProfileWarning: boolean = false
    let sharedProfileNoticeId: string | undefined
    // Whether this run has published an authoritative readiness verdict. The
    // tick's callback cannot read the state it belongs to, and the flag's
    // lifetime is exactly this run's — the same run that clears the status on
    // entry — so the two can never disagree.
    let readinessAnswered = false

    const publishInferenceStatus = (next: RuntimeReadinessResult | null): void => {
      readinessAnswered = next !== null
      setInferenceStatus(next)
    }

    // Status and inference readiness belong to one backend. A source switch
    // can keep gatewayState="open" throughout, so clear the previous source's
    // snapshot and start a fresh scoped request explicitly.
    setStatusSnapshot(null)
    publishInferenceStatus(null)

    // The free-tier verdict belongs to one profile too. Drop it on a real switch
    // only: a flap on the same scope keeps the last answer (refreshFreeTierStatus).
    if (freeTierScope !== undefined && freeTierScope !== gatewayScope) {
      $freeTierStatus.set(null)
    }

    freeTierScope = gatewayScope

    // A closed/connecting gateway cannot have an authoritative live-runtime
    // result. Clear readiness before starting the REST status leg so a hung
    // getStatus() cannot leave a stale "ready" state visible after disconnect.
    if (gatewayState !== 'open') {
      publishInferenceStatus(null)
    }

    const scheduleRefresh = () => {
      if (!cancelled) {
        // Readiness rides the tick only while there is still no authoritative
        // answer to show. A round that came back a transport fallback leaves a
        // null status, which reads as "checking" — and the only other triggers
        // are seams this window may never cross again, so a gateway flap
        // outliving one seam would pin the chip for good. Once a verdict
        // exists the tick stays status-only, keeping the ambient poll cheap.
        timer = window.setTimeout(() => void refresh({ readiness: !readinessAnswered }), REFRESH_MS)
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
      // waits on it or reads the result. Unlike the readiness publish below,
      // its write happens inside the store, so hand it this run's liveness: a
      // reply that lands after a source/profile switch must not repaint the
      // shared atom the new run already answered.
      const [inferenceResult] = await Promise.allSettled([
        evaluateRuntimeReadiness(requestGateway),
        refreshFreeTierStatus(requestGateway, () => !cancelled)
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
        publishInferenceStatus(inference)
        setFreeTierRoute(inference.freeTier)
      }
    }

    const refresh = async ({ readiness, force = false }: { readiness: boolean; force?: boolean }) => {
      if (!force && !isViewed()) {
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
          const next = statusResult.value
          // Preserve reference identity on a no-op: the 60s tick re-reads a
          // usually-unchanged snapshot, and a fresh object for the same content
          // re-renders every consumer for nothing.
          setStatusSnapshot(previous => (JSON.stringify(previous) === JSON.stringify(next) ? previous : next))
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

    // An OAuth sign-in updates backend readiness, but ambient polling is
    // skipped while the window is unfocused — so a stale credential failure can
    // linger indefinitely if the user stays in the browser after completing the
    // flow. Watch the onboarding store for the auth-success transition and force
    // one readiness refresh that bypasses the focus gate. This does not resume
    // background polling: the forced refresh reschedules the normal (focus-gated)
    // cadence in its `finally`.
    let lastFlowStatus = $desktopOnboarding.get().flow.status

    const unsubscribeOnboarding = $desktopOnboarding.listen(state => {
      const status = state.flow.status
      const becameSuccess = status === 'success' && lastFlowStatus !== 'success'
      lastFlowStatus = status

      if (becameSuccess && !cancelled) {
        if (timer !== undefined) {
          window.clearTimeout(timer)
        }

        void refresh({ readiness: true, force: true })
      }
    })

    document.addEventListener('visibilitychange', onReturn)
    window.addEventListener('focus', onReturn)
    void refresh({ readiness: true })

    return () => {
      cancelled = true
      unsubscribeOnboarding()
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
