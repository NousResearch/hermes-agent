import { $apiRequestScope } from '@/api/client'

import { $desktopMetricsGate, type DesktopActionVia, desktopMetricsEnabled, recordAction } from './desktop-metrics'
import type { IaAction } from './ia-metrics-contract'
import { $portalPreferred } from './shell-variant'

// A consent change invalidates in-flight work even if collection is subsequently re-enabled.
let consentGeneration = 0
$desktopMetricsGate.listen(() => consentGeneration++)
$portalPreferred.listen(() => consentGeneration++)
$apiRequestScope.listen((next, previous) => {
  if (next.connectionId !== previous.connectionId || next.profile !== previous.profile) {
    consentGeneration++
  }
})

export function recordIaAction(action: IaAction, via: DesktopActionVia = 'click'): void {
  if ($portalPreferred.get()) {
    recordAction(action, via)
  }
}

/** Async outcomes belong to the consenting profile that started the operation. */
export function captureIaRecorder(): (action: IaAction) => void {
  const scope = $apiRequestScope.get()
  const generation = consentGeneration
  const enabled = desktopMetricsEnabled() && $portalPreferred.get()

  return action => {
    const current = $apiRequestScope.get()

    if (
      enabled &&
      generation === consentGeneration &&
      scope.connectionId === current.connectionId &&
      scope.profile === current.profile
    ) {
      recordIaAction(action, 'state')
    }
  }
}
