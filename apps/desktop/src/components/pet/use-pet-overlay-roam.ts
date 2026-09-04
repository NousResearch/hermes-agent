import { useEffect } from 'react'

import { type OverlayRoamOptions, startPetOverlayRoam } from './overlay-roam-controller'

export * from './overlay-roam-controller'

/** React lifecycle adapter for the imperative overlay roam controller. */
export function usePetOverlayRoam(options: OverlayRoamOptions): void {
  const { enabled, isInteracting, loopMs, petH, petW, replanKey = 0 } = options

  useEffect(
    () => startPetOverlayRoam({ enabled, isInteracting, loopMs, petH, petW, replanKey }),
    [enabled, isInteracting, loopMs, petH, petW, replanKey]
  )
}
