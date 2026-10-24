import type { Details } from 'electron'

import type { NvidiaEglFallbackOptions, NvidiaEglMarker } from './linux-nvidia-egl-fallback'
import {
  decideNvidiaEglFallback,
  nvidiaEglFallbackAllowed,
  nvidiaEglFallbackMarker,
  nvidiaEglMarkerAfterSuccessfulBoot,
  readNvidiaEglMarker,
  shouldRelaunchForNvidiaGpuDeath,
  writeNvidiaEglMarker
} from './linux-nvidia-egl-fallback'

interface NvidiaEglRecoveryOptions extends Omit<NvidiaEglFallbackOptions, 'marker'> {
  userData: string
  queueMarker: (write: () => void) => void
  relaunch: () => void
  log: (message: string) => void
}

/** Persist only for the lock owner, and never turn intentional teardown into GPU failure. */
export function createNvidiaEglRecovery(options: NvidiaEglRecoveryOptions) {
  const previous = readNvidiaEglMarker(options.userData)
  const decision = decideNvidiaEglFallback({ ...options, marker: previous })
  const supported = (options.platform ?? process.platform) === 'linux' && options.driverMajor !== null
  let primary = false
  let quitting = false
  let relaunchAttempted = false

  const persist = (marker: NvidiaEglMarker): void => {
    try {
      writeNvidiaEglMarker(options.userData, marker)
    } catch (error) {
      options.log(`[gpu] could not persist NVIDIA recovery state: ${String(error)}`)
    }
  }

  if (supported) {
    options.queueMarker(() => {
      primary = true
      persist(decision.nextMarker)
    })
  }

  return {
    decision,
    startupMessage: `[gpu] NVIDIA ANGLE fallback=${decision.enable} (${decision.reason ?? 'not selected'}); previous=${previous?.state ?? 'none'}; build=${options.buildIdentity ?? options.appVersion}; driver=${options.driverVersion}`,
    healthy(): void {
      if (!primary || quitting || relaunchAttempted) {
        return
      }

      persist(nvidiaEglMarkerAfterSuccessfulBoot({ ...options, fallbackActive: decision.enable }))
    },
    cleanExit(): void {
      quitting = true

      if (!primary || decision.enable || relaunchAttempted) {
        return
      }

      const marker = readNvidiaEglMarker(options.userData)

      // Do not erase a fallback written by a recovery relaunch or a different build.
      if (
        marker?.state === 'booting' &&
        marker.version === decision.nextMarker.version &&
        marker.driverVersion === decision.nextMarker.driverVersion &&
        marker.build === decision.nextMarker.build
      ) {
        persist({ state: 'ok' })
      }
    },
    gpuDied(details: Pick<Details, 'type' | 'reason' | 'exitCode'>, teardownInProgress = false): boolean {
      if (
        !primary ||
        quitting ||
        teardownInProgress ||
        !nvidiaEglFallbackAllowed(options) ||
        !shouldRelaunchForNvidiaGpuDeath({
          platform: options.platform,
          details,
          fallbackActive: decision.enable,
          relaunchAttempted
        })
      ) {
        return false
      }

      relaunchAttempted = true
      persist(
        nvidiaEglFallbackMarker(
          options.appVersion ?? '',
          options.driverVersion ?? String(options.driverMajor),
          options.buildIdentity
        )
      )
      options.log(
        `[gpu] NVIDIA GPU process died (${details.reason}, exit=${details.exitCode}); relaunching with SwiftShader`
      )

      try {
        options.relaunch()
      } catch (error) {
        options.log(`[gpu] NVIDIA SwiftShader relaunch failed: ${String(error)}`)
      }

      return true
    }
  }
}
