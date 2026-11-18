import type { Details } from 'electron'

import type { LinuxGpuBootWindow } from './linux-gpu-fallback'
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
  /** This process owns the single-instance lock; a secondary launch never persists. */
  primary: boolean
  /** Extra argv for the relaunch; non-empty only when the witness could not be persisted. */
  relaunch: (extraArgs: readonly string[]) => void
  argv?: readonly string[]
  log: (message: string) => void
  /** The launch's shared Linux boot clock; the caller records the first reveal on it. */
  bootWindow: LinuxGpuBootWindow
}

const SWIFTSHADER_SWITCH = '--use-angle=swiftshader'

/** Persist only for the lock owner, and never turn intentional teardown into GPU failure. */
export function createNvidiaEglRecovery(options: NvidiaEglRecoveryOptions) {
  const previous = readNvidiaEglMarker(options.userData)
  const decision = decideNvidiaEglFallback({ ...options, marker: previous })
  const supported = (options.platform ?? process.platform) === 'linux' && options.driverMajor !== null
  const primary = supported && options.primary
  let quitting = false
  let relaunchAttempted = false

  // A relaunch that carried the fallback on argv because the marker was unwritable.
  const argvFallback = (options.argv ?? process.argv).includes(SWIFTSHADER_SWITCH)

  const persist = (marker: NvidiaEglMarker): boolean => {
    try {
      writeNvidiaEglMarker(options.userData, marker)

      return true
    } catch (error) {
      options.log(`[gpu] could not persist NVIDIA recovery state: ${String(error)}`)

      return false
    }
  }

  // Before GPU children start: a `booting` record left by an abort is itself the witness.
  if (primary) {
    persist(decision.nextMarker)
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
        options.bootWindow.afterBootWindow() ||
        !nvidiaEglFallbackAllowed(options) ||
        !shouldRelaunchForNvidiaGpuDeath({
          platform: options.platform,
          details,
          fallbackActive: decision.enable || argvFallback,
          relaunchAttempted
        })
      ) {
        return false
      }

      relaunchAttempted = true

      const persisted = persist(
        nvidiaEglFallbackMarker(
          options.appVersion ?? '',
          options.driverVersion ?? String(options.driverMajor),
          options.buildIdentity
        )
      )

      options.log(
        `[gpu] NVIDIA GPU process died (${details.reason}, exit=${details.exitCode}); relaunching with SwiftShader${persisted ? '' : ' on argv (marker unwritable)'}`
      )

      try {
        // Without the witness on disk the next launch would repeat this one.
        options.relaunch(persisted ? [] : [SWIFTSHADER_SWITCH])
      } catch (error) {
        options.log(`[gpu] NVIDIA SwiftShader relaunch failed: ${String(error)}`)
      }

      return true
    }
  }
}
