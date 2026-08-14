// Renderer end of the diagnostics arming contract.
//
// The capture controller lives in Electron main (it is the only process that
// can reach the renderer AND the gateway), so main owns the capture id and the
// wall-clock anchor and pushes them down:
//
//   main → renderer  'diagnostics:arm'     { captureId, wallClockAnchorMs }
//   main → renderer  'diagnostics:disarm'  (no payload)
//   main → renderer  'diagnostics:collect' { requestId, captureId }  (U4)
//
// The first two arrive through the usual preload subscription shape
// (`window.hermesDesktop.diagnostics.onArm/onDisarm`, each returning its own
// unsubscribe). The bridge is optional at runtime: a renderer running against a
// preload without the diagnostics surface (older shell, tests, the browser
// harness) simply never arms, which is the disarmed — zero cost — state.
//
// The third is a PULL at export time, and preload answers it on this module's
// behalf: `setSnapshotProvider` hands preload a function, preload owns the
// channel and the reply correlation. Without this registration a window is
// armed but silent — it records events into its ring and never contributes
// them to a bundle — so `initDiagnosticsSnapshots()` runs beside
// `initDiagnosticsArming()` at renderer startup.

import { armDiagnostics, disarmDiagnostics, readDiagnosticsCapture } from './capture'

export interface DiagnosticsArmPayload {
  captureId: string
  wallClockAnchorMs: number
}

let teardown: null | (() => void) = null
let snapshotTeardown: null | (() => void) = null

/** Subscribe to main's arm/disarm pushes. Idempotent; returns the unsubscribe. */
export function initDiagnosticsArming(): () => void {
  if (teardown) {
    return teardown
  }

  const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop?.diagnostics

  if (!bridge) {
    return () => undefined
  }

  const offArm = bridge.onArm(payload => {
    if (payload && typeof payload.captureId === 'string') {
      armDiagnostics(payload.captureId, Number(payload.wallClockAnchorMs) || Date.now())
    }
  })

  const offDisarm = bridge.onDisarm(() => disarmDiagnostics())

  teardown = () => {
    offArm()
    offDisarm()
    teardown = null
    disarmDiagnostics()
  }

  return teardown
}

/**
 * Register this renderer's snapshot source for main's export-time collect pull.
 * Idempotent; returns the unsubscribe.
 *
 * The provider answers with the running capture's snapshot, or null when this
 * window is disarmed or is answering for a *different* capture — a stale
 * snapshot would land in the bundle under the wrong capture id and silently
 * corrupt the KTD3 alignment, so it is better to contribute nothing.
 */
export function initDiagnosticsSnapshots(): () => void {
  if (snapshotTeardown) {
    return snapshotTeardown
  }

  const register = typeof window === 'undefined' ? undefined : window.hermesDesktop?.diagnostics?.setSnapshotProvider

  if (!register) {
    return () => undefined
  }

  const off = register(captureId => {
    const snapshot = readDiagnosticsCapture()

    if (!snapshot || (captureId && snapshot.captureId !== captureId)) {
      return null
    }

    return snapshot
  })

  snapshotTeardown = () => {
    off()
    snapshotTeardown = null
  }

  return snapshotTeardown
}
