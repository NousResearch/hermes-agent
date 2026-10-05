// Desktop wiring of an update hold (R8 D3): the marker probe the boot and
// pool-backend gates share, the blocked boot screen's state, and its three IPC
// ways out. The judgement itself lives in update-marker-gate.ts; main.ts keeps
// only the call sites.

import path from 'node:path'

import type { IpcMain, IpcMainInvokeEvent } from 'electron'

import { cachedCreateTimeProbe } from './update-marker'
import {
  allowStartOverHold,
  type HeldState,
  heldWaitMessage,
  HOLD_SCREEN_GRACE_MS,
  liveMarkerProbe,
  PRIMARY_HOLD_OWNER,
  requestHoldRecheck,
  startAnywayLogLine,
  UpdateHoldBoard
} from './update-marker-gate'
import { runMarkerHelper } from './updater/marker-helper'

// What the blocked boot screen shows (R8 D3). The renderer mirrors this as
// `DesktopUpdateHold` in src/global.d.ts.
export interface UpdateHoldWire {
  holdId: string
  verdict: 'held' | 'busy' | 'error'
  ownerPid: number | null
  since: number
  checkedAt: number
  logPath: string
}

export interface MarkerGateCallbacks {
  onLiveMarker?: (marker: { startedAt: number | null }) => void
  onHeld?: (state: HeldState) => void
  onOverride?: (holdId: string) => void
}

export interface MarkerGateHost {
  hermesHome: string
  isWindows: boolean
  log: (line: string) => void
  updateRoot: () => string
}

export function markerGateProbe(host: MarkerGateHost, { onLiveMarker, onHeld, onOverride }: MarkerGateCallbacks = {}) {
  // One creation-time probe per pid for this wait: on Windows each probe is a
  // powershell spawn and the gate polls every second.
  const createTime = cachedCreateTimeProbe()

  // Owner liveness (pid + creation time) only: a failed receipt never
  // outranks a live marker — `latest.json` is written at finalize, so a retry
  // after a failed update still reads "failed" while the new one runs (V2).
  // A dead marker is never deleted here (A7 rule 3); the checkout script's
  // helper decides under its lock whether a completion still holds the
  // checkout (R6) — see update-marker-gate.ts.
  return liveMarkerProbe({
    hermesHome: host.hermesHome,
    createTime,
    onLiveMarker,
    onHeld,
    onOverride,
    log: host.log,
    // A missing or pre-protocol-2 script answers `unsupported`; one that
    // exists but cannot be read answers `error` (R8 M5).
    reclaim: () =>
      runMarkerHelper('reclaim', {
        updateRoot: host.updateRoot(),
        hermesHome: host.hermesHome,
        isWindows: host.isWindows
      })
  })
}

/**
 * A wait's blocked-screen clock: the hold to show once a blocking marker hold
 * has lasted HOLD_SCREEN_GRACE_MS without a break, else null. Never a timeout:
 * past the grace the wait stays parked behind the screen.
 */
export function holdGraceClock(now: () => number = Date.now) {
  let blockedSince: number | null = null

  return (reason: string | null, held: HeldState | null): HeldState | null => {
    blockedSince = reason === 'marker' && held?.blocking ? (blockedSince ?? now()) : null

    return held && blockedSince !== null && now() - blockedSince >= HOLD_SCREEN_GRACE_MS ? held : null
  }
}

export interface UpdateHoldScreenHost {
  hermesHome: string
  log: (line: string) => void
  /** The hold the boot progress currently carries. */
  bootHold: () => UpdateHoldWire | null
  updateBootProgress: (update: Record<string, unknown>) => void
}

/**
 * Every wait blocked past the grace: the primary boot and each pool/profile
 * backend (R8 M6). The screen shows the primary's, else the first pool one.
 */
export function createUpdateHoldScreen(host: UpdateHoldScreenHost) {
  // The hold the boot screen currently shows; the IPC handlers act on it only
  // (a Start anyway for a hold the user never saw is refused).
  let current: HeldState | null = null
  const board = new UpdateHoldBoard()

  const wire = (state: HeldState): UpdateHoldWire => ({
    holdId: state.holdId,
    verdict: state.verdict === 'live' ? 'held' : state.verdict,
    ownerPid: state.ownerPid,
    since: state.since,
    checkedAt: state.checkedAt,
    logPath: path.join(host.hermesHome, 'logs', 'update.log')
  })

  function render(state: HeldState, bootPhase: boolean) {
    const previous = current
    const sameHold = previous?.holdId === state.holdId && previous.verdict === state.verdict

    if (sameHold && previous.checkedAt === state.checkedAt && host.bootHold()) {
      return
    }

    if (previous?.holdId !== state.holdId) {
      host.log(
        `[updates] boot blocked: the update marker is ${state.verdict}` +
          `${state.ownerPid ? ` (update pid ${state.ownerPid}, exited)` : ''}, hold ${state.holdId}; ` +
          'the backend stays stopped until the hold ends, the user quits, or the user confirms Start anyway'
      )
    }

    current = state

    if (!bootPhase) {
      host.updateBootProgress({ updateHold: wire(state) })

      return
    }

    host.updateBootProgress({
      phase: 'backend.update-held',
      // Logged by updateBootProgress: only when what holds the install changes,
      // not on every re-check.
      ...(sameHold ? {} : { message: heldWaitMessage(state) }),
      progress: 12,
      running: true,
      error: null,
      updateHold: wire(state)
    })
  }

  return {
    current: () => current,
    clear(owner = PRIMARY_HOLD_OWNER) {
      board.clear(owner)
      const shown = board.shown()

      if (shown) {
        render(shown, false)

        return
      }

      if (!current && !host.bootHold()) {
        return
      }

      current = null
      host.updateBootProgress({ updateHold: null })
    },
    // A pool/profile wait publishes only the hold (its boot is not the window's).
    show(state: HeldState, owner = PRIMARY_HOLD_OWNER) {
      board.set(owner, state)
      render(board.shown()!, owner === PRIMARY_HOLD_OWNER)
    }
  }
}

export interface UpdateHoldIpcHost {
  /** Only the primary window's boot surface may drive the hold. */
  isPrimaryBootSender: (event: IpcMainInvokeEvent) => boolean
  currentHold: () => HeldState | null
  log: (line: string) => void
  flushLog: () => void
  quit: () => void
}

// The blocked boot screen's three ways out (R8 D3). Only the primary window's
// boot surface can drive them, and only for the hold it is showing.
export function registerUpdateHoldIpc(ipc: IpcMain, host: UpdateHoldIpcHost) {
  ipc.handle('hermes:update-hold:recheck', async event => {
    const hold = host.currentHold()

    if (!host.isPrimaryBootSender(event) || !hold) {
      return { ok: false }
    }

    host.log(`[updates] boot blocked (hold ${hold.holdId}): user asked to check again`)
    requestHoldRecheck()

    return { ok: true }
  })

  ipc.handle('hermes:update-hold:quit', async event => {
    if (!host.isPrimaryBootSender(event)) {
      return { ok: false }
    }

    const hold = host.currentHold()
    host.log(`[updates] user quit Hermes from the update-hold screen${hold ? ` (hold ${hold.holdId})` : ''}`)
    host.quit()

    return { ok: true }
  })

  ipc.handle('hermes:update-hold:start-anyway', async (event, request: { holdId?: unknown; confirmed?: unknown }) => {
    const hold = host.currentHold()

    if (!host.isPrimaryBootSender(event) || !hold || request?.confirmed !== true || request.holdId !== hold.holdId) {
      host.log(
        `[updates] Start anyway refused: ${hold ? `hold ${hold.holdId}` : 'no hold'} is not the confirmed hold ` +
          `(${typeof request?.holdId === 'string' ? request.holdId.slice(0, 32) : 'none'})`
      )

      return { ok: false }
    }

    host.log(startAnywayLogLine(hold))
    // The override must survive whatever the backend start does next.
    host.flushLog()
    allowStartOverHold(hold.holdId)

    return { ok: true }
  })
}
