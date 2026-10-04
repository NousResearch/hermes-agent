// Desktop wiring of an update hold (R8 D3): the marker probe the boot and
// pool-backend gates share, the blocked boot screen's state, and its three IPC
// ways out. The judgement itself lives in update-marker-gate.ts; main.ts keeps
// only the call sites.

import path from 'node:path'

import type { IpcMain, IpcMainInvokeEvent } from 'electron'

import { type UpdateGateDeps, waitForUpdateClearance, type WaitForUpdateClearanceOptions } from './update-gate'
import type { UpdateHoldWire } from './update-hold-types'
import { cachedCreateTimeProbe } from './update-marker'
import {
  allowStartOverHold,
  type HeldState,
  heldWaitMessage,
  holdTicker,
  liveMarkerProbe,
  PRIMARY_HOLD_OWNER,
  requestHoldRecheck,
  startAnywayLogLine,
  UpdateHoldBoard
} from './update-marker-gate'
import { runMarkerHelper } from './updater/marker-helper'

export interface MarkerGateCallbacks {
  onLiveMarker?: (marker: { startedAt: number | null; runId: string | null }) => void
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
  /** main.ts's boot progress snapshot (`hermes:boot-progress:get`). */
  bootProgress: () => { updateHold?: UpdateHoldWire | null }
  currentHold: () => HeldState | null
  log: (line: string) => void
  flushLog: () => void
  quit: () => void
}

// The blocked boot screen's three ways out (R8 D3). Only the primary window's
// boot surface can drive them, and only for the hold it is showing.
export function registerUpdateHoldIpc(ipc: IpcMain, host: UpdateHoldIpcHost) {
  // The boot snapshot every window pulls on mount. Only the primary window
  // gets the hold: boot-progress pushes (the hold clearing included) reach the
  // main window alone and the ways out refuse every other sender, so a HUD or
  // session window that mounted the blocked screen would keep it forever.
  ipc.handle('hermes:boot-progress:get', async event => {
    const state = host.bootProgress()

    return host.isPrimaryBootSender(event) ? state : { ...state, updateHold: null }
  })

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

export interface PoolUpdateWaitHost {
  profile: string
  poolKey: string
  /** main.ts's `updateGateDeps`: the gate deps shared with the primary boot wait. */
  gateDeps: (callbacks: MarkerGateCallbacks) => UpdateGateDeps
  signal: WaitForUpdateClearanceOptions['signal']
  isCancelled: () => boolean
  log: (line: string) => void
  showHold: (state: HeldState, owner: string) => void
  clearHold: (owner: string) => void
  pollMs: number
  timeoutMs: number
}

/**
 * A pool/profile backend's update wait. No boot-progress UI (pool backends
 * boot silently for background profiles), so it only logs while parked. A
 * blocking hold never ages out here either (R8 D3). Past the grace it is
 * published on the window's blocked screen like the primary's (R8 M6): a
 * remote primary, or one that booted before the hold appeared, never shows
 * one of its own. Check again / Start anyway act on its hold id.
 */
export async function waitForPoolUpdateClearance(host: PoolUpdateWaitHost): Promise<void> {
  const { profile } = host
  let poolAnnounced = false
  let poolHoldLogged: string | null = null
  const holdOwner = `pool:${host.poolKey}`

  const hold = holdTicker({
    show: state => host.showHold(state, holdOwner),
    clear: () => host.clearHold(holdOwner)
  })

  const poolGateDeps = host.gateDeps({
    onHeld: state => {
      hold.onHeld(state)

      if (state.blocking && poolHoldLogged !== state.holdId) {
        poolHoldLogged = state.holdId
        host.log(
          `[updates] pool backend start for profile "${profile}" blocked: update marker ${state.verdict}, ` +
            `hold ${state.holdId}; waiting for the hold to end or a Start anyway on the blocked screen`
        )
      }
    }
  })

  try {
    await waitForUpdateClearance(poolGateDeps, {
      signal: host.signal,
      isCancelled: host.isCancelled,
      onWaitTick: reason => {
        if (!poolAnnounced) {
          poolAnnounced = true
          host.log(`[updates] update in progress (${reason}); deferring pool backend start for profile "${profile}"`)
        }

        hold.tick(reason)
      },
      pollMs: host.pollMs,
      timeoutMs: host.timeoutMs
    })
  } finally {
    host.clearHold(holdOwner)
  }
}
