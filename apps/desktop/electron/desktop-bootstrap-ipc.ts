import { decideBootstrapRepair } from './bootstrap-repair-guard'

interface DesktopBootstrapIpcDeps {
  ipcMain: any
  recycleOwnedBackend: (...args: any[]) => Promise<any>
  sendConnectionApplied: () => void
  primaryProfileKey: () => string
  teardownPoolBackendAndWait: (profile: string) => Promise<any>
  teardownPrimaryBackendAndWait: (options?: { soft?: boolean }) => Promise<any>
  teardownSshConnection: (profile: string | null) => Promise<any>
  rememberLog: (message: string) => void
  clearFailures: () => void
  firstRunBoot: any
  incrementRepairAttempt: () => number
  maxBootstrapRepairSoftAttempts: number
  getPrimaryBackendProcess: () => any
  setBootstrapRepairRequested: (value: boolean) => void
  resetHermesConnection: () => void
  getBootstrapAbortController: () => AbortController | null
}

export function registerDesktopBootstrapIpc(deps: DesktopBootstrapIpcDeps) {
  const {
    ipcMain,
    recycleOwnedBackend,
    sendConnectionApplied,
    primaryProfileKey,
    teardownPoolBackendAndWait,
    teardownPrimaryBackendAndWait,
    teardownSshConnection,
    rememberLog,
    clearFailures,
    firstRunBoot,
    incrementRepairAttempt,
    maxBootstrapRepairSoftAttempts,
    getPrimaryBackendProcess,
    setBootstrapRepairRequested,
    resetHermesConnection,
    getBootstrapAbortController
  } = deps

  ipcMain.handle('hermes:backend:recycle', async (_event, profile) => {
    // Models-page recovery after a code-skew 503 (#97046): kill the owned
    // SSH serve (if any) before the local child so reconnect cannot reuse a
    // stale lockfile. Soft primary teardown keeps the renderer shell mounted.
    await recycleOwnedBackend({
      notifyApplied: sendConnectionApplied,
      primaryProfile: primaryProfileKey(),
      profile: typeof profile === 'string' ? profile : '',
      teardownPool: teardownPoolBackendAndWait,
      teardownPrimary: () => teardownPrimaryBackendAndWait({ soft: true }),
      teardownSsh: value => teardownSshConnection(value || null)
    })

    return { ok: true }
  })
  ipcMain.handle('hermes:bootstrap:reset', async () => {
    // Renderer's "Reload and retry" path. Clear the latched failure and
    // reset connection state so the next startHermes() call restarts the
    // full backend flow (including a fresh runBootstrap pass).
    rememberLog('[bootstrap] reset requested by renderer; clearing latched failure')
    await teardownPrimaryBackendAndWait()
    clearFailures()
    firstRunBoot.getFirstRunSetupGate().resetForRetry()
    firstRunBoot.resetBootstrapSnapshot()

    return { ok: true }
  })
  ipcMain.handle('hermes:bootstrap:repair', async () => {
    // Forceful repair: force the next startHermes() through the full installer
    // (refreshing a broken/partial venv) and clear any latched failure + live
    // connection. The renderer reloads afterwards to re-drive the boot flow.
    //
    // We do NOT delete the bootstrap marker here. Repair is also reachable from
    // transient backend errors on a perfectly healthy install, and deleting the
    // marker in that case stranded the app in first-run setup with no way back
    // (#72166). The explicit flag carries the intent instead.
    const bootstrapRepairAttempt = incrementRepairAttempt()

    // Probe the live backend process so the guard can distinguish "venv is
    // genuinely broken" (force reinstall) from "backend is just transiently
    // stalled under GIL pressure" (#74874 — `event loop stalled` followed by
    // `ws ready frame send failed`, then renderer keeps reporting dead).
    const primaryProc = getPrimaryBackendProcess()

    const primaryBackendAlive = Boolean(
      primaryProc &&
      (primaryProc as { exitCode?: number | null }).exitCode === null &&
      (primaryProc as { signalCode?: string | null }).signalCode === null
    )

    const repairDecision = decideBootstrapRepair({
      attempt: bootstrapRepairAttempt,
      maxSoftAttempts: maxBootstrapRepairSoftAttempts,
      primaryBackendAlive
    })

    rememberLog(
      `[bootstrap] repair requested by renderer; forcing reinstall + clearing latched failure ` +
        `(attempt=${repairDecision.attempt}/${maxBootstrapRepairSoftAttempts}, ` +
        `primaryBackendAlive=${primaryBackendAlive}, ` +
        `hardReinstall=${repairDecision.hardReinstall}): ${repairDecision.reason}`
    )

    // The guard may decide the install is healthy enough that a restart
    // (without touching the venv) is the right answer. Translate that into
    // the existing flag: if the guard said "soft restart", we skip the
    // "bypass active runtime" path inside startHermes() and fall through
    // to the normal restart branch, which just kills the current child
    // and respawns it against the same venv. See #74874 — this is what
    // breaks the infinite reinstall loop the user hit.
    setBootstrapRepairRequested(repairDecision.hardReinstall)
    clearFailures()
    firstRunBoot.getFirstRunSetupGate().resetForRepair()
    resetHermesConnection()

    return { ok: true }
  })
  ipcMain.handle('hermes:bootstrap:continue-local', async () => {
    rememberLog('[bootstrap] local install selected by renderer; continuing first-launch bootstrap')
    firstRunBoot.continueFirstRunLocalBootstrap()

    return { ok: true }
  })
  ipcMain.handle('hermes:bootstrap:cancel', async () => {
    // Renderer's Cancel button during first-launch install. Abort the running
    // install script (SIGTERM via the runner's abortSignal). runBootstrap
    // resolves with { cancelled: true }, which surfaces the recovery overlay.
    const bootstrapAbortController = getBootstrapAbortController()

    if (bootstrapAbortController) {
      try {
        bootstrapAbortController.abort()
      } catch {
        void 0
      }

      return { ok: true, cancelled: true }
    }

    return { ok: false, cancelled: false }
  })
  ipcMain.handle('hermes:boot-progress:get', async () => firstRunBoot.getBootProgressState())
  ipcMain.handle('hermes:bootstrap:get', async () => firstRunBoot.getBootstrapState())
}
