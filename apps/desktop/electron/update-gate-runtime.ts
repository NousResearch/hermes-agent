import path from 'node:path'

import type { UpdateClearanceOutcome, UpdateGateDeps, WaitForUpdateClearanceOptions } from './update-gate'

export const UPDATE_WAIT_TIMEOUT_MS = 20 * 60 * 1000
export const UPDATE_WAIT_POLL_MS = 1000

export interface UpdateGateRuntimeDeps {
  hermesHome: string
  isPackaged: boolean
  installStamp: any
  loadInstallStamp: () => any
  getUpdateInFlight: () => boolean
  getHandoffActive: () => boolean
  readLiveUpdateMarker: (home: string) => unknown
  readAndConsumeHandoffResult: (home: string) => any
  waitForUpdateClearance: (
    gate: UpdateGateDeps,
    options: WaitForUpdateClearanceOptions
  ) => Promise<UpdateClearanceOutcome>
  detectBundleSwap: (running: any, installed: any) => boolean
  buildNoSandboxRelaunchArgs: (args: string[]) => string[]
  app: { relaunch: (options: { args: string[] }) => void }
  dialog: { showMessageBox: (options: any) => any }
  shell: { showItemInFolder: (filePath: string) => void }
  localBackendLifecycle: { signal: AbortSignal; assertCanStart: () => void }
  firstRunBoot: { advanceBootProgress: (phase: string, message: string, progress: number) => Promise<unknown> }
  rememberLog: (message: string) => void
  sendOpenUpdatesRequested: () => void
  exitAfterBackendShutdown: (code: number) => unknown
}

export function createUpdateGateRuntime(deps: UpdateGateRuntimeDeps) {
  const HERMES_HOME = deps.hermesHome
  const IS_PACKAGED = deps.isPackaged
  const INSTALL_STAMP = deps.installStamp

  const {
    app,
    buildNoSandboxRelaunchArgs,
    detectBundleSwap,
    dialog,
    exitAfterBackendShutdown,
    firstRunBoot,
    getHandoffActive,
    getUpdateInFlight,
    loadInstallStamp,
    localBackendLifecycle,
    readAndConsumeHandoffResult,
    readLiveUpdateMarker,
    rememberLog,
    sendOpenUpdatesRequested,
    shell,
    waitForUpdateClearance
  } = deps
  // --- in-app update mutual exclusion (#50238) -------------------------------
  // The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
  // duration of an `--update` run (see update.rs UpdateMarkerGuard). If the user
  // relaunches the desktop mid-update — because the window vanished with no
  // progress and looks crashed — a fresh instance must NOT spawn its own local
  // backend: that backend re-locks the venv shim, the updater's straggler cleanup
  // (`force_kill_other_hermes`, taskkill /IM hermes.exe) kills it, the launch
  // fails with the 45s "backend didn't come up" error, and the relaunch/kill
  // cycle loops. Instead the fresh instance parks until the update finishes, then
  // brings the backend up itself (it is the surviving instance — the updater's
  // own relaunch hits our single-instance lock and quits). Marker parsing +
  // staleness self-heal live in update-marker.ts (unit-tested).

  // How long we'll park the launch waiting for a live update to finish before
  // giving up and starting the backend anyway (belt-and-suspenders alongside the
  // marker's own age ceiling; covers a stuck-but-alive updater).
  // Gate deps shared by the primary-window boot path and the pool-backend
  // spawn path. Consulting the on-disk marker, the in-process updateInFlight
  // flag, AND the successful detached hand-off state is load-bearing (#73822):
  // applyUpdates kills its own backend BEFORE the Windows venv-blocker scan but
  // only writes the marker AFTER it, so a marker-only gate lets the renderer's
  // ~1s reconnect respawn a backend inside the update's own critical section —
  // which the scan then reports as a blocker, aborting every update attempt.
  // The hand-off state closes the later Windows `cmd start` wrapper gap: the
  // wrapper exits 0 before the real PowerShell script claims the marker, and
  // `finally` clears updateInFlight immediately after the hand-off is accepted.
  function updateGateDeps() {
    return {
      hasLiveMarker: () => Boolean(readLiveUpdateMarker(HERMES_HOME)),
      isUpdateInFlight: getUpdateInFlight,
      isHandoffActive: getHandoffActive
    }
  }

  // One-shot guard for the automatic bundle-swap relaunch below: the relaunched
  // instance carries this flag so a stamp that still mismatches (unreadable
  // resources, exotic packaging) can never produce a relaunch loop.
  const BUNDLE_SWAP_RELAUNCH_FLAG = '--hermes-bundle-swap-relaunched'

  // How long the parked instance waits for its own scheduled exit to land before
  // giving up and booting the stale build anyway. Better a torn renderer with a
  // banner than a window that never comes back.
  const BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS = 15_000

  // The detached updater swaps the packaged bundle on disk AFTER `hermes update`
  // exits (posix.sh mac_swap / windows.ps1). An instance reopened mid-update —
  // the #50238 gesture the gate above exists for — was launched from the
  // PRE-swap bundle, and the updater's `open` leg then merely focuses us (single
  // instance), so no process ever loads the new build. Letting boot proceed here
  // runs the new runtime under the old renderer: exactly the skew
  // detectRendererSkew() warns about, except the Updates card already says
  // "latest", so the warning's own remedy has nothing to run.
  //
  // This is the earliest point where the swap is PROVABLE — it happens while we
  // are parked on the gate, so checking any sooner (at `ready`, before the gate)
  // only ever compares a stamp with itself. Relaunching here also keeps the
  // boot-progress window up for the whole wait instead of leaving the user with
  // no window at all.
  //
  // Returns true when the relaunch was scheduled; the caller must park rather
  // than continue booting, because the process exits underneath it.
  function relaunchIntoSwappedBundle() {
    if (!IS_PACKAGED || process.argv.includes(BUNDLE_SWAP_RELAUNCH_FLAG)) {
      return false
    }

    if (!detectBundleSwap(INSTALL_STAMP, loadInstallStamp())) {
      return false
    }

    rememberLog('[updates] app bundle was swapped during the update; relaunching into the new build')

    try {
      app.relaunch({
        args: [...buildNoSandboxRelaunchArgs(process.argv.slice(1)), BUNDLE_SWAP_RELAUNCH_FLAG]
      })
    } catch (err) {
      rememberLog(`[updates] bundle-swap relaunch failed: ${err?.message || err}; continuing with the current build`)

      return false
    }

    void exitAfterBackendShutdown(0)

    return true
  }

  // Block until no live update is in progress (or we hit the wait timeout).
  // Emits a boot-progress phase so the renderer shows "Update in progress…"
  // rather than a frozen splash. Returns true if it parked at all.
  async function waitForUpdateToFinish() {
    let announced = false

    const outcome = await waitForUpdateClearance(updateGateDeps(), {
      signal: localBackendLifecycle.signal,
      onWaitTick: async reason => {
        if (!announced) {
          announced = true
          rememberLog(`[updates] update in progress (${reason}); deferring backend start until it finishes`)
        }

        await firstRunBoot.advanceBootProgress(
          'backend.update-wait',
          'An update is finishing — Hermes will start automatically when it completes…',
          12
        )
      },
      pollMs: UPDATE_WAIT_POLL_MS,
      timeoutMs: UPDATE_WAIT_TIMEOUT_MS
    })

    // The detached hand-off script (scripts/desktop-update/windows.ps1) runs hidden;
    // its result file is the ONLY way the user learns a detached update
    // failed. Consume it exactly once, here, right where boot passes the
    // update gate — success gets a log line, failure gets a real dialog
    // (previously a failed detached update was indistinguishable from
    // "nothing happened").
    try {
      const result = readAndConsumeHandoffResult(HERMES_HOME)

      if (result && result.ok && result.manual) {
        // Update landed but the user must act (reopen/reinstall/sandbox). On
        // machines with no shim browser and no notifier this dialog is the
        // FIRST time the message is visible — it must not be a log line.
        rememberLog(
          `[updates] detached update finished with manual action (branch ${result.branch}): ${result.message}`
        )
        dialog.showMessageBox({
          type: 'warning',
          title: 'Hermes update',
          message: 'The update finished, but needs one more step',
          detail: result.message
        })
      } else if (result && result.ok) {
        rememberLog(`[updates] detached update finished OK (branch ${result.branch})`)
      } else if (result) {
        rememberLog(`[updates] detached update FAILED (exit ${result.exitCode}): ${result.message}`)
        const handoffLogPath = path.join(HERMES_HOME, 'logs', 'desktop-update-handoff.log')

        // Async so boot is not blocked behind the dialog; the response handlers
        // reuse the menu's open-updates path (queued until the renderer is ready)
        // and the same reveal primitive as 'hermes:logs:reveal'.
        void dialog
          .showMessageBox({
            type: 'error',
            title: 'Hermes update',
            message: "Hermes couldn't finish updating",
            detail:
              "You're still on the previous version and can keep using it. Try the update again, or open the update log to report the problem.\n\n" +
              `Details: ${result.message}`,
            buttons: ['Try again', 'Open log', 'Close'],
            defaultId: 0,
            cancelId: 2,
            noLink: true
          })
          .then(({ response }) => {
            if (response === 0) {
              sendOpenUpdatesRequested()
            } else if (response === 1) {
              shell.showItemInFolder(handoffLogPath)
            }
          })
      }
    } catch (err) {
      rememberLog(`[updates] could not read hand-off result: ${err.message}`)
    }

    if (outcome === 'cancelled') {
      localBackendLifecycle.assertCanStart()
    }

    if (outcome === 'clear') {
      return false
    }

    if (outcome === 'timeout') {
      rememberLog('[updates] update still in progress after wait timeout; starting backend anyway')
    } else if (relaunchIntoSwappedBundle()) {
      await firstRunBoot.advanceBootProgress('backend.update-restart', 'Restarting Hermes to load the updated app…', 14)
      // Park while the scheduled exit lands so this stale build never starts a
      // backend; the failsafe below only runs if the exit somehow does not.
      await new Promise(resolve => setTimeout(resolve, BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS))
      rememberLog(
        `[updates] relaunch did not land within ${BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS}ms; continuing with the current build`
      )
    } else {
      rememberLog('[updates] update finished; proceeding with backend start')
    }

    return true
  }

  return { updateGateDeps, waitForUpdateToFinish }
}
