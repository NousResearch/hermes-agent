// Checkout update policy and handoff execution. The shell supplies process and UI dependencies.

import { existsSync } from 'node:fs'
import * as path from 'node:path'

import { updateHandoffConflict, writeUpdateMarker } from '../update-marker'
import {
  collectRelaunchArgs,
  observeUpdaterHandoff,
  resolveInstallationLauncher,
  resolvePosixScriptHandoff,
  resolveUpdateScriptHandoff,
  sandboxFallbackFromEnv,
  spawnUpdaterProcess,
  stagedUpdaterSupportsPrewrittenMarker,
  windowsUpdatePrerequisiteError,
  wrapHandoffForDetachedConsole
} from '../updater-process'

import { checkCheckoutUpdates, type CheckoutCheckDeps } from './checkout-check'
import { sourceUpdateEnvironment } from './checkout-source'

import type { UpdaterApplyResultWire, UpdaterMechanism, UpdaterStatusWire, UpdaterStrategy } from './index'

/**
 * Everything the checkout flow needs from the app shell. These are the
 * impure edges only — all update logic lives here.
 */
export interface CheckoutStrategyDeps extends CheckoutCheckDeps {
  hermesHome: string
  isWindows: boolean
  isMac: boolean
  defaultUpdateBranch: string
  updateHandoffDwellMs: number
  directoryExists: (filePath: string) => boolean
  resolveUpdaterBinary: () => string | null
  firstLine: (text: string) => string

  emitUpdateProgress: (payload: { stage: string; message: string; percent: number | null }) => void
  rememberLog: (chunk: unknown) => void
  startHermes: () => Promise<unknown>
  stopBackendsForUpdate: () => Promise<void>
  repairMacUpdaterHelper: (updater: string) => void | Promise<void>
  preflightStateDb: (hermesHome: string, rememberLog: (chunk: string) => void) => void
  runningAppBundle: () => string | null
  markQuittingForHandoff: () => void
  quit: () => void
}

/**
 * The manual command card for a checkout with no staged updater: the exact
 * `hermes update` line to run, branch-pinned to the checkout's current branch
 * for non-main (bare `hermes update` would silently switch the install
 * off-branch).
 */
export function buildManualUpdateCommand(currentBranch: string | null | undefined): string {
  return currentBranch && currentBranch !== 'HEAD' && currentBranch !== 'main'
    ? `hermes update --branch ${currentBranch}`
    : 'hermes update'
}

/**
 * The checkout strategy: windows-handoff on win32, posix-handoff elsewhere.
 * The bodies are the production update flow; the mechanism stamp rides on
 * every result the way the wire contract expects.
 */
export function createCheckoutStrategy(deps: CheckoutStrategyDeps): UpdaterStrategy {
  const mechanism: UpdaterMechanism = deps.isWindows ? 'windows-handoff' : 'posix-handoff'

  async function check(opts: { force?: boolean } = {}): Promise<UpdaterStatusWire> {
    const status = await checkCheckoutUpdates(deps, opts)
    status.mechanism = mechanism

    return status
  }

  async function apply(): Promise<UpdaterApplyResultWire> {
    const result: UpdaterApplyResultWire = await applyBody()
    result.mechanism = mechanism

    return result
  }

  return { mechanism, check, apply }

  async function applyBody(): Promise<UpdaterApplyResultWire> {
    const status: UpdaterStatusWire = await checkCheckoutUpdates(deps, { force: true })

    if (status.reason === 'source-probe-unavailable') {
      return { ok: true, manual: true, command: 'hermes update --help', message: status.message, hermesRoot: status.hermesRoot }
    }

    if (!status.supported || status.error) {
      return { ok: false, error: status.error ?? status.reason, message: status.message }
    }

    const branch: string = status.branch ?? deps.defaultUpdateBranch
    const targetArgs: string[] = status.channel ? ['--channel', status.channel] : ['--branch', branch]
    const targetLabel: string = status.channel ?? branch
    const manualCommand: string = status.channel ? `hermes update --channel ${status.channel}` : buildManualUpdateCommand(branch)
    const updater: string | null = deps.resolveUpdaterBinary()
    const root: string = deps.resolveUpdateRoot()

    // Earlier PM scripts still demand checkout/venv. Do not invoke that known
    // incompatible handoff: one exact-install CLI update obtains the new scripts.
    if (existsSync(path.join(root, 'pm')) && !existsSync(path.join(root, 'scripts', 'desktop-update', 'runtime.ps1'))) {
      const launcher: string | null = resolveInstallationLauncher(root, deps.isWindows, deps.hermesHome)

      if (!launcher) { return { ok: false, error: 'installation-launcher-missing' } }

      const quote = (value: string): string => deps.isWindows
        ? `'${value.replace(/'/g, "''")}'`
        : `'${value.replace(/'/g, "'\\''")}'`

      const command: string = `${deps.isWindows ? '& ' : ''}${quote(launcher)} update ${targetArgs.map(quote).join(' ')}`

      return { ok: true, manual: true, command, hermesRoot: root }
    }

    if (!deps.isWindows && (!updater || status.channel)) {
      // macOS/Linux: hand off to the repo-owned posix script — same shape as
      // Windows (quit → detached orchestrator → `hermes update` → relaunch),
      // minus the venv-lock gauntlet POSIX doesn't need. The old in-app
      // updater (applyUpdatesPosixInApp) is gone with everything it dragged
      // in: the HERMES_DESKTOP_CHILD_PID reaper-exclusion dance (#37532),
      // the in-window rebuild retry, and the relaunch-outcome matrix — the
      // script owns swap/relaunch, and the app is DEAD during the update so
      // there is nothing to reap around. Checkouts that predate the script
      // get the manual `hermes update` card once; their next update pulls it.
      return await applyPosixHandoff(targetArgs, targetLabel, manualCommand)
    }

    if (!updater || status.channel) {
      // No staged updater binary — this is a CLI-installed user (they ran
      // `hermes desktop`, never the Tauri installer that self-copies
      // hermes-setup.exe into HERMES_HOME). On Windows the repo hand-off
      // script serves them just as well as installer users — it only needs
      // PowerShell and the checkout — so fall through to the normal hand-off
      // when the script exists. Only when the checkout predates the script do
      // we surface the manual one-liner.
      const updateRoot = deps.resolveUpdateRoot()

      if (!resolveUpdateScriptHandoff(updateRoot)) {
        const command: string = manualCommand

        deps.rememberLog(`[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot}`)
        deps.emitUpdateProgress({ stage: 'manual', message: command, percent: null })

        return { ok: true, manual: true, command, hermesRoot: updateRoot }
      }

      deps.rememberLog('[updates] no staged updater; using repo hand-off script for CLI install')
    }

    const handoffConflict = updateHandoffConflict(deps.hermesHome)

    if (handoffConflict) {
      // A different updater already owns the marker — most often a previous
      // "Update" click whose updater is still alive and parked mid-run.
      // Spawning another here would overwrite its claim and let two updaters
      // mutate the checkout at once (#75778); refuse instead.
      deps.rememberLog(`[updates] refusing hand-off: ${handoffConflict.message}`)
      deps.emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

      return { ok: false, error: 'update-already-running', message: handoffConflict.message }
    }

    deps.emitUpdateProgress({
      stage: 'restart',
      message:
        'Updating Hermes — this window will close and the updater will open. Don’t reopen Hermes yourself; it restarts automatically when the update finishes.',
      percent: 100
    })
    deps.repairMacUpdaterHelper(updater)

    const updateRoot = deps.resolveUpdateRoot()
    const updaterArgs: string[] = ['--update', ...targetArgs]
    const targetApp = deps.isMac ? deps.runningAppBundle() : null

    if (targetApp) {
      updaterArgs.push('--target-app', targetApp)
    }


    // ── Pre-flight state.db integrity guard (#68474) ─────────────────
    // Emergency backup and header verification before the update touches
    // anything.  Runs while the backend is still alive.
    deps.preflightStateDb(deps.hermesHome, deps.rememberLog)

    if (deps.isWindows && resolveUpdateScriptHandoff(updateRoot)) {
      const message = windowsUpdatePrerequisiteError(updateRoot, deps.hermesHome)

      if (message) {
        deps.emitUpdateProgress({ stage: 'error', message, percent: null })

        return { ok: false, error: message }
      }
    }

    // Release app-owned backends for output replacement. PM publishes a new
    // dependency generation; old Python readers are not update blockers.
    // The CLI owns gateway draining and restart, including failure recovery.
    await deps.stopBackendsForUpdate()

    // Detached so app replacement can outlive this process.
    //
    // Prefer the repo-owned hand-off script over the staged Tauri binary.
    // The staged binary is frozen (no self-update path) and historically runs
    // months-stale updater logic — pre-#67369 cache resolver, pre-#74782
    // marker adoption — producing failures that were fixed on main long ago
    // (2026-08-09 incident). scripts/desktop-update/windows.ps1 ships WITH the
    // checkout, so each `hermes update` refreshes the code that drives the
    // next one. Checkouts that predate the script fall back to the binary
    // path unchanged.
    const scriptHandoff = resolveUpdateScriptHandoff(updateRoot)
    let child

    if (scriptHandoff) {
      const updateStartedAt = Math.floor(Date.now() / 1000)

      // A bare detached+hidden powershell spawn silently dies before -File
      // processing (console-subsystem init failure — see
      // wrapHandoffForDetachedConsole). Route through `cmd start` so the
      // script gets its own minimized console and survives our exit. The
      // wrapper cmd.exe exits immediately, so child.pid is NOT the script's
      // pid — the script claims the update marker itself with its own $PID
      // as its first action, and a relaunched Desktop parks on that.
      const wrapped = wrapHandoffForDetachedConsole(scriptHandoff, [
        '-InstallRoot',
        updateRoot,
        ...(status.channel ? ['-Channel', status.channel] : ['-Branch', branch]),
        '-DesktopPid',
        String(process.pid),
        '-RelaunchExe',
        process.execPath
      ])

      child = spawnUpdaterProcess(wrapped.command, wrapped.args, {
        cwd: deps.hermesHome,
        env: {
          ...sourceUpdateEnvironment(updateRoot, deps.hermesHome),
          HERMES_UPDATE_STARTED_AT: String(updateStartedAt)
        },
        detached: true,
        stdio: 'ignore'
      })

      // Bridge marker: child.pid is the short-lived cmd.exe WRAPPER, not the
      // script (see wrapHandoffForDetachedConsole). Write it anyway to cover
      // the first moments of the hand-off — the script's step 0 overwrites it
      // with its own live $PID, and if the script never starts the wrapper's
      // dead pid makes the marker read as stale and self-delete (no wedge).
      // The `hermes update` child adopts the SCRIPT's claim via
      // update_lock.py's process-ancestry rule; no mtime heuristics needed.
      if (Number.isInteger(child.pid)) {
        writeUpdateMarker(deps.hermesHome, child.pid, { startedAt: updateStartedAt })
      }

      deps.rememberLog(
        `[updates] launched repo hand-off script: ${scriptHandoff.scriptPath} (${targetLabel}); exiting desktop for application replacement`
      )
    } else {
      child = spawnUpdaterProcess(updater, updaterArgs, {
        cwd: deps.hermesHome,
        env: {
          ...sourceUpdateEnvironment(updateRoot, deps.hermesHome)
        },
        detached: true,
        stdio: 'ignore'
      })

      // Write the update-in-progress marker IMMEDIATELY — before the 2.5s
      // quit dwell. The Tauri updater won't write its own marker for several
      // seconds (window init + manifest), and during that gap our renderer
      // can reconnect into an update still replacing application files.
      // By writing the marker ourselves the renderer's
      // waitForUpdateToFinish() gate sees a live update and parks instead.
      // The updater overwrites this with its own PID later; same format.
      //
      // SKIPPED for pre-#74782 staged updaters: those have no self-PID
      // exclusion, so they read this very marker as a foreign live owner and
      // abort with "Another Hermes update is already running (PID <itself>)" —
      // an unbreakable loop, because the update that would replace the stale
      // binary is the one being refused. Losing the anti-respawn hardening is
      // strictly better than never updating again, and the updater still writes
      // its own marker moments later.
      if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
        writeUpdateMarker(deps.hermesHome, child.pid)
      } else if (Number.isInteger(child.pid)) {
        deps.rememberLog(
          `[updates] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
        )
      }

      deps.rememberLog(
        `[updates] launched updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop for application replacement`
      )
    }

    // Linger on the "updating — don't reopen" overlay long enough for the user
    // to actually read it (and to bridge the gap until the updater's own window
    // appears), THEN quit for application replacement. The updater rebuilds and
    // relaunches us when it's done. (#50419 — a 600ms quit looked like a crash
    // and lured users into the #50238 relaunch loop.)
    //
    // The dwell doubles as the hand-off settle window (#66753): watch the
    // detached child for an async spawn `error` (ENOENT/EACCES) or an early
    // non-zero/signal exit. On failure, DON'T quit — the user would be left
    // with no app, no updater, and no evidence. Restart our backend and
    // surface the error instead. The pre-written marker names the dead child
    // pid, so readLiveUpdateMarker self-heals it; no cleanup needed.
    const dwellStartedAt = Date.now()
    const handoffOutcome = await observeUpdaterHandoff(child, deps.updateHandoffDwellMs)

    if (!handoffOutcome.ok) {
      const message = `Update failed to start: ${handoffOutcome.message}. Hermes will keep running — try again, or run \`hermes update\` from a terminal.`

      deps.rememberLog(`[updates] hand-off not viable, aborting quit: ${handoffOutcome.message}`)
      deps.emitUpdateProgress({ stage: 'error', message, percent: null })
      deps.startHermes().catch(() => {})


      return { ok: false, error: 'updater-spawn-failed', message }
    }

    deps.markQuittingForHandoff()
    setTimeout(
      () => {
        deps.quit()
      },
      Math.max(0, deps.updateHandoffDwellMs - (Date.now() - dwellStartedAt))
    )

    return { ok: true, handedOff: true, updater }
  }

  async function applyPosixHandoff(targetArgs: string[], targetLabel: string, manualCommand: string): Promise<UpdaterApplyResultWire> {
  const updateRoot = deps.resolveUpdateRoot()
  const handoff = resolvePosixScriptHandoff(updateRoot)

  if (!handoff) {
    deps.emitUpdateProgress({ stage: 'manual', message: manualCommand, percent: null })

    return { ok: true, manual: true, command: manualCommand, hermesRoot: updateRoot }
  }

  const handoffConflict = updateHandoffConflict(deps.hermesHome)

  if (handoffConflict) {
    // Same hazard as the Windows path (#75778): a live foreign updater
    // already owns the marker — refuse rather than double-mutate the tree.
    deps.rememberLog(`[updates] refusing posix hand-off: ${handoffConflict.message}`)
    deps.emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

    return { ok: false, error: 'update-already-running', message: handoffConflict.message }
  }

  // ── Pre-flight state.db integrity guard (#68474) ──
  deps.preflightStateDb(deps.hermesHome, deps.rememberLog)

  const args: string[] = [...handoff.args, '--install-root', updateRoot, ...targetArgs, '--desktop-pid', String(process.pid)]
  const updateStartedAt = Math.floor(Date.now() / 1000)

  // Relaunch target: the running .app bundle on mac (script swaps the
  // rebuilt bundle over it), the running binary elsewhere. The script's gate
  // (an exact port of update-relaunch.ts's decideRelaunchOutcome) relaunches
  // only a binary the rebuild replaced with a launchable sandbox helper —
  // replaying the original launch context (filtered args, cwd, sandbox
  // opt-out) so a deep-link or --no-sandbox launch survives the update.
  const targetApp = deps.isMac ? deps.runningAppBundle() : process.execPath

  if (targetApp) {
    args.push('--relaunch-target', targetApp)
  }

  const relaunchArgs = collectRelaunchArgs(process.argv.slice(1))

  if (!deps.isMac) {
    args.push('--relaunch-cwd', process.cwd())

    if (sandboxFallbackFromEnv(process.env, relaunchArgs)) {
      args.push('--sandbox-fallback')
    }

    if (relaunchArgs.length) {
      args.push('--', ...relaunchArgs)
    }
  }

  const child = spawnUpdaterProcess(handoff.command, args, {
    cwd: deps.hermesHome,
    env: {
      ...sourceUpdateEnvironment(updateRoot, deps.hermesHome),
      HERMES_UPDATE_STARTED_AT: String(updateStartedAt)
    },
    detached: true,
    stdio: 'ignore'
  })

  // Bridge marker (same contract as the Windows hand-off): cover the gap
  // until the script claims the marker with its own pid as step 0. If the
  // script never starts, the dead pid reads as stale and self-deletes.
  if (Number.isInteger(child.pid)) {
    writeUpdateMarker(deps.hermesHome, child.pid, { startedAt: updateStartedAt })
  }

  deps.rememberLog(`[updates] launched posix hand-off: ${handoff.scriptPath} (${targetLabel}); quitting to hand off`)
  deps.emitUpdateProgress({
    stage: 'restart',
    message:
      'Updating Hermes — this window will close. Don’t reopen Hermes yourself; it restarts automatically when the update finishes.',
    percent: 100
  })

  // Settle window (#66753): the reported macOS failure mode is exactly this
  // path — the app quits, bash/posix.sh dies early (or was never spawnable),
  // and the user is left with no app, no updater, and no relaunch. Watch the
  // child through the dwell; on spawn error or early death, stay alive and
  // surface the failure instead of quitting into nothing.
  const dwellStartedAt = Date.now()
  const handoffOutcome = await observeUpdaterHandoff(child, deps.updateHandoffDwellMs)

  if (!handoffOutcome.ok) {
    const message = `Update failed to start: ${handoffOutcome.message}. Hermes will keep running — try again, or run \`hermes update\` from a terminal.`

    deps.rememberLog(`[updates] posix hand-off not viable, aborting quit: ${handoffOutcome.message}`)
    deps.emitUpdateProgress({ stage: 'error', message, percent: null })

    return { ok: false, error: 'updater-spawn-failed', message }
  }

  deps.markQuittingForHandoff()
  setTimeout(
    () => {
      deps.quit()
    },
    Math.max(0, deps.updateHandoffDwellMs - (Date.now() - dwellStartedAt))
  )

  return { ok: true, handedOff: true, updater: handoff.scriptPath }
  }
}
