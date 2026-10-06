// Checkout update policy and handoff execution. The shell supplies process and UI dependencies.

import { existsSync, readFileSync } from 'node:fs'
import * as path from 'node:path'

import {
  claimBridgeMarker,
  describeSkippedPrewrite,
  HANDOFF_CLAIM_TIMEOUT_MS,
  makeHandoffRunId,
  updateHandoffConflict,
  waitForHandoffClaim,
  writeUpdateMarker
} from '../update-marker'
import {
  collectRelaunchArgs,
  describeUpdaterHandoffFailure,
  killHandoffTree,
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

import { type SourceUpdate, sourceUpdateEnvironment } from './checkout-source'
import { type MarkerHelperOp, type MarkerHelperVerdict, readHandoffProtocol, runMarkerHelper } from './marker-helper'

import type { UpdaterApplyResultWire, UpdaterMechanism, UpdaterStatusWire, UpdaterStrategy } from './index'

/**
 * Everything the checkout flow needs from the app shell. These are the
 * impure edges only — all update logic lives here.
 */
export interface CheckoutStrategyDeps {
  resolveUpdateRoot: () => string
  readSourceUpdate: (root: string, opts: { force?: boolean }) => Promise<SourceUpdate | null>
  hermesHome: string
  isWindows: boolean
  isMac: boolean
  defaultUpdateBranch: string
  updateHandoffDwellMs: number
  /** How long the hand-off script has to take the bridge marker (C2); default 20 s. */
  handoffClaimTimeoutMs?: number
  resolveUpdaterBinary: () => string | null
  /**
   * True when one remote gateway serves this Desktop (app-global remote /
   * cloud / SSH). The hand-off then tells `hermes update` not to (re)start a
   * local messaging gateway: with the same channel credentials as the remote
   * host it would become a competing long-poll consumer (#117529).
   */
  remoteGatewayActive: () => boolean

  emitUpdateProgress: (payload: { stage: string; message: string; percent: number | null }) => void
  rememberLog: (chunk: unknown) => void
  startHermes: () => Promise<unknown>
  stopBackendsForUpdate: () => Promise<void>
  repairMacUpdaterHelper: (updater: string) => void | Promise<void>
  preflightStateDb: (hermesHome: string, rememberLog: (chunk: string) => void) => void | Promise<void>
  runningAppBundle: () => string | null
  markQuittingForHandoff: () => void
  quit: () => void
  /** The checkout script's marker helper (A7 rule 3); injectable for tests. */
  markerHelper?: typeof runMarkerHelper
}

/**
 * How one script hand-off is correlated (SPEC 4a/4b). Protocol 2 scripts adopt
 * the Desktop's bridge carrying `runId`; legacy scripts (no protocol line)
 * claim for themselves and echo `startedAt` on marker line 2.
 */
export interface HandoffPlan {
  protocol: number
  runId: string | null
  startedAt: number
}

const STILL_RUNNING_VERDICTS = new Set(['held', 'busy', 'live'])

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
 * The commit this checkout was installed at, from the install stamp the
 * updater already trusts (written by the CLI's completion tail next to the
 * checkout it attests). Read-only display data for `currentSha`; never a
 * git spawn and never an update input. Null when absent or malformed.
 */
export function readStampedCommit(root: string): string | null {
  try {
    const stamp = JSON.parse(readFileSync(path.join(root, 'install-stamp.json'), 'utf8')) as { commit?: unknown }

    return typeof stamp.commit === 'string' && stamp.commit ? stamp.commit : null
  } catch {
    return null
  }
}

/**
 * The checkout strategy: windows-handoff on win32, posix-handoff elsewhere.
 * The bodies are the production update flow; the mechanism stamp rides on
 * every result the way the wire contract expects.
 */
export function createCheckoutStrategy(deps: CheckoutStrategyDeps): UpdaterStrategy {
  function markerHelper(op: MarkerHelperOp, runId: string | null): Promise<MarkerHelperVerdict> {
    return (deps.markerHelper ?? runMarkerHelper)(op, {
      updateRoot: deps.resolveUpdateRoot(),
      hermesHome: deps.hermesHome,
      desktopPid: process.pid,
      runId,
      isWindows: deps.isWindows
    })
  }

  function refuse(message: string, error: string): { refusal: UpdaterApplyResultWire } {
    deps.rememberLog(`[updates] refusing hand-off: ${message}`)
    deps.emitUpdateProgress({ stage: 'error', message, percent: null })

    return { refusal: { ok: false, error, message } }
  }

  /**
   * Plan the hand-off for this script (SPEC 4). Protocol 2: exclusive-create
   * the bridge in this Desktop's own name with a fresh run id, routing an
   * existing dead marker (or a stale bridge of ours) through the script
   * helper, then retrying the create ONCE. Legacy scripts get no bridge at all
   * (they overwrite unconditionally and know no run id). Never deletes or
   * rewrites the marker itself (A7 rule 3).
   */
  async function planHandoff(
    scriptPath: string,
    updateStartedAt: number
  ): Promise<{ plan: HandoffPlan } | { refusal: UpdaterApplyResultWire }> {
    const protocol = readHandoffProtocol(scriptPath) ?? 1

    if (protocol < 2) {
      deps.rememberLog('[updates] hand-off script predates protocol 2; legacy acknowledgment (live owner + started_at)')

      return { plan: { protocol, runId: null, startedAt: updateStartedAt } }
    }

    const runId = makeHandoffRunId()

    for (let attempt = 0; attempt < 2; attempt++) {
      const bridge = await claimBridgeMarker(deps.hermesHome, { startedAt: updateStartedAt, runId })

      if (bridge.ok) {
        return { plan: { protocol, runId, startedAt: updateStartedAt } }
      }

      if (bridge.conflict) {
        return refuse(bridge.conflict.message, 'update-already-running')
      }

      if (attempt > 0) {
        break
      }

      if (bridge.existing) {
        const op: MarkerHelperOp = bridge.existing.state === 'ours' ? 'withdraw' : 'reclaim'
        const verdict = await markerHelper(op, op === 'withdraw' ? bridge.existing.run : null)
        deps.rememberLog(`[updates] ${bridge.existing.state} update marker: script helper ${op} => ${verdict.kind}`)

        if (STILL_RUNNING_VERDICTS.has(verdict.kind)) {
          return refuse(
            'Another update still holds this installation. Wait for it to finish, then try again.',
            'update-already-running'
          )
        }
      } else {
        deps.rememberLog(`[updates] could not write the bridge marker (${bridge.error}); retrying once`)
      }
    }

    return refuse(
      'The updater could not take the update lock, so nothing was changed. Hermes keeps running on the previous version — try the update again.',
      'updater-spawn-failed'
    )
  }

  /**
   * C2 + R5: the hand-off started only when a LIVE, correlated script process
   * holds the marker — not when a wrapper exited 0 (V7), and not when a pid
   * claimed and died. Returns an error message when it did not; the caller
   * stays up.
   */
  async function confirmScriptHandoff(child, plan: HandoffPlan, startedAtMs: number): Promise<string | null> {
    const outcome = await observeUpdaterHandoff(child, deps.updateHandoffDwellMs)
    let failure: string | null = outcome.ok ? null : describeUpdaterHandoffFailure(outcome)

    if (!failure) {
      const claim = await waitForHandoffClaim(deps.hermesHome, process.pid, {
        timeoutMs: Math.max(0, (deps.handoffClaimTimeoutMs ?? HANDOFF_CLAIM_TIMEOUT_MS) - (Date.now() - startedAtMs)),
        ...(plan.runId ? { runId: plan.runId } : { startedAt: plan.startedAt })
      })

      if (claim.taken) {
        deps.rememberLog(`[updates] hand-off script took the update marker (pid ${claim.pid})`)

        return null
      }

      failure =
        'The updater did not start, so nothing was changed. Hermes keeps running on the previous version — try the update again.\n\n' +
        `Details: the hand-off script never took the update lock within ${Math.round((deps.handoffClaimTimeoutMs ?? HANDOFF_CLAIM_TIMEOUT_MS) / 1000)} s.`
    }

    deps.rememberLog(`[updates] hand-off not viable, aborting quit: ${failure}`)

    // Withdraw the bridge through the script helper (under its lock): an
    // adopt-only script that starts late then finds nothing to adopt (A4).
    // `taken` means a live script adopted our run at the deadline — the
    // update IS running.
    if (plan.runId) {
      const verdict = await markerHelper('withdraw', plan.runId)

      if (verdict.kind === 'taken') {
        deps.rememberLog(`[updates] hand-off script took the update marker at the deadline (pid ${verdict.pid})`)

        return null
      }

      deps.rememberLog(`[updates] bridge withdraw: ${verdict.kind}`)
    }

    // MINOR-3: and stop the script tree we spawned, so nothing the UI just
    // reported as "did not start" can go on to run.
    killHandoffTree(child)

    return failure
  }

  const mechanism: UpdaterMechanism = deps.isWindows ? 'windows-handoff' : 'posix-handoff'

  async function check(opts: { force?: boolean } = {}): Promise<UpdaterStatusWire> {
    const root: string = deps.resolveUpdateRoot()

    // A checkout without the source probe predates source channels, so it can
    // only be on the git line: move it to main. Its update pulls the probe in.
    const status: UpdaterStatusWire = (await deps.readSourceUpdate(root, opts)) ?? {
      supported: true,
      updateAvailable: true,
      behind: null,
      branch: deps.defaultUpdateBranch,
      hermesRoot: root
    }

    status.mechanism = mechanism

    // Display data only (#122727): the statusbar and command palette read
    // currentSha, but a probe-less checkout never supplied it. The install
    // stamp names the checkout's commit without a git spawn; never let it
    // override the probe and never let it influence the update decision.
    if (!status.currentSha) {
      const stamped: string | null = readStampedCommit(root)

      if (stamped) {
        status.currentSha = stamped
      }
    }

    return status
  }

  /**
   * Staged (Tauri) updater path: spawn it detached and pre-write its marker
   * when it is new enough to adopt it.
   */
  async function spawnStagedUpdater(updater, updaterArgs: string[], updateRoot: string) {
    const child = spawnUpdaterProcess(updater, updaterArgs, {
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
    // The marker names the updater's pid AND creation time (v2): the updater
    // adopts it as its own claim, so no age ceiling applies to a slow update
    // and the `hermes update` it runs can add its delegate line.
    //
    // SKIPPED for pre-#74782 staged updaters: those have no self-PID
    // exclusion, so they read this very marker as a foreign live owner and
    // abort with "Another Hermes update is already running (PID <itself>)" —
    // an unbreakable loop, because the update that would replace the stale
    // binary is the one being refused. Losing the anti-respawn hardening is
    // strictly better than never updating again, and the updater still writes
    // its own marker moments later.
    // Exclusive create only (A7 rule 3): over any existing marker the
    // pre-write is skipped and the staged updater claims for itself.
    if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
      const prewrite = await writeUpdateMarker(deps.hermesHome, child.pid)

      if (!prewrite.ok) {
        deps.rememberLog(`[updates] skipping marker pre-write: ${describeSkippedPrewrite(prewrite)}`)
      }
    } else if (Number.isInteger(child.pid)) {
      deps.rememberLog(
        `[updates] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
      )
    }

    deps.rememberLog(
      `[updates] launched updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop for application replacement`
    )

    return child
  }

  /** The hand-off settle window: a failure message when the updater did not take over, else null. */
  async function settleHandoff(child, plan: HandoffPlan | null, handoffStartedAt: number): Promise<string | null> {
    if (plan) {
      return await confirmScriptHandoff(child, plan, handoffStartedAt)
    }

    const handoffOutcome = await observeUpdaterHandoff(child, deps.updateHandoffDwellMs)
    const failure = handoffOutcome.ok ? null : describeUpdaterHandoffFailure(handoffOutcome)

    if (failure) {
      deps.rememberLog(`[updates] hand-off not viable, aborting quit: ${handoffOutcome.message}`)
    }

    return failure
  }

  async function apply(): Promise<UpdaterApplyResultWire> {
    const result: UpdaterApplyResultWire = await applyBody()
    result.mechanism = mechanism

    return result
  }

  return { mechanism, check, apply }

  async function applyBody(): Promise<UpdaterApplyResultWire> {
    const status: UpdaterStatusWire = await check({ force: true })

    if (!status.supported || status.error) {
      return { ok: false, error: status.error ?? status.reason, message: status.message }
    }

    const branch: string = status.branch ?? deps.defaultUpdateBranch
    const targetArgs: string[] = status.channel ? ['--channel', status.channel] : ['--branch', branch]
    const targetLabel: string = status.channel ?? branch

    const manualCommand: string = status.channel
      ? `hermes update --channel ${status.channel}`
      : buildManualUpdateCommand(branch)

    const updater: string | null = deps.resolveUpdaterBinary()
    const root: string = deps.resolveUpdateRoot()

    // Earlier PM scripts still demand checkout/venv. Do not invoke that known
    // incompatible handoff: one exact-install CLI update obtains the new scripts.
    if (existsSync(path.join(root, 'pm')) && !existsSync(path.join(root, 'scripts', 'desktop-update', 'runtime.ps1'))) {
      const launcher: string | null = resolveInstallationLauncher(root, deps.isWindows, deps.hermesHome)

      if (!launcher) {
        return { ok: false, error: 'installation-launcher-missing' }
      }

      const quote = (value: string): string =>
        deps.isWindows ? `'${value.replace(/'/g, "''")}'` : `'${value.replace(/'/g, "'\\''")}'`

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

        deps.rememberLog(
          `[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot}`
        )
        deps.emitUpdateProgress({ stage: 'manual', message: command, percent: null })

        return { ok: true, manual: true, command, hermesRoot: updateRoot }
      }

      deps.rememberLog('[updates] no staged updater; using repo hand-off script for CLI install')
    }

    const handoffConflict = await updateHandoffConflict(deps.hermesHome)

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
    await deps.preflightStateDb(deps.hermesHome, deps.rememberLog)

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
    const handoffStartedAt = Date.now()
    let child
    let plan: HandoffPlan | null = null

    if (scriptHandoff) {
      const updateStartedAt = Math.floor(Date.now() / 1000)
      const planned = await planHandoff(scriptHandoff.scriptPath, updateStartedAt)

      if ('refusal' in planned) {
        deps.startHermes().catch(() => {})

        return planned.refusal
      }

      plan = planned.plan

      // A bare detached+hidden powershell spawn silently dies before -File
      // processing (console-subsystem init failure — see
      // wrapHandoffForDetachedConsole). Spawn the cmd wrapper non-detached
      // so windowsHide gives it a hidden console that `start /b` shares with
      // PowerShell; the script still outlives us. The wrapper exits at once,
      // so its pid means nothing: the bridge marker above names THIS process,
      // and the script takes it over (adopting -DesktopPid) as its first act.
      const wrappedArgs: string[] = [
        '-InstallRoot',
        updateRoot,
        ...(status.channel ? ['-Channel', status.channel] : ['-Branch', branch]),
        '-DesktopPid',
        String(process.pid),
        '-RelaunchExe',
        process.execPath
      ]

      // Same remote-ownership rule as the posix hand-off (#117529): a
      // remote-served Desktop owns no local messaging gateway.
      if (deps.remoteGatewayActive()) {
        wrappedArgs.push('-NoGateway')
      }

      if (plan.runId) {
        wrappedArgs.push('-HandoffRun', plan.runId)
      }

      const wrapped = wrapHandoffForDetachedConsole(scriptHandoff, wrappedArgs)

      child = spawnUpdaterProcess(wrapped.command, wrapped.args, {
        cwd: deps.hermesHome,
        env: {
          ...sourceUpdateEnvironment(updateRoot, deps.hermesHome),
          HERMES_UPDATE_STARTED_AT: String(updateStartedAt)
        },
        // Never `true` here: DETACHED_PROCESS leaves the wrapper console-less, so
        // `start /b` hands PowerShell a new VISIBLE console whose QuickEdit
        // selection can freeze the hand-off before relaunch (#103222).
        detached: wrapped.detached,
        stdio: 'ignore'
      })

      deps.rememberLog(
        `[updates] launched repo hand-off script: ${scriptHandoff.scriptPath} (${targetLabel}); exiting desktop for application replacement`
      )
    } else {
      child = await spawnStagedUpdater(updater, updaterArgs, updateRoot)
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
    // surface the error instead. A script hand-off additionally has to take
    // the bridge marker within 20 s (C2); the staged binary IS the updater, so
    // its own exit status is meaningful.
    const dwellStartedAt = Date.now()
    const failure: string | null = await settleHandoff(child, plan, handoffStartedAt)

    if (failure) {
      deps.emitUpdateProgress({ stage: 'error', message: failure, percent: null })
      deps.startHermes().catch(() => {})

      return { ok: false, error: 'updater-spawn-failed', message: failure }
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

  async function applyPosixHandoff(
    targetArgs: string[],
    targetLabel: string,
    manualCommand: string
  ): Promise<UpdaterApplyResultWire> {
    const updateRoot = deps.resolveUpdateRoot()
    const handoff = resolvePosixScriptHandoff(updateRoot)

    if (!handoff) {
      deps.emitUpdateProgress({ stage: 'manual', message: manualCommand, percent: null })

      return { ok: true, manual: true, command: manualCommand, hermesRoot: updateRoot }
    }

    const handoffConflict = await updateHandoffConflict(deps.hermesHome)

    if (handoffConflict) {
      // Same hazard as the Windows path (#75778): a live foreign updater
      // already owns the marker — refuse rather than double-mutate the tree.
      deps.rememberLog(`[updates] refusing posix hand-off: ${handoffConflict.message}`)
      deps.emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

      return { ok: false, error: 'update-already-running', message: handoffConflict.message }
    }

    // ── Pre-flight state.db integrity guard (#68474) ──
    await deps.preflightStateDb(deps.hermesHome, deps.rememberLog)

    const args: string[] = [
      ...handoff.args,
      '--install-root',
      updateRoot,
      ...targetArgs,
      '--desktop-pid',
      String(process.pid)
    ]

    // A remote-served Desktop owns no local messaging gateway: `hermes update
    // --gateway` would (re)start one here anyway, and with the same channel
    // credentials as the remote host it becomes a competing long-poll consumer
    // (#117529). Keep --gateway for the local-ownership default.
    if (deps.remoteGatewayActive()) {
      args.push('--no-gateway')
    }

    const handoffStartedAt = Date.now()
    const updateStartedAt = Math.floor(handoffStartedAt / 1000)
    const planned = await planHandoff(handoff.scriptPath, updateStartedAt)

    if ('refusal' in planned) {
      return planned.refusal
    }

    const { plan } = planned

    if (plan.runId) {
      args.push('--handoff-run', plan.runId)
    }

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
    // The launcher always exits 0 after daemonizing, so only the daemon taking
    // the bridge marker proves an update is running (C2, V7).
    const dwellStartedAt = Date.now()
    const failure = await confirmScriptHandoff(child, plan, handoffStartedAt)

    if (failure) {
      deps.emitUpdateProgress({ stage: 'error', message: failure, percent: null })

      return { ok: false, error: 'updater-spawn-failed', message: failure }
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
