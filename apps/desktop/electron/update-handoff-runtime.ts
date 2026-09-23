import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { stopBackendTreesForUpdate } from './backend-child'
import { isPidAliveWindows, waitForBackendRelease } from './backend-release-gate'
import { startGatewaysAfterUpdateAbort, stopGatewayBeforeUpdate } from './gateway-stop-before-update'
import { readPreUpdateBackupEnabled } from './pre-update-backup-config'
import { updateHandoffConflict, writeUpdateMarker } from './update-marker'
import {
  collectRelaunchArgs,
  describeUpdaterHandoffFailure,
  observeUpdaterHandoff,
  resolvePosixScriptHandoff,
  resolveStagedUpdaterBinary,
  resolveUpdateScriptHandoff,
  sandboxFallbackFromEnv,
  spawnUpdaterProcess,
  stagedUpdaterSupportsPrewrittenMarker,
  windowsUpdatePrerequisiteError,
  wrapHandoffForDetachedConsole
} from './updater-process'
import {
  formatBlockerMessage,
  formatProbeFailedMessage,
  resolveVenvDir,
  scanVenvBlockers,
  stopSafeVenvBlockers
} from './venv-blocker-scan'
import { isHermesOwnedVenvDaemon } from './venv-holder-select'
import { hiddenWindowsChildOptions } from './windows-child-options'
import { chooseUpdaterArgs } from './windows-hermes-path'

export interface UpdateHandoffRuntimeDeps {
  hermesHome: string
  isWindows: boolean
  isMac: boolean
  isPackaged: boolean
  updateHandoffDwellMs: number
  defaultUpdateBranch: string
  app: any
  backendConnectionState: any
  backendPool: Map<any, any>
  directoryExists: (path: string) => boolean
  emitUpdateProgress: (payload: any) => void
  fileExists: (path: string) => boolean
  getUpdateInFlight: () => boolean
  setUpdateInFlight: (value: boolean) => void
  setHandoffActive: (value: boolean) => void
  globalRemoteActive: () => boolean
  localBackendLifecycle: any
  pathWithHermesManagedNode: (...entries: string[]) => string
  readDesktopUpdateConfig: () => any
  rememberLog: (message: string) => void
  resolveHealedBranch: (root: string, branch: string) => Promise<string>
  resolveHermesBackend: (args: string[]) => Promise<any>
  resolveUpdateRoot: () => string
  runGit: (args: string[], options?: any) => Promise<{ code: number; stdout: string; stderr: string }>
  startHermes: () => Promise<any>
  stopAllPoolBackends: () => Promise<any>
}

export function createUpdateHandoffRuntime(deps: UpdateHandoffRuntimeDeps) {
  const HERMES_HOME = deps.hermesHome
  const IS_WINDOWS = deps.isWindows
  const IS_MAC = deps.isMac
  const IS_PACKAGED = deps.isPackaged
  const UPDATE_HANDOFF_DWELL_MS = deps.updateHandoffDwellMs
  const DEFAULT_UPDATE_BRANCH = deps.defaultUpdateBranch

  const {
    app,
    backendConnectionState,
    backendPool,
    directoryExists,
    emitUpdateProgress,
    fileExists,
    getUpdateInFlight,
    setUpdateInFlight,
    setHandoffActive,
    globalRemoteActive,
    localBackendLifecycle,
    pathWithHermesManagedNode,
    readDesktopUpdateConfig,
    rememberLog,
    resolveHealedBranch,
    resolveHermesBackend,
    resolveUpdateRoot,
    runGit,
    startHermes,
    stopAllPoolBackends
  } = deps

  // Resolve the staged updater binary the desktop may hand an update to. On
  // Windows that binary owns ALL repo mutation — running `hermes update` +
  // rebuilding the desktop — so the desktop never touches its own bits while
  // running. macOS/Linux stage the same binary but deliberately do not use it;
  // see resolveStagedUpdaterBinary for the policy and for #74836. Returns null
  // whenever no hand-off applies; callers degrade gracefully.
  function resolveUpdaterBinary() {
    return resolveStagedUpdaterBinary(HERMES_HOME, { fileExists, isWindows: IS_WINDOWS })
  }

  function repairMacUpdaterHelper(updater) {
    if (!IS_MAC || !updater) {
      return
    }

    try {
      execFileSync('/usr/bin/xattr', ['-cr', updater], { stdio: 'ignore' })
    } catch (err) {
      rememberLog(`[updates] macOS updater helper quarantine repair skipped: ${err.message}`)
    }

    try {
      execFileSync('/usr/bin/codesign', ['--verify', updater], { stdio: 'ignore' })

      return
    } catch {
      // Unsigned or invalid helper. Apply a local ad-hoc signature so Gatekeeper
      // does not block the staged updater before it can run.
    }

    try {
      execFileSync('/usr/bin/codesign', ['--force', '--sign', '-', updater], { stdio: 'ignore' })
      rememberLog('[updates] repaired macOS updater helper signature')
    } catch (err) {
      rememberLog(`[updates] macOS updater helper signature repair skipped: ${err.message}`)
    }
  }

  // Path to the venv shim whose lock decides whether `hermes update` can write
  // fresh entry points. On Windows this is the file the running backend
  // `hermes.exe` holds open; on POSIX it's never mandatory-locked.
  function venvHermesShimPath(updateRoot) {
    const venvDir = resolveVenvDir(updateRoot)

    return IS_WINDOWS ? path.join(venvDir, 'Scripts', 'hermes.exe') : path.join(venvDir, 'bin', 'hermes')
  }

  // Best-effort lock probe mirroring the Rust updater's is_locked(): a running
  // .exe on Windows refuses an O_RDWR open with a sharing violation. On POSIX
  // this practically always succeeds (no mandatory locking), so it returns false
  // — correct, since the shim-contention brick is Windows-only.
  function isShimLocked(shimPath) {
    if (!IS_WINDOWS) {
      return false
    }

    let fd

    try {
      fd = fs.openSync(shimPath, 'r+')

      return false
    } catch (err) {
      // ENOENT ⇒ not there ⇒ nothing locking it. Anything else (EBUSY/EPERM/
      // EACCES) on Windows means a live handle holds it.
      return err && err.code !== 'ENOENT'
    } finally {
      if (fd !== undefined) {
        try {
          fs.closeSync(fd)
        } catch {
          void 0
        }
      }
    }
  }

  // Kill only Hermes-OWNED venv daemons (the memory plugin's hindsight daemon:
  // exe under venv\Scripts AND cmdline referencing hindsight_api.main). The
  // daemon is spawned DETACHED, so it outlives the backend tree-kill and keeps
  // venv files mapped. External holders (a user terminal running `hermes`,
  // unrelated scripts) are NOT killed — scanVenvBlockers reports them and the
  // hand-off aborts, per existing design. Selection lives in the pure
  // venv-holder-select module (ordinal path-prefix, no PowerShell -like
  // wildcard hazards) so it's testable without Electron.
  function killHermesOwnedVenvDaemons(updateRoot) {
    if (!IS_WINDOWS) {
      return
    }

    const scriptsDir = path.join(resolveVenvDir(updateRoot), 'Scripts')

    let holders = []

    try {
      const out = execFileSync(
        'powershell',
        [
          '-NoProfile',
          '-Command',
          'Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.CommandLine } | Select-Object ProcessId, ExecutablePath, CommandLine | ConvertTo-Json -Compress'
        ],
        hiddenWindowsChildOptions({ encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 15_000 })
      )

      const parsed = JSON.parse(String(out || '[]'))

      holders = (Array.isArray(parsed) ? parsed : [parsed]).filter(p =>
        isHermesOwnedVenvDaemon(p?.ExecutablePath, p?.CommandLine, scriptsDir)
      )
    } catch {
      // Best-effort: the venv-blocker scan downstream is the real backstop.
      return
    }

    for (const holder of holders) {
      const pid = Number(holder?.ProcessId)

      if (Number.isInteger(pid) && pid > 0) {
        rememberLog(`[updates] stopping Hermes-owned venv daemon (hindsight) PID ${pid} before hand-off`)
        forceKillProcessTree(pid)
      }
    }
  }

  // Force-kill the entire process TREE rooted at each PID. Node's child.kill()
  // only signals the direct child, so on Windows a backend `hermes.exe` that
  // spawned its own grandchildren (a `hermes` REPL, a pty terminal session, the
  // gateway) would survive and keep the venv shim locked. taskkill /T /F reaps
  // the whole tree synchronously. Windows-only: this is called solely from the
  // Windows shim-unlock path, and the backend is NOT spawned detached (so it's
  // not a process-group leader — a POSIX negative-pgid kill would be meaningless
  // here anyway). POSIX teardown stays with the existing before-quit SIGTERM.
  function forceKillProcessTree(pid) {
    if (!IS_WINDOWS) {
      return
    }

    if (!Number.isInteger(pid) || pid <= 0) {
      return
    }

    try {
      execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], hiddenWindowsChildOptions({ stdio: 'ignore' }))
    } catch {
      // Already gone, or no permission — best effort; the unlock wait below is
      // the real gate.
    }
  }

  // Before handing off the update on Windows, the desktop MUST stop every backend
  // it spawned and WAIT for the venv shim to actually unlock. The old code did
  // `hermesProcess.kill('SIGTERM')` + `app.quit()` fire-and-forget: SIGTERM on
  // Windows doesn't reap the backend's grandchildren, and quit didn't wait for
  // teardown, so the updater raced a still-locked `hermes.exe`, the quarantine
  // rename failed, uv's `pip install` hit "Access is denied", and the git path
  // bailed into a full ZIP re-download that ALSO couldn't write the locked shim —
  // a half-applied install (ryanc's update.log). Here we tree-kill the primary +
  // pool backends and poll the shim until it's writable (or a bounded timeout),
  // so by the time we spawn the updater the lock is genuinely gone.
  //
  // Windows-only: the venv-shim mandatory lock is a Windows phenomenon. On
  // macOS/Linux there's no REPLACE-on-running-exe block, the existing before-quit
  // SIGTERM + app.quit() teardown already works (the macOS path is flawless), and
  // aggressively SIGKILL-ing the backend here would be an untested behavior change
  // for no benefit. So we no-op off Windows and leave that path exactly as it was.
  async function releaseBackendLockForUpdate(updateRoot) {
    return releaseBackendLock(updateRoot, 'updates')
  }

  // Shared backend teardown + venv-shim unlock wait. Used by BOTH the self-update
  // hand-off and the desktop uninstaller — they have the identical Windows
  // problem: the desktop's backend (and the grandchildren IT spawned — a hermes
  // REPL, a pty terminal, the gateway) keep `hermes.exe` and other files in the
  // venv mandatory-locked, so any in-place replace/delete of the install tree
  // races a live handle and half-fails (#37532). We tree-kill every backend PID
  // the desktop owns, then poll the shim until it's genuinely writable.
  //
  // `tag` only flavors the log lines. No-op off Windows (POSIX has no mandatory
  // locks — the before-quit SIGTERM + the cleanup script's own PID-wait suffice).
  async function releaseBackendLock(updateRoot, tag) {
    if (!IS_WINDOWS) {
      return { unlocked: true }
    }

    const hermesProcess = backendConnectionState.getProcess()

    // Seed the release gate with every PID we are about to signal: the
    // supervised primary backend and all pool backends. The gate waits for
    // these to actually LEAVE the process table, not just for the shim to
    // unlock — the shim probe only covers venv\Scripts\hermes.exe, but the
    // backend is `python.exe -m hermes_cli.main serve`, which need not hold
    // the shim at all (#74805 first-attempt race).
    const initialPids = []

    if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
      initialPids.push(hermesProcess.pid)
    }

    for (const entry of backendPool.values()) {
      if (entry.process && Number.isInteger(entry.process.pid)) {
        initialPids.push(entry.process.pid)
      }
    }

    stopBackendTreesForUpdate(hermesProcess, {
      forceKillProcessTree,
      stopAllPoolBackends
    })

    // Stop separately-running messaging gateways (all profiles) BEFORE the
    // release gate. The gateway is launched by the gateway-launcher desktop
    // plugin via /api/gateway/start and is NOT in backendConnectionState or
    // backendPool, so the tree-kills above never see it — on Windows its
    // launcher (venv\Scripts\python.exe) keeps the venv mandatory-locked and
    // the 15s gate aborts the hand-off before the venv-blocker scan's
    // pausable-gateway exemption ever gets a chance (#70337). Delegate to
    // `hermes gateway stop --all`: the CLI discovers every profile's gateway
    // (launcher + worker — gateway.pid records only the uv WORKER, and
    // taskkill /T from the worker never reaches its parent), drains in-flight
    // agents, and force-kills survivors. Best-effort; abort paths restore via
    // startGatewaysAfterUpdateAbort. No-op off Windows.
    stopGatewayBeforeUpdate(venvHermesShimPath(updateRoot), HERMES_HOME)

    // Reap Hermes-OWNED venv daemons the tree-kill above cannot reach: the
    // memory plugin's hindsight daemon is spawned DETACHED (it outlives the
    // backend) yet runs off venv\Scripts\pythonw.exe, keeping venv files
    // mapped past the backend teardown (#75477/#75478). Narrowly scoped
    // (venv-holder-select) — external holders are never killed here.
    killHermesOwnedVenvDaemons(updateRoot)

    const shim = venvHermesShimPath(updateRoot)

    const gate = await waitForBackendRelease(
      initialPids,
      {
        isShimLocked: () => Boolean(isShimLocked(shim)),
        isPidAlive: isPidAliveWindows,
        collectStragglerPids: () => {
          const stragglers = []

          const currentHermesProcess = backendConnectionState.getProcess()

          if (currentHermesProcess && Number.isInteger(currentHermesProcess.pid)) {
            stragglers.push(currentHermesProcess.pid)
          }

          for (const entry of backendPool.values()) {
            if (entry.process && Number.isInteger(entry.process.pid)) {
              stragglers.push(entry.process.pid)
            }
          }

          return stragglers
        },
        killProcessTree: forceKillProcessTree,
        sleep: (ms: number) => new Promise(r => setTimeout(r, ms)),
        now: () => Date.now(),
        log: rememberLog
      },
      tag
    )

    if (gate.unlocked) {
      return { unlocked: true }
    }

    // Do NOT proceed past a held lock: handing off to the updater while another
    // process (a second desktop window, a user terminal, an unkillable child)
    // still maps the venv's files guarantees a half-updated venv — the updater's
    // dependency sync dies on access-denied partway through uninstalls, leaving
    // imports broken (the July 2026 brotlicffi/_sodium.pyd incidents). Failing
    // the update loudly and keeping the app running is strictly better than a
    // bricked install that needs manual venv surgery.
    rememberLog(
      `[${tag}] venv shim still locked after 15s; aborting hand-off (something outside this app holds the venv)`
    )

    return { unlocked: false }
  }

  // applyUpdates — hand off to the installer's --update flow, then exit.
  //
  // The desktop is a pure consumer: it does NOT git pull / pip install / rebuild
  // itself (the old open-coded git dance lived here and drifted from
  // `hermes update`). Instead we spawn the staged Hermes-Setup binary with
  // --update and quit, so it can run `hermes update` (which refuses while we
  // hold the venv shim) and rebuild the desktop with our exe already gone.
  //
  // Detection (checkUpdates / commit changelog / "N behind") stays in the UI;
  // only this apply action changed.
  async function applyUpdates(opts: { stopSafeBlockers?: boolean } = {}) {
    if (getUpdateInFlight()) {
      throw new Error('An update is already in progress.')
    }

    setUpdateInFlight(true)

    try {
      const updater = resolveUpdaterBinary()

      if (!updater && !IS_WINDOWS) {
        // macOS/Linux: hand off to the repo-owned posix script — same shape as
        // Windows (quit → detached orchestrator → `hermes update` → relaunch),
        // minus the venv-lock gauntlet POSIX doesn't need. The old in-app
        // updater (applyUpdatesPosixInApp) is gone with everything it dragged
        // in: the HERMES_DESKTOP_CHILD_PID reaper-exclusion dance (#37532),
        // the in-window rebuild retry, and the relaunch-outcome matrix — the
        // script owns swap/relaunch, and the app is DEAD during the update so
        // there is nothing to reap around. Checkouts that predate the script
        // get the manual `hermes update` card once; their next update pulls it.
        return await applyUpdatesPosixHandoff(opts)
      }

      if (!updater) {
        // No staged updater binary — this is a CLI-installed user (they ran
        // `hermes desktop`, never the Tauri installer that self-copies
        // hermes-setup.exe into HERMES_HOME). On Windows the repo hand-off
        // script serves them just as well as installer users — it only needs
        // PowerShell and the checkout — so fall through to the normal hand-off
        // when the script exists. Only when the checkout predates the script do
        // we surface the manual one-liner.
        const updateRoot = resolveUpdateRoot()

        if (!resolveUpdateScriptHandoff(updateRoot)) {
          // They DO have a working `hermes` on PATH / in the venv, so the
          // correct path is the one-liner in their native medium. We show the
          // EXACT command, branch-pinned to the checkout they're on — bare
          // `hermes update` defaults to main and would silently switch a
          // bb/gui (or any non-main) install off-branch. Mirror the GUI
          // button's contract: append --branch <current> for non-main
          // checkouts, keep it bare for main so the card stays clean.
          let command = 'hermes update'

          try {
            const head = await runGit(['rev-parse', '--abbrev-ref', 'HEAD'], { cwd: updateRoot })
            const current = (head.stdout || '').trim()

            if (head.code === 0 && current && current !== 'HEAD') {
              const branch = await resolveHealedBranch(updateRoot, current)

              if (branch !== 'main') {
                command = `hermes update --branch ${branch}`
              }
            }
          } catch {
            // Best-effort: fall back to bare `hermes update` if branch detection fails.
          }

          rememberLog(`[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot}`)
          emitUpdateProgress({ stage: 'manual', message: command, percent: null })

          return { ok: true, manual: true, command, hermesRoot: updateRoot }
        }

        rememberLog('[updates] no staged updater; using repo hand-off script for CLI install')
      }

      const handoffConflict = updateHandoffConflict(HERMES_HOME)

      if (handoffConflict) {
        // A different updater already owns the marker — most often a previous
        // "Update" click whose updater is still alive and parked mid-run.
        // Spawning another here would overwrite its claim and let two updaters
        // mutate the checkout at once (#75778); refuse instead.
        rememberLog(`[updates] refusing hand-off: ${handoffConflict.message}`)
        emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

        return { ok: false, error: 'update-already-running', message: handoffConflict.message }
      }

      emitUpdateProgress({
        stage: 'restart',
        message:
          'Updating Hermes — this window will close and the updater will open. Don’t reopen Hermes yourself; it restarts automatically when the update finishes.',
        percent: 100
      })
      repairMacUpdaterHelper(updater)

      const updateRoot = resolveUpdateRoot()
      const { branch: configuredBranch } = readDesktopUpdateConfig()
      const branch = await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH)
      const updaterArgs = ['--update', '--branch', branch]
      const targetApp = IS_MAC ? runningAppBundle() : null

      if (targetApp) {
        updaterArgs.push('--target-app', targetApp)
      }

      const venvBin = path.join(resolveVenvDir(updateRoot), IS_WINDOWS ? 'Scripts' : 'bin')

      // ── Pre-flight state.db integrity guard (#68474) ─────────────────
      // Emergency backup and header verification before the update touches
      // anything.  Runs while the backend is still alive.
      await preflightStateDb(HERMES_HOME, rememberLog)

      if (IS_WINDOWS && resolveUpdateScriptHandoff(updateRoot)) {
        const message = windowsUpdatePrerequisiteError(updateRoot)

        if (message) {
          emitUpdateProgress({ stage: 'error', message, percent: null })

          return { ok: false, error: message }
        }
      }

      // Stop our own backend(s) and wait for the venv shim to unlock BEFORE we
      // spawn the updater. Without this the updater races a still-locked
      // hermes.exe (held by the backend child / its grandchildren) and the update
      // bricks. See releaseBackendLockForUpdate for the full failure analysis.
      const lock = await releaseBackendLockForUpdate(updateRoot)

      if (!lock.unlocked) {
        // Something OUTSIDE this app holds the venv (a second window, a user
        // terminal running hermes, an unkillable child). Handing off anyway
        // guarantees a half-updated venv — abort loudly instead and let the
        // user close the holder and retry. Restart our own backend so the app
        // keeps working after the failed attempt.
        const message =
          'Update aborted: another process is holding the Hermes install open ' +
          '(a second Hermes window or a terminal running hermes?). Close it and retry.'

        emitUpdateProgress({ stage: 'error', message, percent: null })
        startHermes().catch(() => {})

        if (IS_WINDOWS) {
          // The pre-gate `gateway stop --all` (#70337) took every profile's
          // gateway down for an update that never happened — bring them back.
          startGatewaysAfterUpdateAbort(venvHermesShimPath(updateRoot))
        }

        return { ok: false, error: message }
      }

      // Preflight: after releasing our own backends, check for remaining
      // Hermes processes running from this venv.  The updater normally refuses
      // when it detects a holder, but because the updater is spawned detached
      // with stdio:ignore, the user never sees that refusal and the update
      // silently fails.  This preflight detects holders early and gives the
      // user an actionable error.  Windows-only; the .pyd lock hazard is a
      // Windows phenomenon.  ALL failures (blocked, missing python, timeout,
      // malformed output, missing psutil) abort the handoff — never proceed
      // to the detached updater when the venv state is unknown.
      if (IS_WINDOWS) {
        let scanOutcome = await scanVenvBlockers(updateRoot)

        if (scanOutcome.kind === 'blocked' && opts.stopSafeBlockers) {
          const stopResult = await stopSafeVenvBlockers(updateRoot, scanOutcome.result)
          rememberLog(
            `[updates] user-approved blocker cleanup: stopped=${stopResult.stopped.join(',') || 'none'} failed=${stopResult.failed.join(',') || 'none'}`
          )
          // Let verified process-tree termination finish unwinding wrapper shells,
          // then make the scanner — not the stale renderer payload — authoritative.
          await new Promise(resolve => setTimeout(resolve, 300))
          scanOutcome = await scanVenvBlockers(updateRoot)
        }

        // Re-scan before aborting on 'blocked' (#74805). Process-table teardown
        // is asynchronous on Windows: even after releaseBackendLock's PID-exit
        // wait, a grandchild the desktop never tracked (or a process an AV /
        // NTFS filter driver is holding in teardown) can stay enumerable for a
        // few more seconds and read as a holder. Each scan already costs
        // seconds (spawns a venv python + psutil sweep), so two retries with a
        // short dwell give the table time to settle without meaningfully
        // delaying the abort path when a REAL holder (a user terminal, second
        // window) is present — that holder is still there on the third scan.
        for (let attempt = 0; scanOutcome.kind === 'blocked' && attempt < 2; attempt++) {
          rememberLog(
            `[updates] venv-blocker scan reported ${scanOutcome.result.processes.length} holder(s); re-scanning after settle (attempt ${attempt + 2}/3)`
          )
          await new Promise(resolve => setTimeout(resolve, 1500))
          scanOutcome = await scanVenvBlockers(updateRoot)
        }

        if (scanOutcome.kind === 'blocked') {
          const message = formatBlockerMessage(scanOutcome.result)

          rememberLog(`[updates] venv-blocked: ${scanOutcome.result.processes.length} process(es) hold the install`)
          emitUpdateProgress({ stage: 'error', message, percent: null })
          startHermes().catch(() => {})
          // Restore the gateways the pre-gate stop took down (#70337 drain
          // semantics): the update aborted, so nothing else will relaunch them.
          startGatewaysAfterUpdateAbort(venvHermesShimPath(updateRoot))

          return { ok: false, error: 'venv-blocked', message, blockers: scanOutcome.result.processes }
        }

        if (scanOutcome.kind === 'probe-failure') {
          const message = formatProbeFailedMessage(scanOutcome.error)

          rememberLog(`[updates] venv-blocker probe failed: ${scanOutcome.error}`)
          emitUpdateProgress({ stage: 'error', message, percent: null })
          startHermes().catch(() => {})
          // Same drain-semantics restore as the venv-blocked abort above.
          startGatewaysAfterUpdateAbort(venvHermesShimPath(updateRoot))

          return { ok: false, error: 'venv-probe-failed', message }
        }
      }

      // Detached so the updater outlives this process — it needs us GONE before
      // `hermes update` will run (the venv shim is locked while we live).
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
        // wrapHandoffForDetachedConsole). Route through a NON-detached, hidden
        // `cmd start /b` wrapper: cmd.exe owns one hidden console, the script
        // runs inside it (no window is ever created, #116161) and outlives
        // both cmd.exe and this process. The wrapper cmd.exe exits
        // immediately, so child.pid is NOT the script's pid — the script
        // claims the update marker itself with its own $PID as its first
        // action, and a relaunched Desktop parks on that.
        const wrappedArgs = [
          '-InstallRoot',
          updateRoot,
          '-Branch',
          branch,
          '-DesktopPid',
          String(process.pid),
          '-RelaunchExe',
          process.execPath
        ]

        // Same remote-ownership rule as the posix hand-off (#117529): a
        // remote-served Desktop must not let the update (re)start a local
        // messaging gateway that competes with the remote host's polling.
        if (globalRemoteActive()) {
          wrappedArgs.push('-NoGateway')
        }

        const wrapped = wrapHandoffForDetachedConsole(scriptHandoff, wrappedArgs)

        child = spawnUpdaterProcess(wrapped.command, wrapped.args, {
          cwd: HERMES_HOME,
          env: {
            ...process.env,
            HERMES_HOME,
            HERMES_UPDATE_STARTED_AT: String(updateStartedAt),
            PATH: pathWithHermesManagedNode(venvBin)
          },
          detached: wrapped.detached,
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
          writeUpdateMarker(HERMES_HOME, child.pid, { startedAt: updateStartedAt })
        }

        rememberLog(
          `[updates] launched repo hand-off script: ${scriptHandoff.scriptPath} (branch ${branch}); exiting desktop to release venv shim`
        )
      } else {
        child = spawnUpdaterProcess(updater, updaterArgs, {
          cwd: HERMES_HOME,
          env: {
            ...process.env,
            HERMES_HOME,
            PATH: pathWithHermesManagedNode(venvBin)
          },
          detached: true,
          stdio: 'ignore'
        })

        // Write the update-in-progress marker IMMEDIATELY — before the 2.5s
        // quit dwell. The Tauri updater won't write its own marker for several
        // seconds (window init + manifest), and during that gap our renderer
        // can reconnect and spawn a fresh backend that re-locks .pyd files in
        // the venv. By writing the marker ourselves the renderer's
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
          writeUpdateMarker(HERMES_HOME, child.pid)
        } else if (Number.isInteger(child.pid)) {
          rememberLog(
            `[updates] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
          )
        }

        rememberLog(
          `[updates] launched updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release venv shim`
        )
      }

      // Linger on the "updating — don't reopen" overlay long enough for the user
      // to actually read it (and to bridge the gap until the updater's own window
      // appears), THEN quit to release the venv shim. The updater rebuilds and
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
      const handoffOutcome = await observeUpdaterHandoff(child, UPDATE_HANDOFF_DWELL_MS)

      if (!handoffOutcome.ok) {
        const message = describeUpdaterHandoffFailure(handoffOutcome)

        rememberLog(`[updates] hand-off not viable, aborting quit: ${handoffOutcome.message}`)
        emitUpdateProgress({ stage: 'error', message, percent: null })
        startHermes().catch(() => {})

        if (IS_WINDOWS) {
          // Same drain-semantics restore as the earlier abort paths (#70337).
          startGatewaysAfterUpdateAbort(venvHermesShimPath(updateRoot))
        }

        return { ok: false, error: 'updater-spawn-failed', message }
      }

      setHandoffActive(true)
      setTimeout(
        () => {
          app.quit()
        },
        Math.max(0, UPDATE_HANDOFF_DWELL_MS - (Date.now() - dwellStartedAt))
      )

      return { ok: true, handedOff: true, updater }
    } finally {
      setUpdateInFlight(false)
    }
  }

  async function handOffWindowsBootstrapRecovery(reason) {
    if (!IS_WINDOWS || !IS_PACKAGED) {
      return false
    }

    const updater = resolveUpdaterBinary()

    if (!updater) {
      return false
    }

    const handoffConflict = updateHandoffConflict(HERMES_HOME)

    if (handoffConflict) {
      // Same hazard as applyUpdates (#75778): a live foreign updater already
      // owns the marker. Spawning another here would overwrite its claim and
      // race a second updater over the same install tree. The live updater
      // is already working on this exact install and will restart us when
      // it finishes, so treat this the same as a successful hand-off instead
      // of clobbering it with our own.
      rememberLog(`[bootstrap] refusing recovery hand-off: ${handoffConflict.message}`)
      setHandoffActive(true)
      setTimeout(() => {
        app.quit()
      }, UPDATE_HANDOFF_DWELL_MS)

      return true
    }

    const updateRoot = resolveUpdateRoot()
    const { branch: configuredBranch } = readDesktopUpdateConfig()

    const branch = directoryExists(path.join(updateRoot, '.git'))
      ? await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH)
      : configuredBranch || DEFAULT_UPDATE_BRANCH

    const venvBin = path.join(resolveVenvDir(updateRoot), IS_WINDOWS ? 'Scripts' : 'bin')
    const venvHermes = path.join(venvBin, IS_WINDOWS ? 'hermes.exe' : 'hermes')
    const venvPython = path.join(venvBin, IS_WINDOWS ? 'python.exe' : 'python')

    // The updater invokes the venv's Hermes launcher, which in turn requires the
    // venv interpreter. A bootstrap-complete marker proves only that setup once
    // finished; it can outlive a manually removed or quarantined venv. Sending a
    // marker-only install through --update dead-ends at "Could not find the hermes
    // CLI" instead of rebuilding the runtime, so only a runnable pair gets the
    // gentle update path. Partial or missing runtimes go through full repair.
    const updaterArgs = chooseUpdaterArgs(
      {
        hasBootstrapMarker: fileExists(path.join(updateRoot, '.hermes-bootstrap-complete')),
        hasVenvHermes: fileExists(venvHermes),
        hasVenvPython: fileExists(venvPython)
      },
      branch
    )

    await releaseBackendLockForUpdate(updateRoot)

    // The recovery resolver may have awaited while quit sealed local startup.
    localBackendLifecycle.assertCanStart()

    const child = spawnUpdaterProcess(updater, updaterArgs, {
      cwd: HERMES_HOME,
      env: {
        ...process.env,
        HERMES_HOME,
        PATH: pathWithHermesManagedNode(venvBin)
      },
      detached: true,
      stdio: 'ignore'
    })

    // Same marker pre-write as applyUpdates — see comment there. The recovery
    // hand-off has the same window where the renderer can respawn a backend
    // before the updater writes its own marker, and the same stale-updater
    // exclusion: a pre-#74782 binary would refuse its own pre-written claim and
    // strand the very recovery meant to heal the install.
    if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
      writeUpdateMarker(HERMES_HOME, child.pid)
    } else if (Number.isInteger(child.pid)) {
      rememberLog(
        `[bootstrap] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
      )
    }

    rememberLog(
      `[bootstrap] handed off ${reason} recovery to updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release app.asar`
    )
    // Same dwell as the in-app update hand-off (#50419): give the updater's
    // window time to appear before we vanish, so the recovery doesn't look like
    // a crash and provoke a mid-recovery relaunch. The dwell doubles as the
    // hand-off settle window (#66753): a spawn error or early updater death
    // returns false so the caller falls through to its next recovery path
    // instead of quitting into nothing.
    const dwellStartedAt = Date.now()
    const handoffOutcome = await observeUpdaterHandoff(child, UPDATE_HANDOFF_DWELL_MS)

    if (!handoffOutcome.ok) {
      rememberLog(`[bootstrap] recovery hand-off not viable, staying alive: ${handoffOutcome.message}`)

      return false
    }

    setHandoffActive(true)
    setTimeout(
      () => {
        app.quit()
      },
      Math.max(0, UPDATE_HANDOFF_DWELL_MS - (Date.now() - dwellStartedAt))
    )

    return true
  }

  // The running app's .app bundle (packaged macOS): execPath is
  // <App>.app/Contents/MacOS/<exe>; climb three levels to the bundle root.
  function runningAppBundle() {
    if (!IS_MAC) {
      return null
    }

    let dir = path.dirname(app.getPath('exe')) // .../Contents/MacOS

    for (let i = 0; i < 2; i++) {
      dir = path.dirname(dir)
    } // -> .../X.app

    return dir.endsWith('.app') ? dir : null
  }

  // ── Pre-flight state.db integrity guard (#68474) ─────────────────────
  // Take an emergency snapshot of state.db and verify the live copy is
  // intact before any update process mutates the install.  Runs in the
  // desktop Electron process itself, before the backend is killed and
  // before the updater is spawned — a separate safety net from the
  // Python-level pre-update snapshot inside `hermes update`.
  async function preflightStateDb(hermesHome, rememberLog) {
    const stateDbPath = path.join(hermesHome, 'state.db')

    if (!fileExists(stateDbPath)) {
      rememberLog('[updates] state.db pre-flight: not found (fresh install?)')

      return
    }

    try {
      const stat = fs.statSync(stateDbPath)

      if (stat.size > 100) {
        const fd = fs.openSync(stateDbPath, 'r')
        const header = Buffer.alloc(16)

        fs.readSync(fd, header, 0, 16, 0)
        fs.closeSync(fd)

        const expectedHeader = Buffer.from('SQLite format 3\0')
        const headerOk = header.equals(expectedHeader)

        rememberLog(
          `[updates] state.db pre-flight: size=${stat.size}, ` +
            `headerOk=${headerOk}, headerHex=${header.toString('hex')}`
        )

        if (!headerOk) {
          rememberLog(
            '[updates] state.db header is INVALID before update — ' +
              'this indicates pre-existing corruption or a concurrent write issue'
          )
        }

        if (
          !(await readPreUpdateBackupEnabled(
            resolveHermesBackend(['config', 'get', 'updates.pre_update_backup', '--json']),
            hermesHome
          ))
        ) {
          rememberLog('[updates] emergency state.db backup disabled by updates.pre_update_backup')

          return
        }

        // Emergency timestamped backup, separate from the Python-level snapshot.
        const ts = new Date().toISOString().replace(/[:.]/g, '-')

        const emergencyPath = path.join(hermesHome, `state.db.pre-update-emergency-${ts}.bak`)

        try {
          fs.copyFileSync(stateDbPath, emergencyPath)
          const emergStat = fs.statSync(emergencyPath)

          rememberLog(`[updates] emergency state.db backup: ${emergencyPath} ` + `(${emergStat.size} bytes)`)

          // Prune to the 2 most recent emergency backups.
          try {
            const homeDir = fs.readdirSync(hermesHome)

            const backups = homeDir
              .filter(
                f =>
                  f.startsWith('state.db.pre-update-emergency-') &&
                  f.endsWith('.bak') &&
                  f !== path.basename(emergencyPath)
              )
              .sort()
              .reverse()

            for (const old of backups.slice(2)) {
              try {
                fs.unlinkSync(path.join(hermesHome, old))
              } catch {
                void 0
              }
            }
          } catch {
            void 0
          }
        } catch (copyErr) {
          rememberLog(`[updates] emergency state.db backup failed: ${copyErr.message}`)
        }
      } else {
        rememberLog(`[updates] state.db too small (${stat.size} bytes) for a valid SQLite database`)
      }
    } catch (statErr) {
      rememberLog(`[updates] could not stat state.db before update: ${statErr.message}`)
    }
  }

  // macOS/Linux update hand-off: spawn the repo-owned posix orchestrator
  // (scripts/desktop-update/posix.sh) detached and QUIT. The script waits us
  // out, runs `hermes update`, swaps/relaunches the app bundle, and writes
  // .hermes-update-result.json for the relaunched Desktop to surface. It shows
  // its own tiny shim window (or nothing, headless) — this process only needs
  // to leave. Checkouts that predate the script get the manual card once.
  async function applyUpdatesPosixHandoff(opts: any) {
    const updateRoot = resolveUpdateRoot()
    const handoff = resolvePosixScriptHandoff(updateRoot)

    if (!handoff) {
      emitUpdateProgress({ stage: 'manual', message: 'hermes update', percent: null })

      return { ok: true, manual: true, command: 'hermes update', hermesRoot: updateRoot }
    }

    const handoffConflict = updateHandoffConflict(HERMES_HOME)

    if (handoffConflict) {
      // Same hazard as the Windows path (#75778): a live foreign updater
      // already owns the marker — refuse rather than double-mutate the tree.
      rememberLog(`[updates] refusing posix hand-off: ${handoffConflict.message}`)
      emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

      return { ok: false, error: 'update-already-running', message: handoffConflict.message }
    }

    // ── Pre-flight state.db integrity guard (#68474) ──
    await preflightStateDb(HERMES_HOME, rememberLog)

    // Branch-pin so a non-main checkout doesn't get switched to main (and
    // self-heal to main when the pinned branch no longer exists on origin).
    let branch = 'main'

    try {
      const head = await runGit(['rev-parse', '--abbrev-ref', 'HEAD'], { cwd: updateRoot })
      const current = (head.stdout || '').trim()

      if (head.code === 0 && current && current !== 'HEAD') {
        branch = await resolveHealedBranch(updateRoot, current)
      }
    } catch {
      // best effort
    }

    const args = [
      ...handoff.args,
      '--install-root',
      updateRoot,
      '--branch',
      branch,
      '--desktop-pid',
      String(process.pid)
    ]

    // A remote-served Desktop owns no local messaging gateway: `hermes update
    // --gateway` would (re)start one here anyway, and with the same channel
    // credentials as the remote host it becomes a competing long-poll consumer
    // (#117529). Keep --gateway for the local-ownership default.
    if (globalRemoteActive()) {
      args.push('--no-gateway')
    }

    const updateStartedAt = Math.floor(Date.now() / 1000)

    // Relaunch target: the running .app bundle on mac (script swaps the
    // rebuilt bundle over it), the running binary elsewhere. The script's gate
    // (an exact port of update-relaunch.ts's decideRelaunchOutcome) relaunches
    // only a binary the rebuild replaced with a launchable sandbox helper —
    // replaying the original launch context (filtered args, cwd, sandbox
    // opt-out) so a deep-link or --no-sandbox launch survives the update.
    const targetApp = IS_MAC ? runningAppBundle() : process.execPath

    if (targetApp) {
      args.push('--relaunch-target', targetApp)
    }

    const relaunchArgs = collectRelaunchArgs(process.argv.slice(1))

    if (!IS_MAC) {
      args.push('--relaunch-cwd', process.cwd())

      if (sandboxFallbackFromEnv(process.env, relaunchArgs)) {
        args.push('--sandbox-fallback')
      }

      if (relaunchArgs.length) {
        args.push('--', ...relaunchArgs)
      }
    }

    const child = spawnUpdaterProcess(handoff.command, args, {
      cwd: HERMES_HOME,
      env: {
        ...process.env,
        HERMES_HOME,
        HERMES_UPDATE_STARTED_AT: String(updateStartedAt),
        PATH: pathWithHermesManagedNode(path.join(resolveVenvDir(updateRoot), 'bin'))
      },
      detached: true,
      stdio: 'ignore'
    })

    // Bridge marker (same contract as the Windows hand-off): cover the gap
    // until the script claims the marker with its own pid as step 0. If the
    // script never starts, the dead pid reads as stale and self-deletes.
    if (Number.isInteger(child.pid)) {
      writeUpdateMarker(HERMES_HOME, child.pid, { startedAt: updateStartedAt })
    }

    rememberLog(`[updates] launched posix hand-off: ${handoff.scriptPath} (branch ${branch}); quitting to hand off`)
    emitUpdateProgress({
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
    const handoffOutcome = await observeUpdaterHandoff(child, UPDATE_HANDOFF_DWELL_MS)

    if (!handoffOutcome.ok) {
      const message = describeUpdaterHandoffFailure(handoffOutcome)

      rememberLog(`[updates] posix hand-off not viable, aborting quit: ${handoffOutcome.message}`)
      emitUpdateProgress({ stage: 'error', message, percent: null })

      return { ok: false, error: 'updater-spawn-failed', message }
    }

    setHandoffActive(true)
    setTimeout(
      () => {
        app.quit()
      },
      Math.max(0, UPDATE_HANDOFF_DWELL_MS - (Date.now() - dwellStartedAt))
    )

    return { ok: true, handedOff: true, updater: handoff.scriptPath }
  }

  return { forceKillProcessTree, releaseBackendLock, applyUpdates, handOffWindowsBootstrapRecovery }
}
