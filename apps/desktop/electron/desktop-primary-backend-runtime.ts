import type * as nodeCrypto from 'node:crypto'
import type * as nodeFs from 'node:fs'

import type * as backendClaim from './backend-claim'
import type * as backendConnection from './backend-connection-state'
import type * as backendExitRecovery from './backend-exit-recovery'
import type * as backendHealth from './backend-health'
import type * as backendOwnership from './backend-ownership'
import type * as backendReady from './backend-ready'
import type * as backendStartFailureFns from './backend-start-failure'
import type * as dashboardToken from './dashboard-token'
import type * as firstRunBootRuntime from './first-run-boot-runtime'
import type * as gatewayWsProbe from './gateway-ws-probe'
import type * as guestOnboarding from './guest-onboarding'
import type * as localBackendLifecycleRuntime from './local-backend-lifecycle'
import type * as parentProcessIdentity from './parent-process-identity'
import type * as primaryBackendStartup from './primary-backend-startup'
import type { PrimaryProfilePin } from './primary-profile-pin'
import type * as windowsChildOptions from './windows-child-options'

// The primary desktop backend owner. Dependencies are captured from main at
// composition time; mutable launch and recovery facts are read through state.
export interface DesktopPrimaryBackendState {
  isPrimaryInstance: boolean
  isQuittingForHandoff: boolean
  primaryStartsInFlight: number
  primaryRecoverySuppressed: boolean
  bootstrapFailure: Error | null
  backendStartFailure: Error | null
  remoteReauthFailure: Error | null
  bootstrapRepairAttempt: number
}

export interface DesktopPrimaryBackendDeps {
  state: DesktopPrimaryBackendState
  BOOT_FAKE_ERROR: string
  DESKTOP_LOG_PATH: string
  FirstRunSetupResetError: typeof primaryBackendStartup.FirstRunSetupResetError
  GUEST_ONBOARDING: boolean
  HERMES_HOME: string
  adoptServedDashboardToken: typeof dashboardToken.adoptServedDashboardToken
  attachToRunningHostBackend: any
  backendConnectionState: ReturnType<typeof backendConnection.createBackendConnectionState<any, any>>
  backendShutdown: ReturnType<typeof backendOwnership.createBackendShutdownCoordinator>
  claimBackendChild: any
  createBackendOutputTail: typeof backendClaim.createBackendOutputTail
  createPrimaryRemoteConnection: typeof primaryBackendStartup.createPrimaryRemoteConnection
  crypto: typeof nodeCrypto
  desktopBackendSpawnEnv: typeof guestOnboarding.desktopBackendSpawnEnv
  desktopParentStartMarker: any
  ensureLoginShellPath: any
  ensureRuntime: any
  firstLine: (message: string) => string
  firstRunBoot: ReturnType<typeof firstRunBootRuntime.createFirstRunBootRuntime>
  formatBackendExitLine: typeof backendClaim.formatBackendExitLine
  fs: typeof nodeFs
  getBackendArgsForRuntime: any
  getWindowState: any
  hermesLog: string[]
  hiddenWindowsChildOptions: typeof windowsChildOptions.hiddenWindowsChildOptions
  invalidatePrimaryConnection: any
  isHostKeyChangedBootFailure: typeof backendStartFailureFns.isHostKeyChangedBootFailure
  isReauthRequiredError: typeof backendHealth.isReauthRequiredError
  isRetryableRemoteBootFailure: typeof backendStartFailureFns.isRetryableRemoteBootFailure
  localBackendLifecycle: ReturnType<typeof localBackendLifecycleRuntime.createLocalBackendLifecycle<any>>
  makeDashboardReadyFile: any
  managedPrimaryRestoreOwners: ReadonlyMap<string, { correlationId: string; profile: string; source: any }>
  migrateActiveProfileIfMissing: any
  parentWatchdogEnv: typeof parentProcessIdentity.parentWatchdogEnv
  primaryBackendIsRemote: any
  primaryExitRecovery: ReturnType<typeof backendExitRecovery.createBackendExitRecoveryLatch>
  primaryProfileKey: any
  primaryProfilePin: PrimaryProfilePin
  probeGatewayWebSocket: typeof gatewayWsProbe.probeGatewayWebSocket
  readActiveDesktopProfile: any
  readStatusCode: any
  reapOrphanedBackendsOnce: any
  recentHermesLog: any
  releaseBackendChild: any
  releaseHostSpawnReservation: any
  rememberLog: (chunk: string | Buffer) => void
  resolveHermesBackend: any
  resolveHermesCwd: any
  resolveRemoteBackend: any
  resolveWebDist: any
  runPrimaryBackendStartup: any
  sendBackendExit: (payload: { code: number | null; signal: string | null; error?: string }) => void
  setActiveGatewayProfile: any
  setWslBridgeProfileState: any
  shouldLatchBackendStartFailure: typeof backendStartFailureFns.shouldLatchBackendStartFailure
  shouldLatchHostKeyChangedFailure: typeof backendStartFailureFns.shouldLatchHostKeyChangedFailure
  shouldLatchRemoteReauthFailure: typeof backendStartFailureFns.shouldLatchRemoteReauthFailure
  showPluginCompatNoticeOnce: any
  spawnOwnedBackend: any
  startAttachedBackendMonitor: any
  stopAttachedBackendMonitor: any
  stopBackendChild: any
  waitForBackendExit: any
  waitForDashboardPortAnnouncement: typeof backendReady.waitForDashboardPortAnnouncement
  waitForHermes: any
  waitForUpdateToFinish: any
}

export function createDesktopPrimaryBackendRuntime(deps: DesktopPrimaryBackendDeps) {
  const state = deps.state

  const {
    BOOT_FAKE_ERROR,
    DESKTOP_LOG_PATH,
    FirstRunSetupResetError,
    GUEST_ONBOARDING,
    HERMES_HOME,
    adoptServedDashboardToken,
    attachToRunningHostBackend,
    backendConnectionState,
    backendShutdown,
    claimBackendChild,
    createBackendOutputTail,
    createPrimaryRemoteConnection,
    crypto,
    desktopBackendSpawnEnv,
    desktopParentStartMarker,
    ensureLoginShellPath,
    ensureRuntime,
    firstLine,
    firstRunBoot,
    formatBackendExitLine,
    fs,
    getBackendArgsForRuntime,
    getWindowState,
    hermesLog,
    hiddenWindowsChildOptions,
    invalidatePrimaryConnection,
    isHostKeyChangedBootFailure,
    isReauthRequiredError,
    isRetryableRemoteBootFailure,
    localBackendLifecycle,
    makeDashboardReadyFile,
    managedPrimaryRestoreOwners,
    migrateActiveProfileIfMissing,
    parentWatchdogEnv,
    primaryBackendIsRemote,
    primaryExitRecovery,
    primaryProfileKey,
    primaryProfilePin,
    probeGatewayWebSocket,
    readActiveDesktopProfile,
    readStatusCode,
    reapOrphanedBackendsOnce,
    recentHermesLog,
    releaseBackendChild,
    releaseHostSpawnReservation,
    rememberLog,
    resolveHermesBackend,
    resolveHermesCwd,
    resolveRemoteBackend,
    resolveWebDist,
    runPrimaryBackendStartup,
    sendBackendExit,
    setActiveGatewayProfile,
    setWslBridgeProfileState,
    shouldLatchBackendStartFailure,
    shouldLatchHostKeyChangedFailure,
    shouldLatchRemoteReauthFailure,
    showPluginCompatNoticeOnce,
    spawnOwnedBackend,
    startAttachedBackendMonitor,
    stopAttachedBackendMonitor,
    stopBackendChild,
    waitForBackendExit,
    waitForDashboardPortAnnouncement,
    waitForHermes,
    waitForUpdateToFinish
  } = deps

  function startHermes({ supervisorRecovery = false }: { supervisorRecovery?: boolean } = {}) {
    state.primaryRecoverySuppressed = false
    state.primaryStartsInFlight += 1

    const start = localBackendLifecycle.start(() => runHermesStart({ supervisorRecovery }))

    const releaseStart = () => {
      state.primaryStartsInFlight -= 1
    }

    // Ordering contract: this reaction is registered on the SAME promise the
    // caller receives, before any caller `.catch`, so releaseStart has already
    // run (primaryStartsInFlight back to 0) when runPrimaryRecoverySpawn's
    // `.catch` evaluates primaryRecoveryState(). Returning a derived promise
    // (start.then(...)) or wrapping `start` would invert that order: every
    // pre-ready retry would see hasPendingStart:true, be refused, and leave the
    // recovery claim stuck with no retry and no UI.
    void start.then(releaseStart, releaseStart)

    return start
  }

  function primaryRecoveryState() {
    return {
      hasCurrentOwner: backendConnectionState.getProcess() !== null || backendConnectionState.getPromise() !== null,
      hasPendingStart: state.primaryStartsInFlight > 0,
      intentionalTeardown: state.primaryRecoverySuppressed || state.isQuittingForHandoff || backendShutdown.hasStarted()
    }
  }

  function reportPrimaryRecoveryCrashLoop(code: number | null, signal: string | null): boolean {
    if (!primaryExitRecovery.isCrashLooping()) {
      return false
    }

    const message =
      'Hermes backend keeps crashing right after it restarts; not restarting it again. Relaunch Hermes Desktop.'

    rememberLog(`[supervisor] ${message}`)
    sendBackendExit({ code, signal, error: message })

    return true
  }

  function runPrimaryRecoverySpawn(code: number | null, signal: string | null) {
    startHermes({ supervisorRecovery: true }).catch(respawnError => {
      rememberLog(`[supervisor] backend respawn failed: ${firstLine(respawnError.message)}`)

      // Terminal boot failures still own their existing recovery UI. Only a
      // supervisor-owned respawn that failed transiently before ready may spend
      // another bounded recovery slot.
      const latched = latchedBootFailure()

      if (latched) {
        rememberLog(`[supervisor] respawn refused: boot failure latched: ${firstLine(latched.message)}`)

        return
      }

      // releaseStart (startHermes) already ran: same-promise reaction order, so
      // hasPendingStart is false here. See the ordering contract in startHermes.
      if (primaryExitRecovery.retryAfterFailedStart(primaryRecoveryState())) {
        rememberLog('[supervisor] backend respawn failed before ready; retrying within crash-loop budget')
        runPrimaryRecoverySpawn(code, signal)

        return
      }

      reportPrimaryRecoveryCrashLoop(code, signal)
    })
  }

  // A ready primary child died. When its exit leaves the primary slot with no
  // owner and no start in flight (outside an intentional teardown), the
  // supervisor owns the respawn (#112344): the stale-classified exit used to
  // "log and return", and recovery then hinged on the renderer noticing its
  // socket drop — a 9 h engine-less window when it did not. Pool children are
  // deliberately not consulted: they never own the window backend.
  function scheduleUnexpectedPrimaryRecovery({
    code = null,
    signal = null,
    error = null,
    ready = false
  }: { code?: number | null; error?: string | null; ready?: boolean; signal?: string | null } = {}) {
    if (!ready) {
      return false
    }

    const claimed = primaryExitRecovery.claim(primaryRecoveryState())

    if (!claimed) {
      return reportPrimaryRecoveryCrashLoop(code, signal)
    }

    rememberLog('[supervisor] backend exit left no primary owner and no start in flight; respawning')
    sendBackendExit({ code, signal, ...(error ? { error } : {}) })
    runPrimaryRecoverySpawn(code, signal)

    return true
  }

  /**
   * The terminal boot failure currently latched in this process, if any. These
   * latches are cleared only by an explicit recovery path (reset, repair,
   * apply-config, confirmed sign-in, or the child 'exit' handler), never by a
   * retry, so both the per-request short-circuit in runHermesStart and the
   * supervisor's respawn refusal must consult the same trio in the same order.
   */
  function latchedBootFailure(): Error | null {
    return state.bootstrapFailure ?? state.backendStartFailure ?? state.remoteReauthFailure ?? null
  }

  async function runHermesStart({ supervisorRecovery = false }: { supervisorRecovery?: boolean } = {}) {
    // Only the single-instance lock holder may reap/spawn/claim the desktop
    // backend. A lock-losing instance must stay inert even if some path reaches
    // here (e.g. the deferred-quit window before `ready`): its reapOrphans()
    // otherwise SIGTERMs the running instance's live backend (#87295).
    if (!state.isPrimaryInstance) {
      rememberLog('[boot] non-primary instance: skipping backend machinery')
      throw new Error('Hermes Desktop is already running in another window.')
    }

    await reapOrphanedBackendsOnce()

    // Shutdown may have started while the orphan sweep was awaiting probes.
    localBackendLifecycle.assertCanStart()

    // Latched-failure short-circuit: once bootstrap has failed in this
    // process, every subsequent startHermes() call re-throws the same error
    // without re-running install.ps1. This prevents the renderer's
    // ensureGatewayOpen retries (and any other getConnection callers) from
    // restarting a 5-10 minute install loop while the user is still reading
    // the failure overlay.
    //
    // A confirmed remote reauth rejection is likewise terminal until the user
    // signs in. Short-circuiting here keeps the boot-failure overlay latched and
    // its "Sign in" button clickable, instead of re-driving boot on every retry.
    //
    // Deliberately silent: this runs on every proxied request while a failure is
    // latched (ensureBackend -> startHermes), so a log line here would flood the
    // bounded rememberLog ring and evict the lines that explain the original
    // failure. The supervisor logs the refusal once in runPrimaryRecoverySpawn.
    const latched = latchedBootFailure()

    if (latched) {
      throw latched
    }

    // E2E: simulate a boot failure without breaking the real backend. The boot
    // progresses a few steps, then fails with the given error message.
    if (BOOT_FAKE_ERROR) {
      await firstRunBoot.advanceBootProgress('backend.resolve', 'Resolving Hermes backend', 8)
      const error = new Error(BOOT_FAKE_ERROR) as any
      error.isBootstrapFailure = true
      state.bootstrapFailure = error
      throw error
    }

    const existingConnectionPromise = backendConnectionState.getPromise()

    if (existingConnectionPromise) {
      return existingConnectionPromise
    }

    // Seed active-profile.json from legacy signals BEFORE the first
    // profile-dependent read (`primaryBackendIsRemote()` on the next line, then
    // `primaryProfileKey()` inside the connection IIFE below). Without this,
    // remote-mode users whose preference file is missing (first boot after
    // update) resolve primaryProfileKey() to 'default' inside the IIFE, then
    // the remote branch returns and never runs the migration. Runs once;
    // no-op when the preference file already exists.
    migrateActiveProfileIfMissing()

    const connectionAttempt = backendConnectionState.startAttempt()
    const primaryProfile = primaryProfileKey()
    // Pin the routing table to the profile this primary actually boots as; a
    // later hermes:profile:remember must not retarget requests mid-life.
    primaryProfilePin.pin(primaryProfile)

    // Legacy path callers without an explicit profile belong to the primary
    // window backend. Profile-scoped callers still pass their key directly.
    setActiveGatewayProfile(primaryProfile)

    // Classify this boot BEFORE the throwing resolve/mint runs: a remote failure
    // must NOT latch (it's transient — see shouldLatchBackendStartFailure), while
    // a local failure latches to break install-restart loops.
    let attemptedRemote = managedPrimaryRestoreOwners.size > 0 || primaryBackendIsRemote()

    const connectionPromise = (async () => {
      const connectRemote = async remote => {
        // resolveRemote() may take arbitrarily long (settings resolve / ws-ticket
        // mint). If a newer attempt started meanwhile (e.g. the user switched
        // remotes and Apply invalidated this attempt), bail before probing.
        backendConnectionState.assertCurrentAttempt(connectionAttempt)

        await firstRunBoot.advanceBootProgress(
          'backend.remote',
          `Connecting to remote Hermes backend at ${remote.baseUrl}`,
          24
        )
        await waitForHermes(remote.baseUrl, remote.token, undefined, remote.authMode, remote.headers)

        // Second async boundary: the health probe itself can outlive the
        // attempt. A late success here must not publish a stale descriptor.
        backendConnectionState.assertCurrentAttempt(connectionAttempt)

        firstRunBoot.updateBootProgress({
          phase: 'backend.ready',
          message: 'Remote Hermes backend is ready',
          progress: 94,
          running: true,
          error: null
        })

        return createPrimaryRemoteConnection(remote, hermesLog.slice(-80), getWindowState())
      }

      await firstRunBoot.advanceBootProgress('backend.resolve', 'Resolving Hermes backend', 8)
      // Resolve for the desktop's primary profile so a per-profile remote
      // override on the active profile is honored (falls back to env / global).

      // GUI launches (Finder/Dock, desktop launchers) inherit a minimal PATH
      // that skips the user's shell profiles. Merge the login-shell PATH into
      // process.env BEFORE resolving the runtime or spawning the backend, so
      // both the Electron-side resolvers and the whole backend subtree (tool
      // availability checks, stdio MCP servers) can find Homebrew-, nvm-, and
      // ~/.local/bin-installed CLIs. Single-flight with the whenReady warmup;
      // failure-hardened — a broken shell profile never blocks boot.
      const loginShellPath = await ensureLoginShellPath()

      if (loginShellPath.applied) {
        rememberLog('[env] merged login-shell PATH into process.env for backend spawn')
      } else if (loginShellPath.reason && !['win32', 'unchanged'].includes(loginShellPath.reason)) {
        rememberLog(`[env] login-shell PATH resolution unavailable (${loginShellPath.reason}); keeping inherited PATH`)
      }

      const token = crypto.randomBytes(32).toString('base64url')
      // --port 0: the OS assigns an ephemeral port; the child announces it on stdout.
      const backendArgs = ['serve', '--host', '127.0.0.1', '--port', '0']
      // Pin the desktop's chosen profile via the global --profile flag. This is
      // deterministic (it wins over the sticky ~/.hermes/active_profile file) and
      // resolves HERMES_HOME the same way `hermes -p <name>` does on the CLI. An
      // unset preference keeps the legacy launch so existing installs are
      // unaffected.
      const activeProfile = readActiveDesktopProfile()

      if (activeProfile) {
        backendArgs.unshift('--profile', activeProfile)
      }

      const setup = await runPrimaryBackendStartup({
        signal: localBackendLifecycle.signal,
        assertCurrentAttempt: () => backendConnectionState.assertCurrentAttempt(connectionAttempt),
        attachHostBackend: attachToRunningHostBackend,
        connectRemote,
        ensureLocalRuntime: backend =>
          ensureRuntime(backend, () => backendConnectionState.assertCurrentAttempt(connectionAttempt)),
        prepareLocalBackend: async () => {
          await firstRunBoot.advanceBootProgress('backend.runtime', 'Resolving Hermes runtime', 28)

          return resolveHermesBackend(backendArgs)
        },
        resolveRemote: () => {
          // Classify immediately before each throwing resolve. This callback runs
          // both for an already-saved remote and after first-run remote Apply.
          attemptedRemote = managedPrimaryRestoreOwners.size > 0 || primaryBackendIsRemote()

          return resolveRemoteBackend(primaryProfile, { primary: true })
        },
        waitForDecision: firstRunBoot.waitForFirstRunSetupChoice,
        // Mutual exclusion with an in-app update (#50238). Remote connections
        // return before this waiter; local starts park until the updater exits.
        waitForLocalStart: waitForUpdateToFinish
      })

      backendConnectionState.assertCurrentAttempt(connectionAttempt)

      if (setup.kind === 'remote') {
        // Paths from the remote backend belong to a host the Windows desktop
        // cannot open via wsl.exe — disable WSL path bridging so native dialogs
        // and file panels don't spawn wsl.exe (or the interactive install prompt
        // on WSL-less machines) for unresolvable paths. (#66433)
        setWslBridgeProfileState(primaryProfile, false)

        return setup.connection
      }

      // Multiplex-only: a backend was already running on this host and we attached
      // to it. Nothing was spawned, so there is no child to own — liveness is
      // polled instead (startAttachedBackendMonitor).
      if (setup.kind === 'attached') {
        const attached = setup.attached

        setWslBridgeProfileState(primaryProfile, true)
        startAttachedBackendMonitor(attached)

        firstRunBoot.updateBootProgress({
          phase: 'backend.ready',
          message: 'Attached to the running Hermes backend',
          progress: 94,
          running: true,
          error: null
        })

        return {
          baseUrl: attached.baseUrl,
          mode: 'local',
          source: 'local',
          authMode: 'token',
          attached: true,
          token: attached.token,
          profile: primaryProfile,
          wsUrl: attached.wsUrl,
          logs: hermesLog.slice(-80),
          ...getWindowState()
        }
      }

      // Local WSL backend — paths are bridgeable.
      setWslBridgeProfileState(primaryProfile, true)

      stopAttachedBackendMonitor()

      const backend = setup.backend
      // Route old runtimes (no `serve`) through the legacy `dashboard --no-open`.
      backend.args = await getBackendArgsForRuntime(backend)
      backendConnectionState.assertCurrentAttempt(connectionAttempt)
      const hermesCwd = resolveHermesCwd()
      const webDist = resolveWebDist()
      const readyFile = backend.readyFile ? makeDashboardReadyFile() : null

      await firstRunBoot.advanceBootProgress('backend.spawn', `Starting Hermes backend via ${backend.label}`, 84)
      rememberLog(`Starting Hermes backend via ${backend.label}`)

      const profile = primaryProfileKey()
      const parentStartMarker = await desktopParentStartMarker()
      const backendNonce = crypto.randomBytes(16).toString('hex')
      const parentIdentityEnv = parentWatchdogEnv(process.pid, parentStartMarker, backendNonce)

      backendConnectionState.assertCurrentAttempt(connectionAttempt)

      const hermesProcess = spawnOwnedBackend(
        backend.command,
        backend.args,
        hiddenWindowsChildOptions({
          cwd: hermesCwd,
          env: desktopBackendSpawnEnv(
            {
              ...process.env,
              // Explicitly pin HERMES_HOME for the child so Python's get_hermes_home()
              // resolves to the SAME location our resolveHermesHome() picked. Without
              // this pin, Python falls back to ~/.hermes on every platform — fine on
              // mac/linux (where our default matches), but on Windows our default is
              // %LOCALAPPDATA%\hermes, which differs from C:\Users\<u>\.hermes.
              // Mismatch would split config / sessions / .env / logs across two
              // directories. install.ps1 sets HERMES_HOME via setx; the desktop
              // can't reliably do that, so we set it inline for every spawn.
              HERMES_HOME,
              ...backend.env,
              TERMINAL_CWD: hermesCwd,
              HERMES_DASHBOARD_SESSION_TOKEN: token,
              // Marks this dashboard backend as desktop-spawned so it runs the cron
              // scheduler tick loop (the gateway isn't running under the app).
              HERMES_DESKTOP: '1',
              // Exact parent identity lets the backend self-exit after an unclean
              // Desktop death without mistaking a reused PID for its owner. If the
              // optional marker probe fails, retain legacy PID-only tracking.
              ...parentIdentityEnv,
              HERMES_WEB_DIST: webDist,
              ...(readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {})
            },
            GUEST_ONBOARDING
          ),
          shell: backend.shell,
          stdio: ['ignore', 'pipe', 'pipe']
        })
      )

      // Buffer stdout+stderr from the instant of spawn (#93608): an early
      // crash's traceback must survive into the claim error and the
      // before-ready exit message shown by the boot UI. rememberLog attaches
      // later, after the claim, and would miss anything printed before it.
      const primaryOutputTail = createBackendOutputTail()
      primaryOutputTail.attach(hermesProcess)

      // Start watching for the READY announcement BEFORE any await (#60323):
      // claimBackendChild can take seconds (its Windows Get-Process probe cold
      // start alone runs 2-8s) and advanceBootProgress awaits renderer IPC.
      // stdout is already flowing into the tail, and Node streams never replay
      // consumed chunks to late listeners, so a sentinel printed during that
      // window was lost forever — the wait then hit its 90s timeout and a
      // healthy backend was killed (deterministic on Windows, racy on
      // macOS/Linux). The tail-buffer accessor covers any residual gap.
      const portAnnouncement = waitForDashboardPortAnnouncement(hermesProcess, {
        bufferedOutput: () => primaryOutputTail.text(),
        describeOutputTail: () => primaryOutputTail.describe(),
        readyFile
      })

      // Mark handled so an early rejection (child dies during the claim) can't
      // surface as an unhandled rejection before the Promise.race below attaches.
      portAnnouncement.catch(() => {})
      await claimBackendChild(
        hermesProcess,
        `${backend.command} ${backend.args.join(' ')}`,
        profile,
        backendNonce,
        primaryOutputTail
      )
      const processOwner = backendConnectionState.attachProcess(connectionAttempt, hermesProcess)

      if (!processOwner) {
        stopBackendChild(hermesProcess)
        await waitForBackendExit(hermesProcess)
        releaseBackendChild(hermesProcess)
        throw new Error('Hermes backend start was superseded by a newer connection attempt.')
      }

      hermesProcess.stdout.on('data', rememberLog)
      hermesProcess.stderr.on('data', rememberLog)
      let backendReady = false
      let rejectBackendStart = null

      const backendStartFailed = new Promise((_resolve, reject) => {
        rejectBackendStart = reject
      })

      hermesProcess.once('error', error => {
        releaseBackendChild(hermesProcess)

        if (!backendConnectionState.clearForCurrentProcess(processOwner)) {
          rememberLog(`Ignoring stale Hermes backend error: ${error.message}`)
          scheduleUnexpectedPrimaryRecovery({ error: error.message, ready: backendReady })
          rejectBackendStart?.(new Error('Hermes backend start was superseded by a newer connection attempt.'))

          return
        }

        rememberLog(`Hermes backend failed to start: ${error.message}`)
        firstRunBoot.updateBootProgress(
          {
            error: error.message,
            message: `Hermes backend failed to start: ${error.message}`,
            phase: 'backend.error',
            running: false
          },
          { allowDecrease: true }
        )
        sendBackendExit({ code: null, signal: null, error: error.message })
        rejectBackendStart?.(error)
      })
      hermesProcess.once('exit', (code, signal) => {
        releaseBackendChild(hermesProcess)

        if (!backendConnectionState.clearForCurrentProcess(processOwner)) {
          rememberLog(formatBackendExitLine('Ignoring stale Hermes backend exit', code, signal, primaryOutputTail))

          scheduleUnexpectedPrimaryRecovery({ code, signal, ready: backendReady })

          if (!backendReady) {
            rejectBackendStart?.(new Error('Hermes backend start was superseded by a newer connection attempt.'))
          }

          return
        }

        rememberLog(formatBackendExitLine('Hermes backend exited', code, signal, primaryOutputTail))

        if (!scheduleUnexpectedPrimaryRecovery({ code, signal, ready: backendReady })) {
          sendBackendExit({ code, signal })
        }

        if (!backendReady) {
          const message = `Hermes backend exited before it became ready (${signal || code}).${primaryOutputTail.describe()}`
          firstRunBoot.updateBootProgress(
            {
              error: message,
              message,
              phase: 'backend.error',
              running: false
            },
            { allowDecrease: true }
          )
          rejectBackendStart?.(
            new Error(
              `Hermes backend exited before it became ready (${signal || code}). Log: ${DESKTOP_LOG_PATH}\n${recentHermesLog()}`
            )
          )
        }
      })

      await firstRunBoot.advanceBootProgress('backend.port', 'Waiting for Hermes backend to launch', 86)
      backendConnectionState.assertCurrentAttempt(connectionAttempt)

      // Discover the ephemeral port the child bound to
      const port = await Promise.race([portAnnouncement, backendStartFailed])
      backendConnectionState.assertCurrentAttempt(connectionAttempt)

      if (readyFile) {
        fs.unlink(readyFile, () => {})
      }

      const baseUrl = `http://127.0.0.1:${port}`
      await firstRunBoot.advanceBootProgress('backend.wait', 'Waiting for Hermes backend to become ready', 90)
      backendConnectionState.assertCurrentAttempt(connectionAttempt)
      await Promise.race([waitForHermes(baseUrl, token), backendStartFailed])
      backendConnectionState.assertCurrentAttempt(connectionAttempt)
      backendReady = true
      // The host now has a bound, registered backend: the next launcher will
      // discover and attach to it, so the spawn gate is done.
      releaseHostSpawnReservation()
      primaryExitRecovery.reset()
      state.backendStartFailure = null

      const authToken = await adoptServedDashboardToken(baseUrl, token, {
        childAlive: () => hermesProcess.exitCode === null && !hermesProcess.killed,
        rememberLog
      })

      backendConnectionState.assertCurrentAttempt(connectionAttempt)

      // Verify the WebSocket session token before declaring backend ready.
      const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`
      const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket })
      backendConnectionState.assertCurrentAttempt(connectionAttempt)

      if (!wsProbe.ok) {
        throw new Error(
          `Local Hermes backend is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: ${wsProbe.reason}`
        )
      }

      firstRunBoot.updateBootProgress({
        phase: 'backend.ready',
        message: 'Hermes backend is ready. Finalizing desktop startup',
        progress: 94,
        running: true,
        error: null
      })

      // A successful boot (including a soft restart that the repair-guard
      // chose over a hard reinstall, see #74874) means any in-flight repair
      // attempt counter has been honoured — reset it so the next genuine
      // failure starts fresh from attempt 1 instead of inheriting the
      // accumulated count of the resolved episode.
      state.bootstrapRepairAttempt = 0

      // The backend's plugin discovery just ran and refreshed HERMES_HOME/.plugin-compat-report.json.
      // Surface it once (per distinct set of affected plugins) after the window is up; never block boot.
      setTimeout(() => void showPluginCompatNoticeOnce(), 1500)

      return {
        baseUrl,
        mode: 'local',
        source: 'local',
        authMode: 'token',
        token: authToken,
        profile,
        wsUrl,
        logs: hermesLog.slice(-80),
        ...getWindowState()
      }
    })().catch(async error => {
      releaseHostSpawnReservation()

      if (!backendConnectionState.clearPromiseForAttempt(connectionAttempt)) {
        throw error
      }

      const failedProcess = invalidatePrimaryConnection()
      stopBackendChild(failedProcess)

      if (error instanceof FirstRunSetupResetError) {
        await waitForBackendExit(failedProcess)
        throw error
      }

      const message = error instanceof Error ? error.message : String(error)
      const hostKeyChanged = isHostKeyChangedBootFailure(error)
      const isReauth = isReauthRequiredError(error)

      // Carry structured Cloud-down metadata through the boot-progress / IPC
      // boundary when present, so the renderer overlay can key on it rather than
      // re-classifying the message string. main owns classification; the renderer
      // only consumes the structured result (#85335).
      const isCloudBackendDown = Boolean(
        error && typeof error === 'object' && (error as any).isCloudBackendDown === true
      )

      const statusCode = readStatusCode(error)

      // Only latch LOCAL boot failures. A remote failure (lapsed session / mint
      // timeout / host briefly unreachable across sleep) is transient and has no
      // child 'exit' handler to clear the cache — latching it would wedge the app
      // on "session expired" until a full restart, defeating reconnect, the
      // "Sign out & sign in" reload, and the wake-recovery revalidate path.
      // A supervisor-owned respawn never latches (see the predicate).
      if (shouldLatchBackendStartFailure({ attemptedRemote, supervisorRecovery })) {
        state.backendStartFailure = error instanceof Error ? error : new Error(message)
      }

      // A host-key CHANGE is the terminal exception among remote failures: SSH
      // fails closed until the user verifies the change and clears the stale
      // known_hosts entry, so retrying re-drives the identical doomed boot (one
      // bundle showed 157 consecutive failures over 2.5h). Latch it like a local
      // failure — reset/repair/apply-config clear the latch after the user fixes
      // known_hosts.
      if (shouldLatchHostKeyChangedFailure({ attemptedRemote, isReauth: false, isHostKeyChanged: hostKeyChanged })) {
        state.backendStartFailure = error instanceof Error ? error : new Error(message)
      }

      // A confirmed reauth rejection latches separately: it can't self-heal, and
      // leaving it unlatched hides the overlay's "Sign in" button on every retry.
      if (shouldLatchRemoteReauthFailure({ attemptedRemote, isReauth })) {
        state.remoteReauthFailure = error instanceof Error ? error : new Error(message)
        rememberLog('[boot] remote reauth latched: holding boot-progress until a recovery path clears it')
      }

      // Every latch above is set BEFORE this first yield back to the event loop.
      // invalidate() already dropped the shared attempt promise, so a concurrent
      // getConnection()/startHermes() caller arriving during the exit wait would
      // otherwise start a brand-new attempt, re-emit running:true over the
      // failure and re-drive the identical rejection. With the latch in place it
      // short-circuits on the cached failure instead: the first confirmed
      // rejection owns the transition into recovery (#95701).
      await waitForBackendExit(failedProcess)

      firstRunBoot.updateBootProgress(
        {
          error: message,
          isCloudBackendDown: isCloudBackendDown || undefined,
          message: `Desktop boot failed: ${message}`,
          phase: 'backend.error',
          // Renderer contract for the self-heal loop (#82679): a transient
          // REMOTE failure (dropped SSH/HTTP registered connection, mint
          // timeout) is retryable — the renderer re-attempts the boot with
          // bounded backoff. Local failures, confirmed reauth rejections, and
          // host-key changes are not: those end in the recovery overlay /
          // sign-in affordance.
          retryable: isRetryableRemoteBootFailure({
            attemptedRemote,
            isReauth,
            isHostKeyChanged: hostKeyChanged
          }),
          running: false,
          statusCode: Number.isInteger(statusCode) ? statusCode : undefined
        },
        { allowDecrease: true }
      )
      throw error
    })

    backendConnectionState.setPromise(connectionAttempt, connectionPromise)

    return connectionPromise
  }

  return { startHermes, scheduleUnexpectedPrimaryRecovery }
}
