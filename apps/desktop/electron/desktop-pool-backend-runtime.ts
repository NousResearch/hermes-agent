import type * as childProcess from 'node:child_process'

import type * as poolRetire from './pool-retire'
import type * as poolRetireHttp from './pool-retire-http'
import type { LocalBackendSpawnPriority, LocalBackendSpawnRequest } from './pool-spawn-coordinator'
import type * as poolStop from './pool-stop'
import type * as updateGate from './update-gate'

// The original main-process pool objects remain the sole admission and teardown
// authority. Compose once after the ownership and SSH runtimes are ready.
export interface DesktopPoolBackendRuntimeDeps {
  adoptServedDashboardToken: any
  assertLocalProfileCanStart: any
  assertNoSecondLocalBackend: any
  backendPool: Map<string, any>
  backgroundSlotRetryBackoff: any
  BackgroundSlotRetryDeferredError: any
  BrowserWindow: any
  claimBackendChild: any
  createBackendOutputTail: any
  createPoolRetirer: typeof poolRetire.createPoolRetirer
  createPoolRetirementClient: typeof poolRetireHttp.createPoolRetirementClient
  createPoolStopper: typeof poolStop.createPoolStopper
  crypto: any
  desktopBackendSpawnEnv: any
  desktopParentStartMarker: any
  directoryExists: any
  ensureRuntime: any
  fetchJson: any
  formatBackendExitLine: any
  fs: any
  getBackendArgsForRuntime: any
  getWindowState: any
  GUEST_ONBOARDING: any
  HERMES_HOME: any
  hermesLog: any
  hiddenWindowsChildOptions: any
  isBackgroundSlotWaitTimeout: any
  localBackendLifecycle: any
  localBackendSpawnCoordinator: any
  makeDashboardReadyFile: any
  parentWatchdogEnv: any
  path: any
  POOL_SLOT_WAIT_MS: any
  poolMaxBackends: any
  probeGatewayWebSocket: any
  profileDeletionGate: any
  profileRouteOptions: any
  reapOrphanedBackendsOnce: any
  registerLocalBackendExitFinalizer: any
  releaseBackendChild: any
  releaseLocalBackendSlotAfterExit: any
  rememberLog: any
  resolveHermesBackend: any
  resolveHermesCwd: any
  resolveRemoteBackend: any
  resolveWebDist: any
  spawnOwnedBackend: (...args: Parameters<typeof childProcess.spawn>) => ReturnType<typeof childProcess.spawn>
  spawnPriorityFrom: any
  sshBootstrapCoordinator: any
  sshRememberLog: any
  stopBackendChild: any
  takeForegroundSpawn: any
  teardownSshConnection: any
  UPDATE_WAIT_POLL_MS: any
  UPDATE_WAIT_TIMEOUT_MS: any
  updateGateDeps: any
  waitForBackendExit: any
  waitForDashboardPortAnnouncement: any
  waitForHermes: any
  waitForUpdateClearance: typeof updateGate.waitForUpdateClearance
}

export function createDesktopPoolBackendRuntime(deps: DesktopPoolBackendRuntimeDeps) {
  const {
    adoptServedDashboardToken,
    assertLocalProfileCanStart,
    assertNoSecondLocalBackend,
    backendPool,
    backgroundSlotRetryBackoff,
    BackgroundSlotRetryDeferredError,
    BrowserWindow,
    claimBackendChild,
    createBackendOutputTail,
    createPoolRetirer,
    createPoolRetirementClient,
    createPoolStopper,
    crypto,
    desktopBackendSpawnEnv,
    desktopParentStartMarker,
    directoryExists,
    ensureRuntime,
    fetchJson,
    formatBackendExitLine,
    fs,
    getBackendArgsForRuntime,
    getWindowState,
    GUEST_ONBOARDING,
    HERMES_HOME,
    hermesLog,
    hiddenWindowsChildOptions,
    isBackgroundSlotWaitTimeout,
    localBackendLifecycle,
    localBackendSpawnCoordinator,
    makeDashboardReadyFile,
    parentWatchdogEnv,
    path,
    POOL_SLOT_WAIT_MS,
    poolMaxBackends,
    probeGatewayWebSocket,
    profileDeletionGate,
    profileRouteOptions,
    reapOrphanedBackendsOnce,
    registerLocalBackendExitFinalizer,
    releaseBackendChild,
    releaseLocalBackendSlotAfterExit,
    rememberLog,
    resolveHermesBackend,
    resolveHermesCwd,
    resolveRemoteBackend,
    resolveWebDist,
    spawnOwnedBackend,
    spawnPriorityFrom,
    sshBootstrapCoordinator,
    sshRememberLog,
    stopBackendChild,
    takeForegroundSpawn,
    teardownSshConnection,
    UPDATE_WAIT_POLL_MS,
    UPDATE_WAIT_TIMEOUT_MS,
    updateGateDeps,
    waitForBackendExit,
    waitForDashboardPortAnnouncement,
    waitForHermes,
    waitForUpdateClearance
  } = deps

  function releaseLocalBackendSlot(entry: any) {
    if (!entry) {
      return
    }

    const release = entry.releaseLocalBackendSlot
    const request = entry.localBackendSpawnRequest as LocalBackendSpawnRequest | null
    entry.releaseLocalBackendSlot = null
    entry.localBackendSlotKey = null
    entry.localBackendSpawnRequest = null

    if (release) {
      release()
    } else {
      request?.cancel()
    }
  }

  // `releaseSlot` must be false once `entry.process` exists: the lease has to
  // stay held until the child has actually exited (pool-spawn-coordinator
  // invariant), and teardownFailedLocalBackend releases it after that exit. A
  // pre-spawn release here would turn that post-exit release into a no-op and
  // let a successor spawn while the superseded child is still alive.
  function assertPoolEntryStillOwned(poolKey: string, entry: any, { releaseSlot = true } = {}) {
    if (localBackendLifecycle.signal.aborted || backendPool.get(poolKey) !== entry) {
      if (releaseSlot) {
        releaseLocalBackendSlot(entry)
      }

      throw new Error(`Profile backend start for "${poolKey}" was cancelled before it became ready.`)
    }
  }

  const failedLocalBackendTeardowns = new WeakMap<object, Promise<void>>()

  function teardownFailedLocalBackend(poolKey: string, entry: any): Promise<void> {
    const existing = failedLocalBackendTeardowns.get(entry)

    if (existing) {
      return existing
    }

    if (backendPool.get(poolKey) === entry) {
      backendPool.delete(poolKey)
    }

    const child = entry.process

    const teardown = releaseLocalBackendSlotAfterExit(
      () => releaseLocalBackendSlot(entry),
      async () => {
        stopBackendChild(child)
        await waitForBackendExit(child)
        releaseBackendChild(child)
      }
    )

    // Keep the settled promise in the WeakMap for the lifetime of this entry.
    // Error + exit + outer catch may all request cleanup; none may run it twice.
    failedLocalBackendTeardowns.set(entry, teardown)

    return teardown
  }

  // Spawn an additional dashboard backend pinned to a named profile. Mirrors the
  // local-spawn portion of startHermes() but without the boot-progress UI,
  // bootstrap, or remote handling (those belong to the primary backend only).
  // `opts.forceLocal` skips remote resolution entirely (the registry 'local'
  // entry means THIS machine regardless of the v1 routing table); `opts.poolKey`
  // is the backendPool key when it differs from the profile name (composite
  // registry scopes) so the exit/error cleanup evicts the right entry.
  function spawnPoolBackend(
    profile,
    entry,
    opts: { forceLocal?: boolean; poolKey?: string; unscopableRequest?: boolean } = {}
  ) {
    return localBackendLifecycle.start(() => runPoolBackendStart(profile, entry, opts))
  }

  async function runPoolBackendStart(
    profile,
    entry,
    opts: { forceLocal?: boolean; poolKey?: string; unscopableRequest?: boolean } = {}
  ) {
    const poolKey = opts.poolKey || profile

    await reapOrphanedBackendsOnce()
    profileDeletionGate.assertCanStart(profile)

    // A profile may point at its OWN remote backend (connection.json
    // `profiles[name]`), or inherit the app-wide remote (env / global settings).
    // In either case there is no local child to spawn — we just verify the
    // remote is reachable and hand back its connection descriptor. The pool
    // entry keeps `entry.process === null`, which stopPoolBackend/evict already
    // tolerate.
    const remote = opts.forceLocal ? null : await resolveRemoteBackend(profile, { poolKey })
    profileDeletionGate.assertCanStart(profile)

    if (remote) {
      await waitForHermes(remote.baseUrl, remote.token, undefined, remote.authMode, remote.headers)

      // Recorded on the entry so revalidation can probe this descriptor without
      // awaiting connectionPromise, which may still be pending for a sibling.
      entry.remoteBaseUrl = remote.baseUrl

      return {
        ...remote,
        profile,
        logs: hermesLog.slice(-80),
        ...getWindowState()
      }
    }

    // Everything below starts a LOCAL `hermes serve` child. Multiplex-only says
    // the host has exactly one, and routing (resolveProfileBackendRoute case 6)
    // keeps local profiles off this path — this is the backstop that makes the
    // pool spawn path genuinely unreachable rather than merely unused.
    // Same options object the router reads, so the guard cannot drift from it
    // (profileRouteOptions folds the per-profile SSH override into
    // profileRemoteOverride; a hand-rolled term here missed that).
    const guardRoute = profileRouteOptions(profile)

    assertNoSecondLocalBackend(poolKey, {
      isolated: guardRoute.isolatedBackend,
      primaryRemoteActive: guardRoute.primaryRemoteActive,
      profileRemoteOverride: opts.forceLocal ? false : guardRoute.profileRemoteOverride,
      unscopableRequest: opts.unscopableRequest
    })

    // Bound the slot wait BELOW the renderer's backend-boot budget (45s): once
    // the renderer has given up on this spawn, a ticket still queued for the
    // pool-idle window (10 min) would hold the pool key hostage and every
    // later click on the profile would join that stale wait. Failing here
    // surfaces the "all N slots busy" reason instead of a generic boot timeout.
    // The caller stamped entry.spawnPriority from its own request; a foreground
    // dial that joined the claim before this entry existed left a mark instead.
    if (takeForegroundSpawn(poolKey, profile)) {
      entry.spawnPriority = 'foreground'
    }

    const spawnPriority: LocalBackendSpawnPriority = spawnPriorityFrom(entry.spawnPriority)

    assertPoolEntryStillOwned(poolKey, entry)

    if (spawnPriority === 'background' && !backgroundSlotRetryBackoff.canAttempt(poolKey)) {
      throw new BackgroundSlotRetryDeferredError(profile)
    }

    // The arbiter subscribes to the actual coordinator queue, so a later
    // foreground promotion receives reclamation too, not just fresh starts.
    const spawnRequest = localBackendSpawnCoordinator.request(poolKey, {
      timeoutMs: POOL_SLOT_WAIT_MS,
      priority: spawnPriority
    })

    entry.localBackendSlotKey = poolKey
    entry.localBackendSpawnRequest = spawnRequest

    if (spawnRequest.queued) {
      rememberLog(
        `Profile backend "${profile}" waiting for a free local slot (${localBackendSpawnCoordinator.activeCount}/${poolMaxBackends()} busy, ${localBackendSpawnCoordinator.queuedCount} queued)`
      )
    }

    const cancelRequest = () => spawnRequest.cancel()
    localBackendLifecycle.signal.addEventListener('abort', cancelRequest, { once: true })

    try {
      entry.releaseLocalBackendSlot = await spawnRequest.acquired
      backgroundSlotRetryBackoff.clear(poolKey)
    } catch (error) {
      if (isBackgroundSlotWaitTimeout(error)) {
        backgroundSlotRetryBackoff.recordFailure(poolKey)
      }

      throw error
    } finally {
      localBackendLifecycle.signal.removeEventListener('abort', cancelRequest)
    }

    if (entry.localBackendSpawnRequest === spawnRequest) {
      entry.localBackendSpawnRequest = null
    }

    assertPoolEntryStillOwned(poolKey, entry)

    const token = crypto.randomBytes(32).toString('base64url')

    // Same update mutual exclusion as the primary window's waitForLocalStart
    // (#73822): pool backends spawn from the same venv, so an ungated respawn
    // during applyUpdates' critical section re-locks the venv and trips the
    // venv-blocker preflight. No boot-progress UI here — pool backends boot
    // silently for background profiles — so we only log while parked.
    {
      let poolAnnounced = false

      const clearance = await waitForUpdateClearance(updateGateDeps(), {
        signal: localBackendLifecycle.signal,
        isCancelled: () => backendPool.get(poolKey) !== entry,
        onWaitTick: reason => {
          if (!poolAnnounced) {
            poolAnnounced = true
            rememberLog(`[updates] update in progress (${reason}); deferring pool backend start for profile "${profile}"`)
          }
        },
        pollMs: UPDATE_WAIT_POLL_MS,
        timeoutMs: UPDATE_WAIT_TIMEOUT_MS
      })

      if (clearance === 'cancelled') {
        assertPoolEntryStillOwned(poolKey, entry)
      }
    }

    profileDeletionGate.assertCanStart(profile)
    assertPoolEntryStillOwned(poolKey, entry)

    // --profile wins over the inherited HERMES_HOME env (see _apply_profile_override
    // step 3 in hermes_cli/main.py), so the child re-homes to this profile.
    // --port 0: the OS assigns an ephemeral port; the child announces it on stdout.
    const backendArgs = ['--profile', profile, 'serve', '--host', '127.0.0.1', '--port', '0']

    const backend = await ensureRuntime(await resolveHermesBackend(backendArgs), () =>
      assertPoolEntryStillOwned(poolKey, entry)
    )

    // Route old runtimes (no `serve`) through the legacy `dashboard --no-open`.
    backend.args = await getBackendArgsForRuntime(backend)
    assertPoolEntryStillOwned(poolKey, entry)
    const hermesCwd = resolveHermesCwd()
    const webDist = resolveWebDist()
    const readyFile = backend.readyFile ? makeDashboardReadyFile() : null

    // Guard BEFORE the "Starting" line: a profile that only exists on a remote
    // backend (remote-primary desktop asked for a forced-local child) rejects
    // here, and logging "Starting" first left an orphaned line with no READY
    // and no exit — the exact undiagnosable burst signature in remote-gateway
    // user bundles (Aug 2026, Dash's report).
    assertLocalProfileCanStart(profile, profileDeletionGate, key =>
      directoryExists(path.join(HERMES_HOME, 'profiles', key))
    )
    rememberLog(`Starting Hermes backend for profile "${profile}" via ${backend.label}`)

    const parentStartMarker = await desktopParentStartMarker()
    const backendNonce = crypto.randomBytes(16).toString('hex')
    const parentIdentityEnv = parentWatchdogEnv(process.pid, parentStartMarker, backendNonce)
    assertPoolEntryStillOwned(poolKey, entry)

    const child = spawnOwnedBackend(
      backend.command,
      backend.args,
      hiddenWindowsChildOptions({
        cwd: hermesCwd,
        env: desktopBackendSpawnEnv(
          {
            ...process.env,
            HERMES_HOME,
            ...backend.env,
            // Pin the gateway's tool/terminal cwd to the same directory we chose for
            // the child process. Inherited TERMINAL_CWD (or a stale config bridge)
            // can still point at the install dir even when spawn cwd is home.
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

    entry.process = child
    entry.token = token
    registerLocalBackendExitFinalizer(backendPool, poolKey, entry, () => releaseLocalBackendSlot(entry))
    // Buffer stdout+stderr from the instant of spawn (#93608): an early crash's
    // traceback must survive into the claim error and the before-ready exit
    // message instead of a bare exit code. rememberLog attaches later, after
    // the claim, and would miss anything printed before it.
    const outputTail = createBackendOutputTail()
    outputTail.attach(child)

    let ready = false
    let rejectStart = null

    const startFailed = new Promise((_resolve, reject) => {
      rejectStart = reject
    })

    // Exit/error can now arrive while the ownership claim is still pending.
    startFailed.catch(() => {})

    child.once('error', error => {
      rememberLog(`Hermes backend for profile "${profile}" failed to start: ${error.message}`)
      void teardownFailedLocalBackend(poolKey, entry).catch(cleanupError => {
        rememberLog(
          `Hermes backend for profile "${profile}" cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`
        )
      })
      rejectStart?.(error)
    })
    child.once('exit', (code, signal) => {
      rememberLog(formatBackendExitLine(`Hermes backend for profile "${profile}" exited`, code, signal, outputTail))
      releaseBackendChild(child)

      if (!ready) {
        rejectStart?.(
          new Error(
            `Hermes backend for profile "${profile}" exited before it became ready (${signal || code}).${outputTail.describe()}`
          )
        )
      }
    })

    // Start watching for the READY announcement BEFORE any await (#60323):
    // stdout is already flowing into the tail, and Node streams never replay
    // consumed chunks to late listeners — a sentinel printed while
    // claimBackendChild runs would otherwise be lost forever, timing out a
    // healthy backend. The tail-buffer accessor covers any residual gap.
    const portAnnouncement = waitForDashboardPortAnnouncement(child, {
      bufferedOutput: () => outputTail.text(),
      describeOutputTail: () => outputTail.describe(),
      readyFile
    })

    portAnnouncement.catch(() => {})
    await claimBackendChild(child, `${backend.command} ${backend.args.join(' ')}`, profile, backendNonce, outputTail)
    assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

    child.stdout.on('data', rememberLog)
    child.stderr.on('data', rememberLog)

    const port = await Promise.race([portAnnouncement, startFailed])
    assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

    if (readyFile) {
      fs.unlink(readyFile, () => {})
    }

    entry.port = port

    const baseUrl = `http://127.0.0.1:${port}`
    await Promise.race([waitForHermes(baseUrl, token), startFailed])
    assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })
    ready = true

    const authToken = await adoptServedDashboardToken(baseUrl, token, {
      childAlive: () => child.exitCode === null && !child.killed,
      label: `Hermes backend for profile "${profile}"`,
      rememberLog
    })

    assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

    entry.token = authToken

    // Verify the WebSocket session token before declaring backend ready.
    // HTTP /api/status can pass while WS auth fails (separate transport, separate guards).
    const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`
    const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket })
    assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

    if (!wsProbe.ok) {
      throw new Error(
        `Hermes backend for profile "${profile}" is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: ${wsProbe.reason}`
      )
    }

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
  }

  // Bounded, deduplicated pool teardown (see pool-stop.ts): every stop path —
  // idle reaper, LRU eviction, profile delete/rename, quit — shares one
  // in-flight stop per key and retains the process handle until the bounded
  // SIGTERM -> SIGKILL escalation in waitForBackendExit() resolves. Previously
  // SIGTERM + immediate entry delete dropped the handle and a slow child
  // survived detached under PID 1.
  const poolStopper = createPoolStopper({
    pool: backendPool,
    stopChild: child => stopBackendChild(child),
    waitForExit: child => waitForBackendExit(child),
    // Remote / SSH-isolated pool entries keep `process: null`. Child exit is
    // immediate; hold the same in-flight fence through bootstrap drain + SSH
    // teardown so a reconnect cannot publish into a dying scope (#106935).
    afterStop: async key => {
      try {
        await sshBootstrapCoordinator.cancelAndWait(key, () => teardownSshConnection(key))
      } catch (err) {
        // The idle reaper calls stopPoolBackend un-awaited; a failed SSH teardown
        // must not surface as an unhandled rejection or block the pool fence.
        sshRememberLog(`[ssh-teardown] ${key}: ${String(err)}`)
      }
    }
  })

  function stopPoolBackend(profile: string): Promise<void> {
    const entry = backendPool.get(profile)

    const stopping = releaseLocalBackendSlotAfterExit(
      () => releaseLocalBackendSlot(entry),
      () => poolStopper.stop(profile)
    )

    // Fire-and-forget callers still need diagnostics; awaiters receive the
    // rejection, while physical ownership and the exit finalizer remain live.
    void stopping.catch(error => {
      rememberLog(`Profile backend "${profile}" stop failed: ${error instanceof Error ? error.message : String(error)}`)
    })

    return stopping
  }

  // Tell every window the pooled backend under `poolKey` is being retired so the
  // renderer parks that scope (wantOpen=false) instead of redialing into the
  // slot it just vacated. Fired BEFORE the SIGTERM (pool-retire.ts contract).
  function broadcastPoolBackendRetiring(poolKey: string) {
    for (const win of BrowserWindow.getAllWindows()) {
      const { webContents } = win

      if (webContents && !webContents.isDestroyed()) {
        webContents.send('hermes:pool:retiring', { poolKey })
      }
    }
  }

  const poolRetirer = createPoolRetirer({
    pool: backendPool,
    coordinator: localBackendSpawnCoordinator,
    ...createPoolRetirementClient(fetchJson),
    stopBackend: stopPoolBackend,
    onRetiring: broadcastPoolBackendRetiring,
    log: rememberLog
  })

  return { releaseLocalBackendSlot, teardownFailedLocalBackend, spawnPoolBackend, poolStopper, stopPoolBackend, poolRetirer }
}
