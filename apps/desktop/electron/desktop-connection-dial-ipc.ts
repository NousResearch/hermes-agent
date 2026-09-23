import { backendScopeKey, parseBackendScopeKey, resolvedConnectionId } from './connection-registry'
import { type DesktopProfileRoute, resolveDesktopConnectionRequest } from './desktop-profile'
import { type LocalBackendSpawnPriority } from './pool-spawn-coordinator'
import { revalidatePooledRemoteBackends, revalidateRemoteConnection, revalidateSuspectPooledRemoteBackends } from './remote-liveness'

interface DesktopConnectionDialIpcDeps {
  ipcMain: any
  windowConnectionRoutes: any
  applySpawnPriority: any
  backendConnectionState: any
  backendDialClaims: any
  backendPool: any
  ensureBackend: any
  ensureRegistryBackend: any
  fetchJsonForBackend: any
  primaryProfileKey: any
  readDesktopConnectionsRegistry: any
  rememberLog: any
  remoteLiveness: any
  remoteRevalidation: any
  resetHermesConnection: any
  resetPreviewReach: any
  spawnPriorityFrom: any
  sshBootstrapCoordinator: any
  sshScopeKey: any
  stopPoolBackend: any
  teardownSshConnection: any
}

export function registerDesktopConnectionDialIpc(deps: DesktopConnectionDialIpcDeps) {
  const {
    ipcMain,
    windowConnectionRoutes,
    applySpawnPriority,
    backendConnectionState,
    backendDialClaims,
    backendPool,
    ensureBackend,
    ensureRegistryBackend,
    fetchJsonForBackend,
    primaryProfileKey,
    readDesktopConnectionsRegistry,
    rememberLog,
    remoteLiveness,
    remoteRevalidation,
    resetHermesConnection,
    resetPreviewReach,
    spawnPriorityFrom,
    sshBootstrapCoordinator,
    sshScopeKey,
    stopPoolBackend,
    teardownSshConnection
  } = deps

ipcMain.handle('hermes:connection', async (event, profile, extra) => {
  const route = resolveDesktopConnectionRequest(
    profile,
    windowConnectionRoutes.get(event.sender.id),
    primaryProfileKey()
  )

  return connectDesktopProfileRoute(route, spawnPriorityFrom(extra?.priority))
})

async function connectDesktopProfileRoute(
  route: DesktopProfileRoute,
  spawnPriority: LocalBackendSpawnPriority = 'foreground'
) {
  // Coalesce concurrent renderer dials for one profile scope (#90812): the
  // renderer-side reconnect lock is per-window, so two windows waking at once
  // both land here. The claim key mirrors ensureBackend()'s own profile
  // normalization so every spelling of the primary coalesces onto one dial.
  const scopeKey = backendScopeKey(route.connectionId, route.profile)
  const clearSpawnPriority = applySpawnPriority(scopeKey, spawnPriority)

  let connection

  try {
    connection = await backendDialClaims.run(scopeKey, () =>
      route.connectionId
        ? ensureRegistryBackend(route.connectionId, route.profile, '', { spawnPriority })
        : ensureBackend(route.profile, { spawnPriority })
    )
  } finally {
    clearSpawnPriority()
  }

  if (route.connectionId) {
    return { ...connection, connectionId: route.connectionId, registryScoped: true }
  }

  const connectionId = resolvedConnectionId(readDesktopConnectionsRegistry(), connection)

  return connectionId ? { ...connection, connectionId } : connection
}

// Registry-scoped variant: resolve a backend for (connectionId, profile).
// connectionId '' / 'local' / the registry primary all behave sensibly; the
// local kind delegates to ensureBackend when the v1 route is local, and
// forces a genuinely-local child when the v1 global mode is remote (the
// registry 'local' entry always means this machine).
ipcMain.handle('hermes:connection:for', async (_event, payload) => {
  const { connectionId, profile, priority } = payload && typeof payload === 'object' ? (payload as any) : ({} as any)
  const registry = readDesktopConnectionsRegistry()
  const id = String(connectionId || '').trim() || registry.primary
  const spawnPriority = spawnPriorityFrom(priority)

  return connectDesktopProfileRoute(
    { connectionId: id, profile: String(profile ?? '').trim() || 'default' },
    spawnPriority
  )
})

const windowConnectionRouteOwners = new Set<number>()

function recordWindowConnectionRoute(sender: Electron.WebContents, route: unknown) {
  const id = sender.id
  const previous = windowConnectionRoutes.get(id)
  const next = windowConnectionRoutes.set(id, route)

  if (
    previous?.connectionId !== next?.connectionId ||
    previous?.profile !== next?.profile ||
    previous?.registryScoped !== next?.registryScoped
  ) {
    void resetPreviewReach(id)
  }

  if (!windowConnectionRouteOwners.has(id)) {
    windowConnectionRouteOwners.add(id)
    sender.once('destroyed', () => {
      windowConnectionRoutes.delete(id)
      windowConnectionRouteOwners.delete(id)
      void resetPreviewReach(id)
    })
  }
}

ipcMain.on('hermes:connection:active-route', (event, route) => recordWindowConnectionRoute(event.sender, route))
// Reconnect-after-wake recovery. A REMOTE primary backend has no child process,
// so the 'exit'/'error' handlers that would clear a dead connection promise never
// fire — once the remote becomes unreachable across a sleep/wake the renderer
// re-dials the same dead descriptor forever and the composer stays stuck on
// "Starting Hermes…". Before the renderer's backoff loop reconnects, it asks us
// to confirm the cached PRIMARY backend is still reachable; if a remote one is
// not, we drop the cache so the next getConnection() rebuilds it. Local backends
// self-heal via their child 'exit' handler, so we never touch them here.
ipcMain.handle('hermes:connection:revalidate', async () => {
  const connectionPromise = backendConnectionState.getPromise()

  if (!connectionPromise) {
    await revalidatePool()

    return { ok: true, rebuilt: false }
  }

  // Main and every session pop-out have their own renderer reconnect loop but
  // share this primary connection. Coalesce simultaneous requests so one outage
  // produces one failure observation rather than exhausting the whole streak.
  return remoteRevalidation.run(connectionPromise, async () => {
    const [result] = await Promise.all([
      revalidateRemoteConnection({
        connectionPromise,
        currentConnectionPromise: () => backendConnectionState.getPromise(),
        log: rememberLog,
        probe: (connection, path, options) => fetchJsonForBackend(connection, path, options),
        resetConnection: () => resetHermesConnection({ soft: true }),
        tracker: remoteLiveness
      }),
      revalidatePool()
    ])

    // A rebuilt SSH connection must also tear down its tunnel/master before the
    // renderer re-dials (which only happens after this handler resolves), so the
    // fresh bootstrap can't reattach to a dying transport.
    if (result.rebuilt) {
      const conn = await connectionPromise.catch(() => null)

      if (conn?.remoteKind === 'ssh') {
        const profile = primaryProfileKey()
        await sshBootstrapCoordinator.cancelAndWait(sshScopeKey(profile))
        await teardownSshConnection(profile)
      }
    }

    return result
  })
})

// Pooled remote descriptors get the same treatment as the primary: they have no
// child process to signal their host's death, and the renderer's keepalive touch
// spares them from the idle reaper, so nothing else can retire a dead one.
function revalidatePool() {
  return revalidatePooledRemoteBackends({
    entries: backendPool.entries(),
    log: rememberLog,
    probe: (connection, path, options) => fetchJsonForBackend(connection, path, options),
    stopBackend: stopPoolBackend,
    tracker: remoteLiveness
  })
}

// Re-dial one retired pool key through the SAME claim-guarded ensure path a
// renderer dial takes (#90812), so a resume-driven rebuild and a concurrent
// renderer reconnect coalesce onto one spawn instead of racing.
function redialPoolBackendAfterResume(poolKey: string) {
  const { connectionId, profile } = parseBackendScopeKey(poolKey)

  return backendDialClaims.run(poolKey, () =>
    connectionId ? ensureRegistryBackend(connectionId, profile) : ensureBackend(profile)
  )
}

// Identity for coalescing post-resume sweeps in the shared revalidation
// coordinator: overlapping resume/unlock/network-restore kicks join the one
// in-flight sweep instead of stacking probes.
const suspectPoolSweepScope = {}

// Sleep/wake recovery for POOLED remote/SSH backends (#93910). The primary
// renderer socket already has wake-path probe/reconnect nudges, but pooled
// descriptors (Bots pane, secondary connections) kept serving dead SSH
// tunnels after macOS resume: no child 'exit' fires for a remote, and the
// background failure-streak policy takes several rounds to drop one. On
// resume every pooled remote is suspect — probe each once (bounded), tear
// down the dead ones (pool entry + SSH bootstrap + tunnel/master) and rebuild
// them through the claim-guarded dial path.
function revalidateSuspectPoolAfterResume() {
  return remoteRevalidation.run(suspectPoolSweepScope, () =>
    revalidateSuspectPooledRemoteBackends({
      entries: backendPool.entries(),
      log: rememberLog,
      probe: (connection, path, options) => fetchJsonForBackend(connection, path, options),
      rebuild: poolKey => redialPoolBackendAfterResume(poolKey),
      retire: async poolKey => {
        await stopPoolBackend(poolKey)
        // The pool key doubles as the SSH scope for registry SSH backends and
        // resolves through sshScopeKey() for bare-profile remotes; both
        // teardown calls no-op when the scope holds no SSH state.
        await sshBootstrapCoordinator.cancelAndWait(poolKey)
        await teardownSshConnection(poolKey)
      },
      tracker: remoteLiveness
    })
  )
}


  return { connectDesktopProfileRoute, recordWindowConnectionRoute, revalidateSuspectPoolAfterResume }
}
