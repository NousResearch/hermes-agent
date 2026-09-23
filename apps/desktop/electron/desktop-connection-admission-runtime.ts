import {
  normAuthMode,
  normalizeSshConfig,
  resolveProfileBackendRoute,
  unscopableMutatingRequest
} from './connection-config'
import {
  backendScopeKey,
  registrySourceOwnsPrimaryBackend,
  resolveRegistryLocalRoute,
  reuseMatchingPrimarySshBackend
} from './connection-registry'
import { assertNotPassiveSpawn } from './host-backend-singleton'
import type { LocalBackendSpawnPriority } from './pool-spawn-coordinator'
import { ensureHealthyPooledRemoteBackendForDispatch } from './remote-liveness'

// Main owns the live pools, gates and lifecycle. Compose after poolStopper and
// poolRetirer exist so admission always sees their current initialized state.
export interface DesktopConnectionAdmissionRuntimeDeps {
  backendPool: Map<string, any>
  bootstrapSshConnection: any
  buildRemoteConnection: any
  decryptDesktopSecret: any
  effectiveSshConfigFingerprint: any
  evictLruPoolBackends: any
  fetchJsonForBackend: any
  getWindowState: any
  globalRemoteActive: any
  hermesLog: string[]
  localBackendLifecycle: any
  logPoolSpawnFailure: any
  managedConnectionUpdateGate: any
  poolMaxBackends: any
  poolRetirer: any
  poolStopper: any
  primaryProfileKey: any
  profileDeletionGate: any
  profileHasRemoteOverride: any
  profileRouteOptions: any
  promotePoolEntry: any
  readDesktopConnectionsRegistry: any
  registryDispatchRevalidation: any
  rememberLog: any
  setWslBridgeProfileState: any
  spawnPoolBackend: any
  spawnPriorityFrom: any
  sshBootstrapCoordinator: any
  startHermes: any
  startPoolIdleReaper: any
  stopPoolBackend: any
  teardownFailedLocalBackend: any
  teardownSshConnection: any
  waitForHermes: any
}

export function createDesktopConnectionAdmissionRuntime(deps: DesktopConnectionAdmissionRuntimeDeps) {
  const {
    backendPool,
    bootstrapSshConnection,
    buildRemoteConnection,
    decryptDesktopSecret,
    effectiveSshConfigFingerprint,
    evictLruPoolBackends,
    fetchJsonForBackend,
    getWindowState,
    globalRemoteActive,
    hermesLog,
    localBackendLifecycle,
    logPoolSpawnFailure,
    managedConnectionUpdateGate,
    poolMaxBackends,
    poolRetirer,
    poolStopper,
    primaryProfileKey,
    profileDeletionGate,
    profileHasRemoteOverride,
    profileRouteOptions,
    promotePoolEntry,
    readDesktopConnectionsRegistry,
    registryDispatchRevalidation,
    rememberLog,
    setWslBridgeProfileState,
    spawnPoolBackend,
    spawnPriorityFrom,
    sshBootstrapCoordinator,
    startHermes,
    startPoolIdleReaper,
    stopPoolBackend,
    teardownFailedLocalBackend,
    teardownSshConnection,
    waitForHermes
  } = deps

  // Resolve a backend connection for the given profile, per the routing table in
  // resolveProfileBackendRoute(). An empty / unknown profile resolves to the
  // primary, so legacy callers are unchanged.
  async function ensureBackend(
    profile,
    opts: {
      passive?: boolean
      request?: { method?: string; path?: string }
      spawnPriority?: LocalBackendSpawnPriority
    } = {}
  ) {
    localBackendLifecycle.assertCanStart()
    const key = profile && String(profile).trim() ? String(profile).trim() : primaryProfileKey()
    const spawnPriority = spawnPriorityFrom(opts.spawnPriority)
    poolRetirer.assertCanOpen(key, spawnPriority)
    const passive = Boolean(opts.passive)

    profileDeletionGate.assertCanStart(key)

    // The REQUEST is part of the routing decision (case 5/6): resolving without
    // it would collapse a profile onto the shared backend that the caller's
    // resolveProfileApiRequest deliberately kept pooled, and the unscopable
    // destructive write would execute against the primary's home after all.
    const routeOpts = profileRouteOptions(key, opts.request)
    const route = resolveProfileBackendRoute(key, routeOpts)

    if (route.backend === 'primary') {
      const connection = await startHermes()
      setWslBridgeProfileState(key, connection.mode !== 'remote')

      // A shared backend still owes the caller its profile scope, so renderer-side
      // WebSocket, filesystem, and cache routing target the selected profile.
      // `sharedPrimary` marks this as the shared-primary route: pooled backends
      // also carry `profile`, so only this descriptor gets the flag. The
      // unshared primary carries its own key too: a profile-less descriptor
      // reads as "default" downstream, which breaks per-source profile memory
      // (the primary IS "default" only when it actually booted as default).
      return route.descriptorProfile
        ? { ...connection, profile: route.descriptorProfile, sharedPrimary: true }
        : { ...connection, profile: key }
    }

    // A backend for this key may still be dying (idle reap, LRU eviction, a
    // just-finished delete). Wait for its bounded exit before reusing or
    // spawning, so two children never share one profile's HERMES_HOME.
    const stopping = poolStopper.inFlight(key)

    if (stopping) {
      await stopping
    }

    const existing = backendPool.get(key)

    if (existing) {
      if (!passive) {
        existing.lastActiveAt = Date.now()
      }

      if (spawnPriority === 'foreground') {
        promotePoolEntry(existing)
      }

      const connection = await existing.connectionPromise
      setWslBridgeProfileState(key, connection.mode !== 'remote')

      return connection
    }

    assertNotPassiveSpawn(passive, key)
    // The hard slot is released only after the evicted child exits. Wait for
    // that teardown before entering the spawn queue; otherwise a successful
    // LRU choice still leaves this wake racing the old child for 30 seconds.
    await evictLruPoolBackends(poolMaxBackends() - 1)

    const entry = {
      process: null,
      port: null,
      token: null,
      connectionPromise: null,
      lastActiveAt: Date.now(),
      remoteBaseUrl: null,
      releaseLocalBackendSlot: null,
      localBackendSlotKey: null,
      localBackendSpawnRequest: null,
      spawnPriority
    }

    entry.connectionPromise = spawnPoolBackend(key, entry, {
      unscopableRequest: unscopableMutatingRequest(routeOpts)
    }).catch(async error => {
      // Land the failure in desktop.log: without this a spawn that dies before
      // its child exists (guard rejection, runtime resolution) leaves no trace
      // beyond renderer-side rejections users never see in a bundle.
      logPoolSpawnFailure(`"${key}"`, error)

      await teardownFailedLocalBackend(key, entry)
      throw error
    })
    backendPool.set(key, entry)
    startPoolIdleReaper()

    const connection = await entry.connectionPromise
    setWslBridgeProfileState(key, connection.mode !== 'remote')

    return connection
  }

  // ── Registry-scoped backends (multi-connection, PR 2 of the campaign) ──────
  // Resolve a backend for (connectionId, profile) against the v2 registry.
  // The LOCAL connection routes through ensureBackend() when the v1 route is
  // itself local (so every single-source path stays byte-identical), and forces
  // a genuinely-local child when the v1 mode says remote; non-local connections
  // pool under the composite key from backendScopeKey() and reuse the same pool
  // entry lifecycle (LRU, idle reaper, touch) as per-profile local backends.
  async function ensureRegistryBackend(
    connectionId,
    profile,
    managedUpdateCorrelation = '',
    opts: { passive?: boolean; spawnPriority?: LocalBackendSpawnPriority } = {}
  ) {
    const spawnPriority = spawnPriorityFrom(opts.spawnPriority)
    const passive = Boolean(opts.passive)
    const registry = readDesktopConnectionsRegistry()
    const id = String(connectionId || '').trim() || registry.primary
    const source = registry.connections.find(c => c.id === id)

    if (!source) {
      throw new Error(`No connection with id "${id}".`)
    }

    if (source.kind === 'ssh') {
      managedConnectionUpdateGate.assertCanDial(id, managedUpdateCorrelation)
    }

    const profileKey = String(profile ?? '').trim() || 'default'
    let resolvedRegistrySshConfig
    let registryEffectiveFingerprintPromise: null | Promise<string> = null

    const resolveRegistrySshConfig = () => {
      if (source.kind !== 'ssh') {
        return null
      }

      if (!resolvedRegistrySshConfig) {
        resolvedRegistrySshConfig = normalizeSshConfig({
          mode: 'ssh',
          host: source.host,
          user: source.user,
          port: source.port,
          keyPath: source.keyPath,
          remoteHermesPath: source.remoteHermesPath,
          remoteProfile: source.remoteProfile || (profileKey === 'default' ? '' : profileKey)
        })
      }

      return resolvedRegistrySshConfig
    }

    const resolveRegistryEffectiveFingerprint = () => {
      if (!registryEffectiveFingerprintPromise) {
        const sshConfig = resolveRegistrySshConfig()

        registryEffectiveFingerprintPromise = sshConfig
          ? effectiveSshConfigFingerprint(sshConfig)
          : Promise.reject(new Error(`SSH connection "${source.label}" has no host configured.`))
      }

      return registryEffectiveFingerprintPromise
    }

    // The v2 registry is migrated from (but intentionally coexists with) the
    // v1 primary connection config. Reuse the already-booted primary descriptor
    // when both identities match; otherwise a default-profile registry request
    // opens a second SSH dashboard under a different scope and the competing
    // lifecycle probes repeatedly tear down each other's tunnel.
    const primary = await reuseMatchingPrimarySshBackend({
      connectionId: id,
      effectiveFingerprint: resolveRegistryEffectiveFingerprint,
      ensurePrimary: () => ensureBackend(profile, { passive, spawnPriority }),
      profile,
      registry,
      source
    })

    if (primary) {
      return {
        ...primary,
        profile: profileKey,
        connectionId: id
      }
    }

    // The v1 primary and the registry primary can describe the same remote
    // backend beyond the SSH-fingerprint path above (cloud/url remotes have no
    // ssh -G identity). Reuse the already-running primary when the registry
    // resolves its live descriptor back to this exact source id; otherwise one
    // Desktop window starts two isolated servers whose transient runtime ids
    // are not interchangeable.
    if (id === registry.primary && source.kind !== 'local' && source.kind !== 'ssh') {
      const primaryDescriptor = await ensureBackend(profile, { passive })

      if (registrySourceOwnsPrimaryBackend(registry, id, primaryDescriptor)) {
        return {
          ...primaryDescriptor,
          profile: profileKey,
          connectionId: id,
          sharedRemote: true
        }
      }
    }

    if (source.kind === 'local') {
      // The registry's 'local' entry means THIS machine's runtime — always.
      // ensureBackend() follows the v1 routing table, which resolves to a
      // REMOTE descriptor when the v1 global mode is remote (or the profile
      // has its own remote override). A migrated remote-mode user would then
      // see the roster's "This device" rows enumerate + dial the remote box
      // (every profile duplicated, -slug handles forced). Delegate only when
      // the v1 route is genuinely local; otherwise spawn/reuse a forced-local
      // child pooled under the composite 'conn:local::<profile>' key so it
      // can't collide with the v1 remote descriptor cached at the bare key.
      profileDeletionGate.assertCanStart(profileKey)

      const localRoute = resolveRegistryLocalRoute(profileKey, {
        globalRemote: globalRemoteActive(),
        profileRemoteOverride: Boolean(profileHasRemoteOverride(profileKey))
      })

      if (localRoute.delegate) {
        return ensureBackend(profile, { passive, spawnPriority })
      }

      const stoppingLocal = poolStopper.inFlight(localRoute.poolKey)

      if (stoppingLocal) {
        await stoppingLocal
      }

      const existingLocal = backendPool.get(localRoute.poolKey)

      if (existingLocal) {
        if (!passive) {
          existingLocal.lastActiveAt = Date.now()
        }

        if (spawnPriority === 'foreground') {
          promotePoolEntry(existingLocal)
        }

        return existingLocal.connectionPromise
      }

      assertNotPassiveSpawn(passive, localRoute.poolKey)
      await evictLruPoolBackends(poolMaxBackends() - 1)

      const localEntry = {
        process: null,
        port: null,
        token: null,
        connectionPromise: null,
        lastActiveAt: Date.now(),
        remoteBaseUrl: null,
        releaseLocalBackendSlot: null,
        localBackendSlotKey: null,
        localBackendSpawnRequest: null,
        spawnPriority
      }

      localEntry.connectionPromise = spawnPoolBackend(profileKey, localEntry, {
        forceLocal: true,
        poolKey: localRoute.poolKey
      }).catch(async error => {
        // Same trace rule as the v1 pool path: a forced-local child whose spawn
        // rejects before the child exists must still land in desktop.log.
        logPoolSpawnFailure(`"${profileKey}" (forced-local)`, error)

        await teardownFailedLocalBackend(localRoute.poolKey, localEntry)
        throw error
      })
      backendPool.set(localRoute.poolKey, localEntry)
      startPoolIdleReaper()

      return localEntry.connectionPromise
    }

    const key = backendScopeKey(id, profile)
    const existing = backendPool.get(key)

    if (existing) {
      if (!passive) {
        existing.lastActiveAt = Date.now()
      }

      const connectionPromise = existing.connectionPromise

      // A remote process can die while its local SSH forward stays LISTENing.
      // Validate the exact cached descriptor at dispatch time; background
      // revalidation is renderer-driven and may never run while the Bots pane is
      // closed. Concurrent clicks share one retire/reconnect sequence.
      return registryDispatchRevalidation.run(connectionPromise, () =>
        ensureHealthyPooledRemoteBackendForDispatch({
          connectionPromise,
          currentConnectionPromise: () => backendPool.get(key)?.connectionPromise || null,
          probe: (connection, requestPath, options) => fetchJsonForBackend(connection, requestPath, options),
          reconnect: () => ensureRegistryBackend(id, profile, '', { passive }),
          retire: async (error: any) => {
            // A late failure from an old descriptor must never tear down a newer
            // entry that another caller has already installed.
            if (backendPool.get(key) !== existing) {
              return
            }

            rememberLog(
              `Pooled remote backend "${key}" failed its dispatch probe (${error?.message || error}); reconnecting on demand.`
            )
            await stopPoolBackend(key)

            if (source.kind === 'ssh') {
              await sshBootstrapCoordinator.cancelAndWait(key)
              await teardownSshConnection(key)
            }
          }
        })
      )
    }

    assertNotPassiveSpawn(passive, key)
    await evictLruPoolBackends(poolMaxBackends() - 1)

    const entry = {
      process: null,
      port: null,
      token: null,
      connectionPromise: null,
      lastActiveAt: Date.now(),
      remoteBaseUrl: null
    }

    entry.connectionPromise = connectRegistryBackend(
      source,
      profile,
      key,
      entry,
      resolveRegistrySshConfig(),
      source.kind === 'ssh' ? resolveRegistryEffectiveFingerprint() : null,
      managedUpdateCorrelation
    ).catch(error => {
      if (backendPool.get(key) === entry) {
        backendPool.delete(key)
      }

      throw error
    })
    backendPool.set(key, entry)
    startPoolIdleReaper()

    return entry.connectionPromise
  }

  // Dial a non-local registry connection for one profile. Never spawns a local
  // child (entry.process stays null — stopPoolBackend/evict already tolerate
  // that shape from remote per-profile overrides).
  async function connectRegistryBackend(
    source,
    profile,
    key,
    poolEntry,
    resolvedSshConfig?,
    resolvedEffectiveFingerprint?: null | Promise<string>,
    managedUpdateCorrelation = '',
    tokenPersistenceSource = ''
  ) {
    const profileKey = String(profile ?? '').trim() || 'default'

    if (source.kind === 'ssh') {
      // The composite key doubles as the ssh scope so each (connection, profile)
      // pair owns its own tunnel + remote dashboard; the profile that re-homes
      // the REMOTE process is the entry's remoteProfile or the requested one —
      // never the composite string.
      const sshConfig = resolvedSshConfig

      if (!sshConfig) {
        throw new Error(`SSH connection "${source.label}" has no host configured.`)
      }

      const connection = await bootstrapSshConnection(
        key,
        sshConfig,
        decryptDesktopSecret(source.token),
        tokenPersistenceSource || `registry:${source.id}`,
        resolvedEffectiveFingerprint ? await resolvedEffectiveFingerprint : undefined,
        {
          managedScope: 'pool',
          managedUpdateCorrelation,
          poolKey: key,
          registryConnectionId: source.id
        }
      )

      poolEntry.remoteBaseUrl = connection.baseUrl

      // SSH backends always run on a remote host — their POSIX paths can never
      // be opened through this machine's wsl.exe. Register the profile as
      // bridge-inactive so file panels/dialogs don't spawn wsl.exe (visible
      // black window on WSL-less Windows) for remote paths. The v1 path does
      // this in ensureBackend(); the registry SSH path was missing it.
      setWslBridgeProfileState(profileKey, false)

      return {
        ...connection,
        profile: profileKey,
        connectionId: source.id,
        // The remote process runs as this profile; the desktop-side profile key
        // is only the routing label. hermes:api uses it to translate explicit
        // self-profile query filters into the backend's namespace.
        remoteProfile: sshConfig.remoteProfile || '',
        logs: hermesLog.slice(-80),
        ...getWindowState()
      }
    }

    // remote / cloud: one gateway host serves every profile of that source,
    // scoped per request — the descriptor carries the profile + connectionId so
    // renderer-side WS minting and REST scoping target the right agent.
    const token = source.authMode === 'oauth' ? null : decryptDesktopSecret(source.token)

    const connection = await buildRemoteConnection(
      source.url,
      normAuthMode(source.authMode),
      token,
      `registry:${source.id}`,
      undefined,
      source.kind === 'cloud' ? 'cloud' : 'url',
      undefined,
      source.headers
    )

    await waitForHermes(connection.baseUrl, connection.token, undefined, connection.authMode, connection.headers)
    poolEntry.remoteBaseUrl = connection.baseUrl

    // Remote/cloud backends live on another host too — disable the WSL path
    // bridge for their profiles for the same reason as the SSH branch above.
    setWslBridgeProfileState(profileKey, false)

    return {
      ...connection,
      profile: profileKey,
      connectionId: source.id,
      // One host, many profiles: REST paths must carry ?profile= (same contract
      // as the global-remote shared-primary route).
      sharedRemote: true,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    }
  }

  return { ensureBackend, ensureRegistryBackend, connectRegistryBackend }
}
