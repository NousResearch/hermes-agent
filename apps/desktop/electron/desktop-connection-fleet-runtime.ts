import crypto from 'node:crypto'

import type { RosterProfileMetadata } from './connection-registry'

// Register the fleet surface explicitly after the shell's live backend and
// managed SSH update runtimes are composed. Importing this module is inert.
export function registerDesktopConnectionFleetIpc(deps: any) {
  const {
    applyUpdates,
    backendDialClaims,
    backendPool,
    backendScopeKey,
    buildAgentRoster,
    buildGatewayWsUrlWithTicket,
    connectionInstallIds,
    createRegistryGatewayWsUrlHandler,
    createSshProbeConnection,
    ensureRegistryBackend,
    fetchRosterSourceData,
    gatewayWsUrlIpcResult,
    getJsonForBackend,
    globalRemoteActive,
    ipcMain,
    managedConnectionUpdateGate,
    managedConnectionUpdates,
    mintGatewayWsTicket,
    normalizeSshConfig,
    postJsonForBackend,
    primaryProfileKey,
    profileHasRemoteOverride,
    readDesktopConnectionsRegistry,
    refusedManagedSshUpdate,
    rememberRemoteWsHeaders,
    rememberSshEnumeration,
    remoteLifecycle,
    resolveRegistryLocalRoute,
    rosterSourceEnumerationTimeoutMs,
    shouldDeferLocalEnumeration,
    shouldRetrySshInventory,
    sshInventoryAttemptedAt,
    sshRememberLog,
    sshRosterCache,
    updateEligibility,
    updateManagedSshConnection
  } = deps

  const SSH_INVENTORY_RETRY_MS = 60_000

  // Stable backend identity per registered connection: the `install_id` its
  // /api/status reports (absent on backends older than the field). Enumeration
  // runs on the ~5s Bot Mode roster poll and only hits /api/profiles, so the
  // status probe is cached per connection with a TTL to avoid doubling roster
  // traffic; the Test button refreshes it eagerly. A missing id simply bypasses
  // the same-backend roster collapse — fully backward compatible.
  const INSTALL_ID_TTL_MS = 5 * 60_000
  const INSTALL_ID_NEGATIVE_TTL_MS = 60_000

  function rememberConnectionInstallId(connectionId: string, statusBody: any) {
    const raw = statusBody && typeof statusBody === 'object' ? statusBody.install_id : undefined
    const id = typeof raw === 'string' && raw.trim() ? raw.trim() : undefined
    connectionInstallIds.set(connectionId, { id, ts: Date.now() })

    return id
  }

  async function probeConnectionInstallId(connectionId: string, descriptor: any): Promise<string | undefined> {
    const cached = connectionInstallIds.get(connectionId)

    if (cached && Date.now() - cached.ts < (cached.id ? INSTALL_ID_TTL_MS : INSTALL_ID_NEGATIVE_TTL_MS)) {
      return cached.id
    }

    try {
      const status: any = await getJsonForBackend(descriptor, '/api/status', { timeoutMs: 8_000 })

      return rememberConnectionInstallId(connectionId, status)
    } catch {
      // Keep any previously-known id (identity is stable; a transient fetch
      // failure must not flap the roster collapse), but do not cache a MISS
      // over it.
      if (cached?.id) {
        return cached.id
      }

      connectionInstallIds.set(connectionId, { id: undefined, ts: Date.now() })

      return undefined
    }
  }

  async function probeSshProfileInventory(connection) {
    if (
      !shouldRetrySshInventory(
        sshRosterCache.has(connection.id),
        sshInventoryAttemptedAt.get(connection.id),
        Date.now(),
        SSH_INVENTORY_RETRY_MS
      )
    ) {
      return
    }

    sshInventoryAttemptedAt.set(connection.id, Date.now())

    const sshConfig = normalizeSshConfig({
      mode: 'ssh',
      host: connection.host,
      user: connection.user,
      port: connection.port,
      keyPath: connection.keyPath,
      remoteHermesPath: connection.remoteHermesPath
    })

    if (!sshConfig) {
      return
    }

    const ssh = createSshProbeConnection(
      { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
      { rememberLog: sshRememberLog }
    )

    try {
      await ssh.open()
      const profiles = await remoteLifecycle.listRemoteHermesProfiles(ssh)

      if (profiles.length > 0) {
        sshRosterCache.set(connection.id, profiles)
      }

      // Backend identity, on the session we already have open: without it an ssh connection has no
      // install id at all, so two addresses for one machine never collapse into one roster row
      // (#88828 wired this for remote/local only, through /api/status).
      connectionInstallIds.set(connection.id, {
        id: await remoteLifecycle.readRemoteInstallId(ssh),
        ts: Date.now()
      })
    } catch (error: any) {
      sshRememberLog(`[ssh] profile inventory failed for ${connection.id}: ${error?.message || error}`)
    } finally {
      try {
        await ssh.close()
      } catch {
        void 0
      }
    }
  }

  async function enumerateRegistryAgentSources(registry = readDesktopConnectionsRegistry()) {
    // One dead source must not wedge the whole roster: ensureRegistryBackend on
    // an unreachable remote can block up to the 45s readiness timeout, and the
    // Bot Mode poll runs every 5s — each poll queued behind the dead dial, so
    // the renderer painted stale rows for the entire outage (and the roster IPC
    // hung >30s in live repro). Bound each source's enumeration; a timeout is
    // reported like any other unreachable source and retried on the next poll.
    const withEnumerationDeadline = async <T>(work: Promise<T>, perSourceTimeoutMs: number): Promise<T> => {
      let timer: ReturnType<typeof setTimeout> | null = null

      try {
        return await Promise.race([
          work,
          new Promise<never>((_resolve, reject) => {
            timer = setTimeout(() => reject(new Error('roster enumeration timed out')), perSourceTimeoutMs)
          })
        ])
      } finally {
        if (timer !== null) {
          clearTimeout(timer)
        }
      }
    }

    return Promise.all(
      registry.connections.map(async connection => {
        let raw: {
          connection: typeof connection
          error?: string
          installId?: string
          profiles: null | string[]
          profileMetadata?: Record<string, RosterProfileMetadata>
        }

        try {
          // SSH roster listing must never spawn a dashboard. A stale
          // sshConnections key used to fall into ensureRegistryBackend and
          // respawn Spark/Mini every Bot Mode poll (~5s), then the mux died
          // (ECONNRESET / liveness probe drop).
          if (connection.kind === 'ssh') {
            await probeSshProfileInventory(connection)
            // The inventory probe learns the backend's install id on its own session; carrying it
            // here is what lets two ssh addresses for one machine collapse to one row.
            raw = {
              connection,
              profiles: null,
              error: 'connect-on-demand',
              installId: connectionInstallIds.get(connection.id)?.id
            }
          } else {
            // Same connect-on-demand courtesy for the forced-local path: when
            // the primary route is remote, enumerating "This device" would
            // SPAWN a local backend this user has never asked for — a phantom
            // `default` agent that also forces -device handle disambiguation
            // onto the real one (remote-gateway-only desktops showed their main
            // agent twice, Aug 17 2026). Enumerate the local source only when
            // it is the delegate route (local-primary desktops, unchanged
            // behavior) or a forced-local child is ALREADY pooled (the user
            // opened one).
            if (connection.kind === 'local') {
              const localRoute = resolveRegistryLocalRoute('default', {
                globalRemote: globalRemoteActive(),
                profileRemoteOverride: Boolean(profileHasRemoteOverride(primaryProfileKey()))
              })

              if (shouldDeferLocalEnumeration(localRoute, backendPool.keys(), connection.id)) {
                return { connection, profiles: null, error: 'connect-on-demand' }
              }
            }

            // Claim-guarded (#90812): this ~5s roster poll can race a renderer's
            // own reconnect dial for the same connection; coalescing avoids
            // bootstrapping a second SSH tunnel / remote dashboard.
            const descriptor: any = await withEnumerationDeadline(
              Promise.resolve(
                backendDialClaims.run(backendScopeKey(connection.id, null), () =>
                  ensureRegistryBackend(connection.id, null)
                )
              ),
              rosterSourceEnumerationTimeoutMs(connection)
            )

            const { body, installId } = await fetchRosterSourceData(
              () => getJsonForBackend(descriptor, '/api/profiles', { timeoutMs: 8_000 }),
              () => probeConnectionInstallId(connection.id, descriptor)
            )

            // The install-id probe is TTL-cached, so the 5s roster poll usually
            // pays zero extra requests; on a miss it runs beside /api/profiles.

            const profiles = Array.isArray(body?.profiles)
              ? body.profiles.map(p => String(p?.name || '').trim()).filter(Boolean)
              : []

            const profileMetadata = Array.isArray(body?.profiles)
              ? Object.fromEntries(
                  body.profiles
                    .map(profile => {
                      const name = String(profile?.name || '').trim()

                      if (!name) {
                        return null
                      }

                      const metadata: RosterProfileMetadata = {}

                      if (typeof profile?.display_name === 'string' && profile.display_name.trim()) {
                        metadata.display_name = profile.display_name.trim()
                      }

                      if (typeof profile?.title === 'string' && profile.title.trim()) {
                        metadata.title = profile.title.trim()
                      }

                      if (profile?.ui_meta && typeof profile.ui_meta === 'object') {
                        metadata.ui_meta = profile.ui_meta
                      }

                      if (typeof profile?.has_avatar === 'boolean') {
                        metadata.has_avatar = profile.has_avatar
                      }

                      return [name, metadata] as const
                    })
                    .filter((entry): entry is readonly [string, RosterProfileMetadata] => Boolean(entry))
                )
              : undefined

            // The root HERMES_HOME is an agent too; enumerations that omit it
            // (older backends list only named profiles) still get a default row.
            if (!profiles.includes('default')) {
              profiles.unshift('default')
            }

            raw = {
              connection,
              profiles,
              ...(installId ? { installId } : {}),
              ...(profileMetadata ? { profileMetadata } : {})
            }
          }
        } catch (error: any) {
          raw = { connection, profiles: null, error: String(error?.message || error) }
        }

        if (raw.profiles && raw.profiles.length > 0) {
          sshRosterCache.set(connection.id, raw.profiles)
        }

        const remembered = rememberSshEnumeration(raw, sshRosterCache.get(connection.id), connection.kind)

        return {
          connection,
          ...remembered,
          ...(raw.installId ? { installId: raw.installId } : {}),
          ...(raw.profileMetadata ? { profileMetadata: raw.profileMetadata } : {})
        }
      })
    )
  }

  ipcMain.handle('hermes:agents:roster', async () => {
    const registry = readDesktopConnectionsRegistry()
    const enumerations = await enumerateRegistryAgentSources(registry)

    return {
      agents: buildAgentRoster(enumerations, { primaryConnectionId: registry.primary }),
      // The active gateway owns the renderer's profiles.list — union agents
      // that report THIS connection are the same identities, not extra rows.
      // Expose the primary id so the plugin merger can annotate them in place
      // instead of appending duplicates (remote-only desktops doubled every
      // bot otherwise; see #88344).
      primaryConnectionId: registry.primary,
      sources: enumerations.map(({ connection, error, installId, profiles }) => ({
        connectionId: connection.id,
        label: connection.label,
        kind: connection.kind,
        reachable: profiles !== null,
        ...(installId ? { installId } : {}),
        ...(error ? { error } : {})
      }))
    }
  })

  // Registry-scoped fresh WS URL: the (connectionId, profile) analogue of
  // hermes:gateway:ws-url. Same single-use-ticket discipline for OAuth sources.
  const registryGatewayWsUrlHandler = createRegistryGatewayWsUrlHandler({
    ensureBackend: ensureRegistryBackend,
    mintTicket: mintGatewayWsTicket,
    buildTicketUrl: buildGatewayWsUrlWithTicket,
    rememberHeaders: rememberRemoteWsHeaders
  })

  ipcMain.handle('hermes:gateway:ws-url-for', async (_event, payload) => {
    return gatewayWsUrlIpcResult(() => registryGatewayWsUrlHandler(payload))
  })

  // Transactional update for a Desktop-managed SSH install. Unlike the generic
  // fleet fan-out below, this path owns the remote serve lifecycle: it gates new
  // dials, drains only exact Desktop-owned processes, runs the launcher outside
  // those serves, proves the correlated receipt, and restores every prior scope.
  async function requestManagedSshUpdate(rawId) {
    const connectionId = String(rawId || '').trim()
    const existing = managedConnectionUpdates.get(connectionId)

    if (existing) {
      return existing
    }

    const correlationId = crypto.randomUUID()
    const registry = readDesktopConnectionsRegistry()
    const source = registry.connections.find(connection => connection.id === connectionId)

    if (!source) {
      return refusedManagedSshUpdate(connectionId, correlationId, `No connection with id "${connectionId}".`)
    }

    if (source.kind !== 'ssh') {
      return refusedManagedSshUpdate(
        connectionId,
        correlationId,
        'Only registered Desktop-managed SSH connections can use this update lifecycle.'
      )
    }

    if (!managedConnectionUpdateGate.claim(connectionId, correlationId)) {
      return refusedManagedSshUpdate(connectionId, correlationId, 'A managed update is already in progress.')
    }

    const operation = (async () => {
      try {
        return await updateManagedSshConnection(source, correlationId)
      } catch (error: any) {
        return refusedManagedSshUpdate(connectionId, correlationId, String(error?.message || error))
      } finally {
        managedConnectionUpdateGate.release(connectionId, correlationId)
        managedConnectionUpdates.delete(connectionId)
      }
    })()

    managedConnectionUpdates.set(connectionId, operation)

    return operation
  }

  ipcMain.handle('hermes:connections:update-managed', async (_event, rawId) => requestManagedSshUpdate(rawId))

  // Fan out `hermes update` to every eligible registered connection at once.
  // Cloud entries are excluded (platform-managed); each dispatch reports
  // independently so one dead LAN box can't wedge the batch. Local reuses the
  // app's own update pipeline; Desktop-managed SSH uses the transactional
  // drain/update/restore lifecycle; URL remotes POST their backend updater.
  ipcMain.handle('hermes:connections:update-all', async (_event, payload) => {
    const registry = readDesktopConnectionsRegistry()

    // Optional renderer-side exclusions: the everything-update flow dispatches
    // the ACTIVE backend through its own detailed-progress path and chains the
    // local client apply LAST (it relaunches the app), so it excludes those ids
    // here to avoid double-dispatch. No payload keeps the Settings button's
    // original all-rows behavior byte-identical.
    const excludeIds = new Set<string>(
      Array.isArray((payload as any)?.excludeIds) ? (payload as any).excludeIds.map((id: unknown) => String(id)) : []
    )

    const results = await Promise.all(
      registry.connections
        .filter(connection => !excludeIds.has(connection.id))
        .map(async connection => {
          const base = { connectionId: connection.id, label: connection.label, kind: connection.kind }
          const eligibility = updateEligibility(connection)

          if (!eligibility.eligible) {
            return { ...base, ok: false, skipped: true, reason: eligibility.reason }
          }

          try {
            if (connection.kind === 'local') {
              // The app-managed runtime updates through the same pipeline as the
              // Settings → Updates button (marker + venv gate + relaunch flow).
              const result: any = await applyUpdates({})

              return { ...base, ok: result?.ok !== false, detail: result?.message || 'update started' }
            }

            if (connection.kind === 'ssh') {
              const result = await requestManagedSshUpdate(connection.id)

              return {
                ...base,
                ok: result.ok,
                detail: result.message,
                managed: result,
                ...(result.ok ? {} : { error: result.error || result.outcome })
              }
            }

            // Claim-guarded (#90812): coalesce with a concurrent renderer dial
            // for the same connection instead of bootstrapping a second backend.
            const descriptor: any = await backendDialClaims.run(backendScopeKey(connection.id, null), () =>
              ensureRegistryBackend(connection.id, null)
            )

            const body: any = await postJsonForBackend(descriptor, '/api/hermes/update', {}, { timeoutMs: 15_000 })

            if (body?.ok === false) {
              // The backend refused (docker/nix/externally-managed installs) —
              // surface ITS message, per-row, instead of failing the batch.
              return {
                ...base,
                ok: false,
                skipped: true,
                reason: body?.error || 'backend-refused',
                detail: body?.message
              }
            }

            return { ...base, ok: true, detail: body?.message || 'update started' }
          } catch (error: any) {
            return { ...base, ok: false, error: String(error?.message || error) }
          }
        })
    )

    return { ok: true, results }
  })


  return {
    rememberConnectionInstallId,
    probeSshProfileInventory,
    enumerateRegistryAgentSources
  }
}
