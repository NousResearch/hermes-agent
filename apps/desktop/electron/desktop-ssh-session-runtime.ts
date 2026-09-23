import { PreviewReachRegistry } from './preview-reach'

// A renderer borrows only the SSH tunnel selected by its current route. The
// shell owns the live maps and coordinator; this runtime holds their helpers.
export function createDesktopSshSessionRuntime(deps: any) {
  const {
    GUEST_ONBOARDING,
    SshConnection,
    adoptServedDashboardToken,
    backendDialClaims,
    backendScopeKey,
    buildRemoteConnection,
    connectWindowsRemote,
    connectionScopeKey,
    createDesktopSshBootstrapRuntime,
    desktopInstallationId,
    detectRemotePlatform,
    ensureBackend,
    ensureRegistryBackend,
    execText,
    fetchJson,
    managedConnectionUpdateGate,
    persistSshConnectionToken,
    pickLocalPort,
    primaryProfileKey,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    redactSecrets,
    registrySshPoolScopeByConnectionId,
    registrySshScopeForWindowRoute,
    rememberLog,
    remoteLifecycle,
    resolveDesktopRemoteRoute,
    resolveRemoteSshDashboardProfile,
    sshBootstrapCoordinator,
    sshConnections,
    sshIsolatedKeepalives,
    sshOwnershipId,
    sshTeardowns,
    teardownSshState,
    terminalIpc,
    terminateOwnedWindowsDashboardForUpdate,
    v1SshTerminalPoolKey,
    waitForHermes,
    windowConnectionRoutes
  } = deps

  function sshScopeKey(profile) {
    return connectionScopeKey(profile) || ''
  }

  function sshOwnershipKey(profile) {
    return sshOwnershipId(desktopInstallationId, sshScopeKey(profile))
  }

  function sshRememberLog(chunk) {
    rememberLog(redactSecrets(String(chunk == null ? '' : chunk)))
  }

  async function sshProbeReuseProof(baseUrl, token, spawnNonce) {
    try {
      const proof: any = await fetchJson(`${baseUrl}/api/ssh/ownership`, token)

      return remoteLifecycle.classifySshReuseProof(proof, spawnNonce)
    } catch (error: any) {
      if (/^(401|403|404):/.test(String(error?.message || ''))) {
        return 'authenticated-stale'
      }

      throw error
    }
  }

  async function teardownSshConnection(profile) {
    const scope = sshScopeKey(profile)
    sshIsolatedKeepalives.stop(scope)
    const state = sshConnections.get(scope)

    if (!state) {
      return
    }

    sshConnections.delete(scope)

    terminalIpc.disposeTerminalSessionsForSshScope(scope)

    // Kill the owned remote serve --isolated *before* closing the SSH
    // transport. Spawn detaches with setsid/nohup, so closing the tunnel
    // alone leaves the backend at pid 1 holding state.db (#91668).
    // Windows remotes use a different lifecycle (connectWindowsRemote) and
    // are left to a follow-up; POSIX is the leak that OOM'd gateways.
    await sshTeardowns.track(state.ssh, () =>
      teardownSshState(
        {
          ...state,
          ownershipId: state.ownershipId || sshOwnershipKey(profile)
        },
        {
          cleanupRemote:
            state.remotePlatform === 'Windows'
              ? async () => {
                  // connectWindowsRemote does not share POSIX lock/kill. Stay
                  // silent on the kill path, but leave a log so quit is not a
                  // mysterious no-op on Windows remotes.
                  sshRememberLog('[ssh] skip remote serve teardown on Windows remotes; POSIX disconnect does not apply')
                }
              : remoteLifecycle.disconnect
        }
      )
    )
  }

  // CRITICAL: this must mirror resolveRemoteBackend's precedence, not just return
  // any cached SSH state. A per-profile token/OAuth override wins over a global
  // SSH connection — so if the active profile resolves to a NON-SSH backend, the
  // terminal must NOT fall through to a global SSH host.
  function activeSshTerminalTarget(webContentsId?: number) {
    const windowRoute = typeof webContentsId === 'number' ? windowConnectionRoutes.get(webContentsId) : null

    if (windowRoute?.registryScoped && windowRoute.connectionId) {
      const scope = registrySshScopeForWindowRoute(windowRoute, readDesktopConnectionsRegistry())

      if (!scope) {
        return null
      }

      const state = sshConnections.get(scope)

      if (state && state.ssh) {
        return { ssh: state.ssh, scope }
      }

      // The pool's single writer publishes under the per-profile bootstrap key
      // while stamping the entry with its registry connection id (#97345), so a
      // composite-key miss must still resolve the live tunnel by that identity
      // instead of reporting 'pending' forever.
      const pooledScope = registrySshPoolScopeByConnectionId(sshConnections, windowRoute.connectionId)
      const pooledState = pooledScope === null ? null : sshConnections.get(pooledScope)

      return pooledState && pooledState.ssh ? { ssh: pooledState.ssh, scope: pooledScope } : 'pending'
    }

    const profile = windowRoute?.profile ?? primaryProfileKey()
    const config = readDesktopConnectionConfig()

    const route = resolveDesktopRemoteRoute({
      config,
      env: {
        token: process.env.HERMES_DESKTOP_REMOTE_TOKEN,
        url: process.env.HERMES_DESKTOP_REMOTE_URL
      },
      profile,
      registry: readDesktopConnectionsRegistry()
    })

    if (!route || route.kind !== 'ssh') {
      return null
    }

    const scope = v1SshTerminalPoolKey(route, profile)

    const state = sshConnections.get(scope)

    return state && state.ssh ? { ssh: state.ssh, scope } : 'pending'
  }

  async function ensureTerminalBackend(webContentsId: number) {
    const windowRoute = windowConnectionRoutes.get(webContentsId)

    // Claim-guarded (#90812): opening a terminal pane can race a renderer's own
    // reconnect dial for the same (connectionId, profile) scope; coalescing
    // here avoids bootstrapping a second SSH tunnel / remote dashboard.
    if (windowRoute?.registryScoped && windowRoute.connectionId) {
      return backendDialClaims.run(backendScopeKey(windowRoute.connectionId, windowRoute.profile), () =>
        ensureRegistryBackend(windowRoute.connectionId, windowRoute.profile)
      )
    }

    const profile = windowRoute?.profile ?? primaryProfileKey()

    return backendDialClaims.run(backendScopeKey(null, profile), () => ensureBackend(profile))
  }

  // Loopback reach for the browser pane. Scoped to the SSH connection that
  // authorized it: a different host (or none) must never inherit live forwards
  // into somebody else's machine.
  const previewReachByWebContents = new Map<number, { registry: PreviewReachRegistry; scope: string }>()

  async function resetPreviewReach(webContentsId?: number) {
    if (typeof webContentsId === 'number') {
      const current = previewReachByWebContents.get(webContentsId)

      previewReachByWebContents.delete(webContentsId)

      if (current) {
        await current.registry.closeAll()
      }

      return
    }

    const open = [...previewReachByWebContents.values()]

    previewReachByWebContents.clear()
    await Promise.allSettled(open.map(entry => entry.registry.closeAll()))
  }

  /**
   * Rewrite a gateway-loopback URL into one this machine can actually load.
   *
   * Returns the URL unchanged when no rewrite is needed or possible — a local
   * backend (the address is already true), a non-loopback host, or a url/cloud
   * remote with no tunnel to borrow. Callers must not treat an unchanged URL as
   * failure; the pane explains an unreachable one on its own.
   */
  async function reachablePreviewUrl(webContentsId: number, rawUrl: string): Promise<string> {
    let target = activeSshTerminalTarget(webContentsId)

    if (target === 'pending') {
      await ensureTerminalBackend(webContentsId).catch(() => undefined)
      target = activeSshTerminalTarget(webContentsId)
    }

    if (!target || target === 'pending') {
      // No SSH transport behind this renderer's gateway. Another window's
      // forward must never be reused for this preview.
      await resetPreviewReach(webContentsId)

      return rawUrl
    }

    const { scope, ssh } = target as { scope: string; ssh: any }
    let reach = previewReachByWebContents.get(webContentsId)

    if (!reach || reach.scope !== scope) {
      await resetPreviewReach(webContentsId)
      reach = { registry: new PreviewReachRegistry(), scope }
      previewReachByWebContents.set(webContentsId, reach)
    }

    try {
      const rewritten = await reach.registry.resolve(rawUrl, {
        cancel: (localPort, remotePort) => ssh.cancelForward(localPort, remotePort),
        forward: (localPort, remotePort, remoteHost) => ssh.forward(localPort, remotePort, remoteHost),
        isCurrent: () => sshConnections.get(scope)?.ssh === ssh,
        // pickLocalPort predates the typed surface here and infers `unknown`.
        pickLocalPort: () => pickLocalPort() as Promise<number>
      })

      return rewritten || rawUrl
    } catch (error: any) {
      sshRememberLog(`preview reach failed for ${rawUrl}: ${error?.message || error}`)

      return rawUrl
    }
  }

  const { effectiveSshConfigFingerprint, bootstrapSshConnection } = createDesktopSshBootstrapRuntime({
    GUEST_ONBOARDING,
    SshConnection,
    adoptServedDashboardToken,
    buildRemoteConnection,
    connectWindowsRemote,
    detectRemotePlatform,
    execText,
    managedConnectionUpdateGate,
    persistSshConnectionToken,
    pickLocalPort,
    remoteLifecycle,
    resolveRemoteSshDashboardProfile,
    sshBootstrapCoordinator,
    sshConnections,
    sshIsolatedKeepalives,
    sshOwnershipKey,
    sshProbeReuseProof,
    sshRememberLog,
    sshScopeKey,
    teardownSshConnection,
    terminateOwnedWindowsDashboardForUpdate,
    waitForHermes
  })


  return {
    sshScopeKey,
    sshRememberLog,
    teardownSshConnection,
    activeSshTerminalTarget,
    ensureTerminalBackend,
    resetPreviewReach,
    reachablePreviewUrl,
    effectiveSshConfigFingerprint,
    bootstrapSshConnection
  }
}
