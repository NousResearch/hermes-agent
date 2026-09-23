import { applyConnectionChange } from './connection-apply'
import { authModeFromStatus, connectionScopeKey, modeIsRemoteLike, normalizeRemoteBaseUrl } from './connection-config'
import { applyConnectionConfigAtomically } from './connection-config-apply'
import { reconcileAppliedGlobalConnection } from './connection-registry'
import { NativeAuthChangedError } from './native-access-token'
import { resolveLoginStrategy } from './native-oauth'
import { runNativeLogin } from './native-oauth-login'
import { rehomePrimaryConnection } from './primary-connection-rehome'

interface DesktopConnectionAuthIpcDeps {
  ipcMain: any
  probeRemoteAuthMode: any
  nativeAccessTokenCoordinator: any
  fetchPublicJson: any
  gatewayAuthProviders: any
  postJsonNoAuth: any
  rememberLog: any
  clearOauthSession: any
  hasLiveOauthSession: any
  hasNativeSession: any
  hasOauthSessionCookie: any
  openOauthLoginWindow: any
  resolvePortalBaseUrl: any
  hasLivePortalSession: any
  openPortalLoginWindow: any
  discoverCloudAgents: any
  cloudAgentSilentSignIn: any
  assertCanMutateManagedPrimaryRouting: any
  coerceDesktopConnectionConfig: any
  writeDesktopConnectionConfig: any
  sanitizeDesktopConnectionConfig: any
  readDesktopConnectionConfig: any
  readDesktopConnectionsRegistry: any
  testDesktopConnectionConfig: any
  writeDesktopConnectionsRegistry: any
  sshBootstrapCoordinator: any
  primaryProfileKey: any
  sendConnectionApplied: any
  firstRunBoot: any
  teardownPrimaryBackendAndWait: any
  stopPoolBackend: any
  teardownSshConnection: any
  clearRemoteReauthFailure: any
  clearLocalBootstrapFailure: any
  shell: any
}

export function registerDesktopConnectionAuthIpc(deps: DesktopConnectionAuthIpcDeps) {
  const {
    ipcMain,
    probeRemoteAuthMode,
    nativeAccessTokenCoordinator,
    fetchPublicJson,
    gatewayAuthProviders,
    postJsonNoAuth,
    rememberLog,
    clearOauthSession,
    hasLiveOauthSession,
    hasNativeSession,
    hasOauthSessionCookie,
    openOauthLoginWindow,
    resolvePortalBaseUrl,
    hasLivePortalSession,
    openPortalLoginWindow,
    discoverCloudAgents,
    cloudAgentSilentSignIn,
    assertCanMutateManagedPrimaryRouting,
    coerceDesktopConnectionConfig,
    writeDesktopConnectionConfig,
    sanitizeDesktopConnectionConfig,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    testDesktopConnectionConfig,
    writeDesktopConnectionsRegistry,
    sshBootstrapCoordinator,
    primaryProfileKey,
    sendConnectionApplied,
    firstRunBoot,
    teardownPrimaryBackendAndWait,
    stopPoolBackend,
    teardownSshConnection,
    clearRemoteReauthFailure,
    clearLocalBootstrapFailure,
    shell
  } = deps

ipcMain.handle('hermes:connection-config:probe', async (_event, rawUrl) => probeRemoteAuthMode(rawUrl))
ipcMain.handle('hermes:connection-config:oauth-login', async (_event, rawUrl) => {
  // Capability-gated login (RFC 8252). Probe the gateway's public /api/status
  // for supported auth_flows and /api/auth/providers for provider capabilities:
  //   - all providers support password → always use the embedded login window
  //     (password providers require the dashboard login form; native PKCE
  //     can never complete for that provider shape)
  //   - advertises "native_pkce" AND at least one non-password provider →
  //     run the system-browser + loopback + PKCE flow
  //   - older gateway with no provider metadata → fall back to the auth_flows
  //     check (existing compatibility)
  //   - a failed native login reports the error rather than auto-falling back
  //     to the embedded flow — one sign-in action opens at most one window.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)
  // Order login attempts without interrupting rotation of the existing session.
  const authIsCurrent = nativeAccessTokenCoordinator.beginLogin(baseUrl)

  let statusBody: any = null

  try {
    statusBody = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8_000 })
  } catch {
    // Can't read status — fall through to the embedded flow, which has its
    // own error handling and works against any gated gateway.
  }

  const authRequired = statusBody && authModeFromStatus(statusBody) === 'oauth'
  const providers = authRequired ? await gatewayAuthProviders(baseUrl) : []

  const strategy = resolveLoginStrategy(statusBody, { providers })

  if (!authIsCurrent()) {
    throw new NativeAuthChangedError()
  }

  if (strategy === 'native') {
    try {
      const tokens = await runNativeLogin(baseUrl, {
        openExternal: url => shell.openExternal(url),
        postJson: (url, body, opts) => postJsonNoAuth(url, body, opts),
        rememberLog
      })

      if (!authIsCurrent()) {
        throw new NativeAuthChangedError()
      }

      nativeAccessTokenCoordinator.storeTokens(baseUrl, tokens)
      // Confirmed sign-in — release the reauth latch so the next
      // startHermes() re-dials instead of replaying the stale rejection.
      clearRemoteReauthFailure()

      return { ok: true, baseUrl, connected: true }
    } catch (error) {
      rememberLog(`[native-oauth] native login failed (${error instanceof Error ? error.message : String(error)})`)

      return { ok: false, error: error instanceof Error ? error.message : String(error), connected: false }
    }
  }

  // Legacy embedded-webview cookie flow.
  await openOauthLoginWindow(baseUrl)

  const connected = await hasOauthSessionCookie(baseUrl)

  // Only a CONFIRMED sign-in releases the latch. A cancelled/closed login
  // window must leave it set, or the overlay's "Sign in" button starts
  // flickering again on the next retry.
  if (!authIsCurrent()) {
    throw new NativeAuthChangedError()
  }

  if (connected) {
    // A confirmed cookie login supersedes any older native identity.
    nativeAccessTokenCoordinator.clearTokens(baseUrl)
    clearRemoteReauthFailure()
  }

  return { ok: true, baseUrl, connected }
})
ipcMain.handle('hermes:connection-config:oauth-logout', async (_event, rawUrl) => {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  // Also drop any native (RFC 8252) bearer tokens for this gateway so a
  // logout clears BOTH auth shapes.
  // Clear before awaiting cookie I/O: a pending login/refresh cannot restore
  // logout, and a later login must not be erased when cookie clearing settles.
  nativeAccessTokenCoordinator.clearTokens(baseUrl)
  await clearOauthSession(baseUrl)

  // Report against the SAME liveness notion the Settings indicator uses
  // (AT-or-RT cookie, or a native token) so a logout that left any session
  // behind is reflected as still-connected rather than silently signed-out.
  const connected = (await hasLiveOauthSession(baseUrl)) || hasNativeSession(baseUrl)

  return { ok: true, connected }
})

// --- Hermes Cloud (cloud-auto-discovery Phase 3) ---
// One portal login in the OAuth partition powers both discovery and the silent
// per-agent cascade. See the discovery/cascade helpers above.
ipcMain.handle('hermes:cloud:status', async () => ({
  portalBaseUrl: resolvePortalBaseUrl(),
  signedIn: await hasLivePortalSession()
}))
ipcMain.handle('hermes:cloud:login', async () => {
  await openPortalLoginWindow()

  return { ok: true, signedIn: await hasLivePortalSession() }
})
ipcMain.handle('hermes:cloud:logout', async () => {
  await clearOauthSession(resolvePortalBaseUrl())

  return { ok: true, signedIn: await hasLivePortalSession() }
})
ipcMain.handle('hermes:cloud:discover', async (_event, org) => {
  // Returns { agents } or { needsOrgSelection: true, orgs }. `org` (optional)
  // scopes discovery to a chosen org for multi-org users.
  return discoverCloudAgents(typeof org === 'string' && org ? org : undefined)
})
ipcMain.handle('hermes:cloud:agent-sign-in', async (_event, dashboardUrl) => {
  // Silent per-agent sign-in via the shared portal session. Returns the agent's
  // gateway baseUrl + whether its session cookie landed; the renderer then
  // saves a cloud-mode connection pointed at this dashboardUrl.
  return cloudAgentSilentSignIn(dashboardUrl)
})
ipcMain.handle('hermes:connection-config:save', async (_event, payload) => {
  assertCanMutateManagedPrimaryRouting()
  const config = coerceDesktopConnectionConfig(payload)
  writeDesktopConnectionConfig(config)

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
})
ipcMain.handle('hermes:connection-config:apply', async (_event, payload) => {
  assertCanMutateManagedPrimaryRouting()
  const previousConfig = readDesktopConnectionConfig()
  const previousRegistry = readDesktopConnectionsRegistry()
  const config = coerceDesktopConnectionConfig(payload, previousConfig)

  const key = connectionScopeKey(payload?.profile)
  const scope = key || ''
  const nextRegistry = key ? previousRegistry : reconcileAppliedGlobalConnection(previousRegistry, config)

  await applyConnectionConfigAtomically({
    previousConfig,
    previousRegistry,
    nextConfig: config,
    nextRegistry,
    // Exercise the same authenticated REST + real WebSocket legs before either
    // config file changes. A rejected OAuth session or blocked /api/ws leaves
    // the previous primary/current connection intact.
    preflight: !key && modeIsRemoteLike(config.mode) ? () => testDesktopConnectionConfig(payload) : undefined,
    writeConfig: writeDesktopConnectionConfig,
    writeRegistry: writeDesktopConnectionsRegistry,
    apply: () =>
      applyConnectionChange({
        cancelAndWait: value => sshBootstrapCoordinator.cancelAndWait(value),
        isPrimary: !key || key === primaryProfileKey(),
        rehomePrimary: () =>
          rehomePrimaryConnection({
            clearLocalBootstrapFailure: () => {
              // A remote connection bypasses local runtime/bootstrap failures. Clear
              // the local-install latch so unsupported/failure escape paths can re-home.
              clearLocalBootstrapFailure()
            },
            mode: config.mode,
            notifyConnectionApplied: sendConnectionApplied,
            resumeFirstRunRemote: firstRunBoot.abandonFirstRunSetupChoiceForRemoteApply,
            teardownPrimaryBackend: teardownPrimaryBackendAndWait
          }),
        scope,
        sendApplied: sendConnectionApplied,
        stopPool: stopPoolBackend,
        teardownPrimary: () => teardownPrimaryBackendAndWait({ soft: true }),
        teardownSsh: value => teardownSshConnection(value || null)
      })
  })

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
})


}
