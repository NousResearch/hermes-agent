import { execText } from './backend-claim'
import { teardownSshState } from './connection-apply'
import { connectionScopeKey, resolveRemoteSshDashboardProfile } from './connection-config'
import { backendScopeKey } from './connection-registry'
import { adoptServedDashboardToken } from './dashboard-token'
import { createDesktopConnectionAuthRuntime } from './desktop-connection-auth-runtime'
import { createDesktopConnectionDescriptorRuntime } from './desktop-connection-descriptor-runtime'
import { createDesktopConnectionNotifications } from './desktop-connection-notifications'
import { createDesktopConnectionProbeRuntime } from './desktop-connection-probe-runtime'
import { createDesktopConnectionRestRuntime } from './desktop-connection-rest-runtime'
import { createDesktopConnectionStorageRuntime } from './desktop-connection-storage-runtime'
import { loadOrCreateInstallationId, sshOwnershipId } from './desktop-installation'
import { createManagedPrimaryRoutingGuard } from './desktop-managed-primary-routing-guard'
import { createDesktopOauthSessionRuntime } from './desktop-oauth-session-runtime'
import { createDesktopProfileRoutingRuntime } from './desktop-profile-routing-runtime'
import { resolveDesktopRemoteRoute, v1SshTerminalPoolKey } from './desktop-remote-route'
import { createDesktopSshBootstrapRuntime } from './desktop-ssh-bootstrap-runtime'
import { createDesktopSshSessionRuntime } from './desktop-ssh-session-runtime'
import { createManagedSshRecoveryJournal } from './managed-ssh-recovery-journal'
import { ManagedConnectionUpdateGate } from './managed-ssh-update'
import * as remoteLifecycle from './remote-lifecycle'
import { createBootstrapCoordinator } from './ssh-bootstrap-coordinator'
import { pickLocalPort, redactSecrets, SshConnection } from './ssh-connection'
import { createSshIsolatedKeepaliveRegistry } from './ssh-isolated-keepalive'
import { createSshTeardownTracker } from './ssh-teardown'
import { registrySshPoolScopeByConnectionId, registrySshScopeForWindowRoute } from './window-connection-route'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { connectWindowsRemote, detectRemotePlatform, terminateOwnedWindowsDashboardForUpdate } from './windows-remote-lifecycle'

// This composition owns the connection authority chain in the same order as
// the former main-process declarations. Early consumers receive lazy callbacks;
// no auth, storage, or SSH factory may invoke them before assembly is complete.
export function createDesktopConnectionAssembly(deps: any) {
  const {
    app, BrowserWindow, electronNet, session, safeStorage,
    DESKTOP_CONNECTION_CONFIG_PATH, DESKTOP_CONNECTIONS_REGISTRY_PATH,
    DESKTOP_INSTALLATION_PATH, DESKTOP_MANAGED_SSH_RECOVERY_PATH,
    DESKTOP_PROFILE_CONFIG_PATH, HERMES_HOME, PROFILE_NAME_RE, GUEST_ONBOARDING,
    fetchJson, fetchPublicJson, rememberLog, writeFileAtomic,
    ensureBackend, ensureRegistryBackend, stopRegistryConnectionBackends,
    primaryProfileKey, getIsolatedBackend, backendDialClaims, waitForHermes,
    getWindowConnectionRoute, disposeTerminalSessionsForSshScope,
    managedSshConfig, startHermes, getMainWindow
  } = deps

  // ---------------------------------------------------------------------------
  // OAuth remote-gateway auth.
  //
  // Hosted Hermes gateways gate the dashboard behind an OAuth provider (e.g.
  // Nous Research) instead of a static session token. The auth model is
  // fundamentally different from the token path:
  //
  //   * REST is authed by HttpOnly session cookies (``hermes_session_at``),
  //     established by a browser redirect round-trip (/login → IDP →
  //     /auth/callback sets cookies). We cannot read the HttpOnly cookie value
  //     in JS — instead we let an Electron BrowserWindow complete the round
  //     trip into a PERSISTENT session partition, and thereafter route our REST
  //     through Electron's ``net`` bound to that same partition so the cookie
  //     jar attaches the cookie automatically.
  //   * WebSocket upgrades require a single-use ``?ticket=`` minted at
  //     ``POST /api/auth/ws-ticket`` (cookie-authed). The legacy ``?token=``
  //     path is unconditionally rejected by gated gateways.
  //   * Nous Portal now issues a 24h ROTATING, reuse-detected refresh token
  //     alongside the ~15-min access token (Portal NAS #293 / hermes #37247).
  //     Both are set as HttpOnly cookies (``hermes_session_at`` ~15 min,
  //     ``hermes_session_rt`` 24h). When the AT cookie lapses but the RT cookie
  //     is still alive, the gateway middleware transparently rotates a fresh AT
  //     on the next authenticated request — so connectivity must NOT be gated on
  //     the AT cookie alone. We probe liveness by actually minting a ws-ticket
  //     (which triggers that server-side refresh) and treat a real 401 as
  //     "needs re-login"; the AT-or-RT cookie presence check is only a cheap
  //     "is the user signed in at all?" gate / display signal.
  // ---------------------------------------------------------------------------

  const {
    getOauthSession,
    getOauthSessionForUrl,
    warmOauthCookieStore,
    hasOauthSessionCookie,
    hasLiveOauthSession,
    clearOauthSession,
    openOauthLoginWindow,
    fetchJsonViaOauthSession
  } = createDesktopOauthSessionRuntime({
    app,
    BrowserWindow,
    electronNet,
    session,
    readDesktopConnectionsRegistry,
    readDesktopConnectionConfig,
    installRemoteHeaderRulesOnSession,
    headersForRemoteRequest,
    rememberLog,
    installWindowRendererLifecycle
  })

  const {
    _nativeTokenStoreIo,
    nativeAccessTokenCoordinator,
    ensureNativeAccessToken,
    hasNativeSession,
    postJsonNoAuth,
    mintGatewayWsTicket,
    freshGatewayWsUrl,
    resolvePortalBaseUrl,
    hasLivePortalSession,
    hasPortalAccessToken,
    renewPortalAccessSilently,
    openPortalLoginWindow,
    discoverCloudAgents,
    cloudAgentSilentSignIn
  } = createDesktopConnectionAuthRuntime({
    app,
    BrowserWindow,
    fetchJson,
    fetchJsonViaOauthSession,
    encryptDesktopSecret: (value, options) => encryptDesktopSecret(value, options),
    decryptDesktopSecret: secret => decryptDesktopSecret(secret),
    ensureBackend: profile => ensureBackend(profile),
    getOauthSession,
    warmOauthCookieStore,
    hasOauthSessionCookie,
    openOauthLoginWindow,
    rememberRemoteWsHeaders: (url, headers) => rememberRemoteWsHeaders(url, headers),
    rememberLog
  })

  const { postJsonForBackend, getJsonForBackend, fetchJsonForBackend } = createDesktopConnectionRestRuntime({
    ensureNativeAccessToken,
    fetchJson,
    fetchJsonViaOauthSession
  })

  const { sendConnectionApplied, broadcastConnectionsChanged } = createDesktopConnectionNotifications({
    BrowserWindow,
    getMainWindow
  })

  // ---------------------------------------------------------------------------
  // Opt-in keychain encryption (secret-storage-policy.ts owns the decision).
  // Default OFF: no safeStorage call is ever made, so a broken/locked macOS
  // login keychain can never throw its password dialog on launch. Settings →
  // Gateway exposes the toggle; flipping it re-encrypts (or decrypts) the
  // stored secrets in place.
  // ---------------------------------------------------------------------------
  const {
    secretStoragePolicy,
    applySecretStorageEncryption,
    probeSecureTokenStorage,
    encryptDesktopSecret,
    decryptDesktopSecret,
    decryptRemoteHeaders,
    encryptIncomingRemoteHeaders,
    rememberRemoteWsHeaders,
    headersForRemoteRequest: headersForRemoteRequestImpl,
    installRemoteHeaderRulesOnSession: installRemoteHeaderRulesOnSessionImpl,
    installRemoteHeaderRules,
    readDesktopConnectionConfig: readDesktopConnectionConfigImpl,
    writeDesktopConnectionConfig,
    readDesktopConnectionsRegistry: readDesktopConnectionsRegistryImpl,
    writeDesktopConnectionsRegistry,
    sanitizeConnectionsRegistry,
    sanitizeRegistryConnection,
    saveRegistryConnection,
    migrateLegacyEncryptedSecretsOnce
  } = createDesktopConnectionStorageRuntime({
    app,
    safeStorage,
    session,
    connectionConfigPath: DESKTOP_CONNECTION_CONFIG_PATH,
    connectionsRegistryPath: DESKTOP_CONNECTIONS_REGISTRY_PATH,
    profileNameRe: PROFILE_NAME_RE,
    nativeTokenStoreIo: _nativeTokenStoreIo,
    rememberLog,
    assertCanMutateRegistryConnection: id => managedConnectionUpdateGate.assertCanMutate(id),
    stopRegistryConnectionBackends,
    broadcastConnectionsChanged
  })

  // These callbacks are passed into earlier runtimes during module evaluation.
  // Keep their declarations hoisted; the storage runtime is ready before any
  // callback is invoked by the app lifecycle.
  function headersForRemoteRequest(requestUrl: string) {
    return headersForRemoteRequestImpl(requestUrl)
  }

  function installRemoteHeaderRulesOnSession(sess: Electron.Session) {
    return installRemoteHeaderRulesOnSessionImpl(sess)
  }

  function readDesktopConnectionConfig() {
    return readDesktopConnectionConfigImpl()
  }

  function readDesktopConnectionsRegistry() {
    return readDesktopConnectionsRegistryImpl()
  }

  const {
    desktopProfilePreferences,
    validateDesktopProfileRoute,
    readActiveDesktopProfile,
    writeActiveDesktopProfile,
    migrateActiveProfileIfMissing,
    profileRouteOptions
  } = createDesktopProfileRoutingRuntime({
    configPath: DESKTOP_PROFILE_CONFIG_PATH,
    hermesHome: HERMES_HOME,
    profileNameRe: PROFILE_NAME_RE,
    BrowserWindow,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    primaryProfileKey,
    globalRemoteActive,
    primaryBackendIsRemote: () => primaryBackendIsRemote(),
    getIsolatedBackend,
    writeFileAtomic
  })

  const { sanitizeDesktopConnectionConfig, coerceDesktopConnectionConfig, buildRemoteConnection } =
    createDesktopConnectionDescriptorRuntime({
      readDesktopConnectionConfig,
      decryptDesktopSecret,
      decryptRemoteHeaders,
      encryptDesktopSecret,
      probeSecureTokenStorage,
      hasNativeSession,
      hasLiveOauthSession,
      mintGatewayWsTicket,
      rememberRemoteWsHeaders
    })

  const sshConnections = new Map<string, any>()

  const sshIsolatedKeepalives = createSshIsolatedKeepaliveRegistry({
    log: chunk => sshRememberLog(chunk)
  })

  const desktopInstallationId = loadOrCreateInstallationId(DESKTOP_INSTALLATION_PATH)

  // Managed SSH update lifecycle (#93042): while an update owns a registered
  // SSH connection, the gate pauses new dials and dial-material mutations for
  // that connection id; the durable recovery journal below survives a crash
  // mid-transaction so the next launch can restore every drained scope.
  const managedConnectionUpdateGate = new ManagedConnectionUpdateGate(
    connectionId =>
      readManagedSshRecoveryRecords().find(record => record.connectionId === connectionId)?.correlationId || null
  )

  const managedConnectionUpdates = new Map<string, Promise<any>>()
  const managedConnectionRecoveries = new Map<string, Promise<void>>()
  const managedPrimaryRestoreOwners = new Map<string, { correlationId: string; profile: string; source: any }>()
  let managedUpdateQuitWait: Promise<void> | null = null
  let managedUpdateQuitWaitDone = false

  const assertCanMutateManagedPrimaryRouting = createManagedPrimaryRoutingGuard({
    managedConnectionUpdates,
    managedConnectionRecoveries,
    managedPrimaryRestoreOwners,
    readManagedSshRecoveryRecords: () => readManagedSshRecoveryRecords()
  })

  const {
    readManagedSshRecoveryRecords,
    persistManagedSshRecovery,
    markManagedSshRecoveryLaunching,
    clearManagedSshRecovery
  } = createManagedSshRecoveryJournal(DESKTOP_MANAGED_SSH_RECOVERY_PATH)


  const sshBootstrapCoordinator = createBootstrapCoordinator()
  const sshTeardowns = createSshTeardownTracker()

  const {
    sshScopeKey,
    sshRememberLog,
    teardownSshConnection,
    activeSshTerminalTarget,
    ensureTerminalBackend,
    resetPreviewReach,
    reachablePreviewUrl,
    effectiveSshConfigFingerprint,
    bootstrapSshConnection
  } = createDesktopSshSessionRuntime({
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
    ensureBackend: profile => ensureBackend(profile),
    ensureRegistryBackend: (connectionId, profile) => ensureRegistryBackend(connectionId, profile),
    execText,
    fetchJson,
    managedConnectionUpdateGate,
    persistSshConnectionToken: (profile, source, token, id) => persistSshConnectionToken(profile, source, token, id),
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
    terminalIpc: { disposeTerminalSessionsForSshScope },
    terminateOwnedWindowsDashboardForUpdate,
    v1SshTerminalPoolKey,
    waitForHermes,
    windowConnectionRoutes: { get: getWindowConnectionRoute }
  })

  const {
    persistSshConnectionToken: persistSshConnectionTokenImpl,
    resolveRemoteBackend,
    profileHasRemoteOverride,
    configuredRemoteProfileNames,
    globalRemoteActive: globalRemoteActiveImpl,
    registryPrimaryIsRemote,
    primaryBackendIsRemote,
    fetchJsonForProfile,
    requestJsonForProfile,
    probeRemoteAuthMode,
    testDesktopConnectionConfig,
    fetchConnectionStatus
  } = createDesktopConnectionProbeRuntime({
    bootstrapSshConnection,
    buildRemoteConnection,
    coerceDesktopConnectionConfig,
    decryptDesktopSecret,
    decryptRemoteHeaders,
    encryptDesktopSecret,
    ensureBackend: profile => ensureBackend(profile),
    fetchJsonForBackend,
    fetchPublicJson,
    managedConnectionUpdateGate,
    managedPrimaryRestoreOwners,
    managedSshConfig: (source, profile) => managedSshConfig(source, profile),
    mintGatewayWsTicket,
    primaryProfileKey,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    sshRememberLog,
    startHermes,
    writeDesktopConnectionConfig,
    writeDesktopConnectionsRegistry
  })

  // Earlier runtimes receive these callbacks before the probe runtime is built.
  function persistSshConnectionToken(profile, source, token, registryConnectionId = '') {
    return persistSshConnectionTokenImpl(profile, source, token, registryConnectionId)
  }

  function globalRemoteActive() {
    return globalRemoteActiveImpl()
  }

  const managedUpdateQuitState = {
    get wait() { return managedUpdateQuitWait },
    set wait(value: Promise<void> | null) { managedUpdateQuitWait = value },
    get done() { return managedUpdateQuitWaitDone },
    set done(value: boolean) { managedUpdateQuitWaitDone = value }
  }

  return {
    getOauthSessionForUrl, hasOauthSessionCookie, hasLiveOauthSession,
    clearOauthSession, openOauthLoginWindow, fetchJsonViaOauthSession,
    nativeAccessTokenCoordinator, ensureNativeAccessToken, hasNativeSession,
    postJsonNoAuth, mintGatewayWsTicket, freshGatewayWsUrl,
    resolvePortalBaseUrl, hasLivePortalSession, openPortalLoginWindow,
    discoverCloudAgents, cloudAgentSilentSignIn,
    postJsonForBackend, getJsonForBackend, fetchJsonForBackend,
    sendConnectionApplied, broadcastConnectionsChanged,
    secretStoragePolicy, applySecretStorageEncryption,
    encryptDesktopSecret, decryptDesktopSecret, decryptRemoteHeaders,
    rememberRemoteWsHeaders, headersForRemoteRequest, installRemoteHeaderRules,
    readDesktopConnectionConfig, writeDesktopConnectionConfig,
    readDesktopConnectionsRegistry, writeDesktopConnectionsRegistry,
    sanitizeConnectionsRegistry, saveRegistryConnection,
    migrateLegacyEncryptedSecretsOnce,
    desktopProfilePreferences, validateDesktopProfileRoute,
    readActiveDesktopProfile, writeActiveDesktopProfile,
    migrateActiveProfileIfMissing, profileRouteOptions,
    sanitizeDesktopConnectionConfig, coerceDesktopConnectionConfig,
    buildRemoteConnection, sshConnections, sshIsolatedKeepalives,
    managedConnectionUpdateGate, managedConnectionUpdates,
    managedConnectionRecoveries, managedPrimaryRestoreOwners,
    managedUpdateQuitState, assertCanMutateManagedPrimaryRouting,
    readManagedSshRecoveryRecords, persistManagedSshRecovery,
    markManagedSshRecoveryLaunching, clearManagedSshRecovery,
    sshBootstrapCoordinator, sshTeardowns, sshScopeKey, sshRememberLog,
    teardownSshConnection, activeSshTerminalTarget, ensureTerminalBackend,
    resetPreviewReach, reachablePreviewUrl, effectiveSshConfigFingerprint,
    bootstrapSshConnection, resolveRemoteBackend, profileHasRemoteOverride,
    configuredRemoteProfileNames, globalRemoteActive, primaryBackendIsRemote,
    fetchJsonForProfile, requestJsonForProfile, probeRemoteAuthMode,
    testDesktopConnectionConfig, fetchConnectionStatus
  }
}
