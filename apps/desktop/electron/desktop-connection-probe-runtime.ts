import { authModeFromStatus, connectionScopeKey, modeIsRemoteLike, normalizeRemoteBaseUrl, normalizeSshConfig, normAuthMode, profileHasRemoteConnection, resolveTestWsUrl } from './connection-config'
import { upsertConnection } from './connection-registry'
import { resolveDesktopRemoteRoute } from './desktop-remote-route'
import { probeGatewayWebSocket } from './gateway-ws-probe'
import { DEFAULT_FETCH_TIMEOUT_MS } from './hardening'
import { managedSshTokenPersistencePlan } from './managed-ssh-update'
import * as remoteLifecycle from './remote-lifecycle'
import { createSshProbeConnection } from './ssh-connection'
import { detectRemotePlatform, helper } from './windows-remote-lifecycle'

export function createDesktopConnectionProbeRuntime(deps: {
  bootstrapSshConnection: any
  buildRemoteConnection: any
  coerceDesktopConnectionConfig: any
  decryptDesktopSecret: any
  decryptRemoteHeaders: any
  encryptDesktopSecret: any
  ensureBackend: any
  fetchJsonForBackend: any
  fetchPublicJson: any
  managedConnectionUpdateGate: any
  managedPrimaryRestoreOwners: Map<string, any>
  managedSshConfig: any
  mintGatewayWsTicket: any
  primaryProfileKey: any
  readDesktopConnectionConfig: any
  readDesktopConnectionsRegistry: any
  sshRememberLog: any
  startHermes: any
  writeDesktopConnectionConfig: any
  writeDesktopConnectionsRegistry: any
}) {
  const {
    bootstrapSshConnection,
    buildRemoteConnection,
    coerceDesktopConnectionConfig,
    decryptDesktopSecret,
    decryptRemoteHeaders,
    encryptDesktopSecret,
    ensureBackend,
    fetchJsonForBackend,
    fetchPublicJson,
    managedConnectionUpdateGate,
    managedPrimaryRestoreOwners,
    managedSshConfig,
    mintGatewayWsTicket,
    primaryProfileKey,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    sshRememberLog,
    startHermes,
    writeDesktopConnectionConfig,
    writeDesktopConnectionsRegistry
  } = deps

function persistSshConnectionToken(profile, source, token, registryConnectionId = '') {
  try {
    const persistence = managedSshTokenPersistencePlan(source, registryConnectionId)
    const id = persistence.registryConnectionId
    const encrypted = encryptDesktopSecret(token)

    // A primary legacy route can also be qualified with a stable registry id.
    // Mirror the adopted per-serve token to both stores so the next primary
    // launch and a later registry-scoped launch reuse the same owned process.
    if (id) {
      const registry = readDesktopConnectionsRegistry()
      const entry = registry.connections.find(c => c.id === id)

      if (entry && entry.kind === 'ssh') {
        writeDesktopConnectionsRegistry(upsertConnection(registry, { ...entry, token: encrypted }))
      }
    }

    if (!persistence.legacySource) {
      return
    }

    const config = readDesktopConnectionConfig()

    if (persistence.legacySource === 'profile') {
      const key = connectionScopeKey(profile)

      if (key && config.profiles?.[key]?.mode === 'ssh') {
        config.profiles[key].token = encrypted
        writeDesktopConnectionConfig(config)
      }
    } else if (config.mode === 'ssh' && config.remote) {
      config.remote.token = encrypted
      writeDesktopConnectionConfig(config)
    }
  } catch (error: any) {
    sshRememberLog(`[ssh] could not persist served token: ${error.message}`)
  }
}

// Resolve the remote backend for a given profile, or null when that profile
// should run a LOCAL backend. Precedence:
//   1. explicit per-profile remote override (connection.json `profiles[name]`)
//   2. env override (HERMES_DESKTOP_REMOTE_URL/_TOKEN) — applies app-wide
//   3. global remote (connection.json `mode: 'remote'`)
// A null/empty profile resolves the env/global remote, so legacy callers and
// the connection test (which pass no profile) are unchanged.
async function resolveRemoteBackend(profile, options: { poolKey?: string; primary?: boolean } = {}) {
  const profileKey = String(profile || '').trim() || 'default'

  const managedPrimary = options.primary
    ? [...managedPrimaryRestoreOwners.values()].find(owner => owner.profile === profileKey)
    : null

  if (managedPrimary) {
    // A managed update is restoring the primary: dial the exact connection
    // snapshot the transaction captured, not whatever routing says now.
    const source = managedPrimary.source
    const sshConfig = managedSshConfig(source, profileKey)

    if (!sshConfig) {
      throw new Error(`SSH connection "${source.label}" has no host configured.`)
    }

    managedConnectionUpdateGate.assertCanDial(source.id, managedPrimary.correlationId)

    const currentRoute = resolveDesktopRemoteRoute({
      config: readDesktopConnectionConfig(),
      env: {
        token: process.env.HERMES_DESKTOP_REMOTE_TOKEN,
        url: process.env.HERMES_DESKTOP_REMOTE_URL
      },
      profile: profileKey,
      registry: readDesktopConnectionsRegistry()
    })

    const persistenceSource =
      currentRoute?.kind === 'ssh' && currentRoute.connectionId === source.id
        ? currentRoute.source
        : `registry:${source.id}`

    const connection = await bootstrapSshConnection(
      persistenceSource === 'profile' ? profileKey : null,
      sshConfig,
      decryptDesktopSecret(source.token),
      persistenceSource,
      undefined,
      {
        managedScope: 'primary',
        managedUpdateCorrelation: managedPrimary.correlationId,
        primaryRegistryScope: true,
        registryConnectionId: source.id
      }
    )

    return { ...connection, connectionId: source.id }
  }

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

  if (!route) {
    return null
  }

  let connection

  if (route.kind === 'ssh') {
    if (route.connectionId) {
      managedConnectionUpdateGate.assertCanDial(route.connectionId)
    }

    connection = await bootstrapSshConnection(
      route.source === 'profile' ? profile : null,
      route.ssh,
      decryptDesktopSecret(route.token),
      route.source,
      undefined,
      {
        managedScope: options.primary ? 'primary' : options.poolKey ? 'pool' : 'transient',
        poolKey: options.poolKey || '',
        primaryRegistryScope: options.primary === true && Boolean(route.connectionId),
        registryConnectionId: route.connectionId || ''
      }
    )
  } else {
    const token =
      route.authMode === 'oauth' ? null : route.source === 'env' ? route.token : decryptDesktopSecret(route.token)

    connection = await buildRemoteConnection(
      route.url,
      route.authMode,
      token,
      route.source,
      undefined,
      route.kind === 'cloud' ? 'cloud' : 'url',
      undefined,
      route.headers
    )
  }

  return route.connectionId ? { ...connection, connectionId: route.connectionId } : connection
}

// A remote profile's sessions live on its remote host's state.db, not on a local
// file the primary can open — so reads for it must route to the remote backend,
// not the local-disk fast path. These three helpers drive that (see
// interceptSessionReadForRemote).
function profileHasRemoteOverride(profile) {
  return profileHasRemoteConnection(readDesktopConnectionConfig(), profile)
}

function configuredRemoteProfileNames() {
  const config = readDesktopConnectionConfig()

  return Object.keys(config.profiles || {}).filter(name => profileHasRemoteConnection(config, name))
}

// True when the app is in app-global remote mode (Settings → "All profiles" →
// Remote/Cloud, or the env override): a SINGLE remote backend serves every
// profile via ?profile=. Cloud counts — it resolves to a remote backend (Q6).
// Distinct from per-profile overrides — here there's one host for all.
function globalRemoteActive() {
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return true
  }

  const mode = readDesktopConnectionConfig().mode

  if (modeIsRemoteLike(mode) || mode === 'ssh') {
    return true
  }

  // Registry-primary transport (#91564/#90316): a registered remote/cloud/ssh
  // gateway promoted to primary via connections.json makes the primary
  // backend remote even while the v1 config.mode still says 'local'. Every
  // consumer of this flag ("one remote host serves every profile") must see
  // that, or the local-entry routes delegate into a primary that now dials
  // remote — respawning the exact loopback children the resolver rung in
  // desktop-remote-route.ts eliminates.
  return registryPrimaryIsRemote()
}

// True when the v2 registry PRIMARY names a non-local connection. Mirrors the
// registry fallback rung in resolveDesktopRemoteRoute.
function registryPrimaryIsRemote() {
  try {
    const registry = readDesktopConnectionsRegistry()
    const entry = registry.connections.find(c => c.id === registry.primary)

    return Boolean(entry && (entry.kind === 'remote' || entry.kind === 'cloud' || entry.kind === 'ssh'))
  } catch {
    return false
  }
}

// True when the PRIMARY profile's backend resolves to a remote/cloud host —
// i.e. resolveRemoteBackend(primaryProfileKey()) would return a descriptor
// rather than null. Mirrors that function's precedence (per-profile override →
// env → global) so a startHermes() failure can be classified as remote (never
// latch — transient, must stay retryable) vs local (latch to break install
// loops) BEFORE the throwing resolve/mint runs.
function primaryBackendIsRemote() {
  return Boolean(profileHasRemoteOverride(primaryProfileKey())) || globalRemoteActive()
}

// GET a profile's resolved backend (remote pool or local primary), parsed JSON.
async function fetchJsonForProfile(profile, path) {
  return requestJsonForProfile(profile, path, 'GET')
}

// Issue an arbitrary method against a profile's resolved backend, parsed JSON.
async function requestJsonForProfile(profile: string, path: string, method: string, body?: string) {
  const conn = await ensureBackend(profile)

  return fetchJsonForBackend(conn, path, { method, body, timeoutMs: DEFAULT_FETCH_TIMEOUT_MS })
}

async function probeRemoteAuthMode(rawUrl) {
  // Determine how a remote gateway expects callers to authenticate, WITHOUT
  // sending any credentials. ``/api/status`` is public on every Hermes
  // gateway (it backs the portal liveness probe) and reports:
  //   auth_required: true  → OAuth gate is engaged (cookie + ws-ticket auth)
  //   auth_required: false → loopback/--insecure: legacy session-token auth
  // ``/api/auth/providers`` (also public, only meaningful when gated) gives
  // the human-facing provider name(s) for the login button label.
  //
  // The settings UI calls this as the user types a URL so it can render an
  // OAuth login button vs a session-token entry box. Network/parse failures
  // surface as ``reachable: false`` rather than throwing, so a half-typed or
  // unreachable URL degrades to "can't tell yet" instead of a hard error.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  let status

  try {
    status = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8_000 })
  } catch (error: any) {
    return {
      baseUrl,
      reachable: false,
      authMode: 'unknown',
      providers: [],
      version: null,
      error: error instanceof Error ? error.message : String(error)
    }
  }

  const authRequired = authModeFromStatus(status) === 'oauth'
  let providers = []

  if (authRequired) {
    // Best-effort: a gated gateway exposes the registered providers so the
    // button can read "Sign in with Nous Research" instead of a generic
    // label, and so a username/password provider can be distinguished from
    // an OAuth-redirect one (``supports_password``). A failure here doesn't
    // change the auth mode, so swallow it.
    try {
      const body = (await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8_000 })) as any

      if (Array.isArray(body?.providers)) {
        providers = body.providers
          .filter(p => p && typeof p === 'object')
          .map(p => ({
            name: String(p.name || ''),
            displayName: String(p.display_name || p.name || ''),
            supportsPassword: Boolean(p.supports_password)
          }))
          .filter(p => p.name)
      }
    } catch {
      // Provider listing is optional metadata; the auth mode is already known.
    }
  }

  return {
    baseUrl,
    reachable: true,
    authMode: authRequired ? 'oauth' : 'token',
    providers,
    version: status?.version || null,
    error: null
  }
}

async function testDesktopConnectionConfig(input: any = {}) {
  if (input.mode === 'ssh') {
    const sshConfig = normalizeSshConfig({
      mode: 'ssh',
      host: input.sshHost,
      user: input.sshUser,
      port: input.sshPort,
      keyPath: input.sshKeyPath,
      remoteHermesPath: input.sshRemoteHermesPath
    })

    if (!sshConfig) {
      return { reachable: false, sshError: 'unreachable', error: 'SSH host is required.' }
    }

    const ssh = createSshProbeConnection(
      { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
      { rememberLog: sshRememberLog }
    )

    try {
      // One bounded retry on TIMEOUT only: a cold Windows backend's first
      // PowerShell exec can exceed the budget (observed live), and a timeout is
      // indeterminate — unlike auth/host-key/unreachable, which are verdicts.
      let attempt = 0

      for (;;) {
        try {
          await ssh.open()
          const platform: any = await detectRemotePlatform(ssh, sshConfig.remoteHermesPath || '')
          let hermesPath
          let hermesVersion
          let supported

          if (platform.os === 'Windows') {
            const runtime = platform
            hermesPath = runtime.hermesPath
            const inspection = await helper(ssh, runtime, 'inspect', [runtime.hermesPath])
            hermesVersion = inspection.version
            supported = inspection.supported
          } else {
            hermesPath = await remoteLifecycle.locateHermes(ssh, sshConfig.remoteHermesPath || '')
            hermesVersion = await remoteLifecycle.probeHermesVersion(ssh, hermesPath)
            supported = await remoteLifecycle.remoteSupportsSshOwnership(ssh, hermesPath)
          }

          if (!supported) {
            return {
              reachable: false,
              sshError: 'update-required',
              error: 'Update Hermes on the remote host before connecting with Desktop SSH.'
            }
          }

          return {
            reachable: true,
            sshError: null,
            error: null,
            remotePlatform: `${platform.os}/${platform.arch}`,
            remoteHermesPath: hermesPath,
            remoteHermesVersion: hermesVersion,
            host: sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host
          }
        } catch (error: any) {
          if (error?.kind === 'timeout' && attempt === 0) {
            attempt += 1
            sshRememberLog('[ssh] test probe timed out once; retrying')

            continue
          }

          throw error
        }
      }
    } catch (error: any) {
      return { reachable: false, sshError: error.kind || 'unknown', error: error.message }
    } finally {
      try {
        await ssh.close()
      } catch {
        void 0
      }
    }
  }

  const config = coerceDesktopConnectionConfig(input, readDesktopConnectionConfig(), { persistToken: false })
  const key = connectionScopeKey(input.profile)
  // The block under test: a per-profile entry or the global remote. Coerce has
  // already normalized the URL and resolved token inheritance for the scope.
  const block = key ? config.profiles?.[key] || null : config.remote

  const wantRemote =
    modeIsRemoteLike(block?.mode) || (!key && modeIsRemoteLike(config.mode)) || (modeIsRemoteLike(input.mode) && block)

  // Test ``/api/status`` through the connection's real auth path. Self-hosted
  // gateways may protect it, and an anonymous success/failure would not prove
  // that the OAuth cookie/native bearer or configured token is reusable. For
  // a remote config we normalize the URL from the input; for local we fall
  // back to the resolved/started backend.
  let baseUrl
  let token = null
  let authMode = 'token'
  let testHeaders = {}

  if (wantRemote && block?.url) {
    baseUrl = normalizeRemoteBaseUrl(block.url)
    authMode = normAuthMode(block.authMode)
    testHeaders = decryptRemoteHeaders(block.headers)

    if (authMode !== 'oauth') {
      token = decryptDesktopSecret(block.token)
    }
  } else {
    const remote = (await resolveRemoteBackend(key)) || (await startHermes())
    baseUrl = remote.baseUrl
    token = remote.token
    authMode = normAuthMode(remote.authMode)
    testHeaders = remote.headers || {}
  }

  const status = (await fetchConnectionStatus(baseUrl, authMode, token, testHeaders)) as any

  // The HTTP status check above proves the backend is reachable, but the chat
  // surface only works once the renderer's live WebSocket to ``/api/ws``
  // connects — a separate transport with separate server-side guards (Host/
  // Origin, ws-ticket/token auth). Validating only the HTTP side produced a
  // false-positive "reachable" while the real boot still failed with "Could not
  // connect to Hermes gateway". Mirror the renderer's connect here so the test
  // reflects the full path the app actually uses.
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, {
    mintTicket: url => mintGatewayWsTicket(url, testHeaders)
  })

  // Skip the WS leg only when the runtime genuinely lacks a WebSocket (so an
  // older Electron/Node never fails the test spuriously); Electron's main
  // process ships a global WebSocket on every supported version.
  if (wsUrl && typeof globalThis.WebSocket === 'function') {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket, headers: testHeaders })

    if (!probe.ok) {
      throw new Error(
        `Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ${probe.reason} ` +
          'The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or gateway auth/origin guard.'
      )
    }
  }

  return {
    ok: true,
    baseUrl,
    version: status?.version || null
  }
}

async function fetchConnectionStatus(baseUrl, authMode, token, headers = {}) {
  // /api/status is public on newer gateways; the subsequent ticket/WS probe
  // remains authoritative. Older gated status routes retain cookie fallback.
  return fetchJsonForBackend({ baseUrl, authMode, token, headers }, '/api/status', { timeoutMs: 8_000 })
}


  return {
    persistSshConnectionToken,
    resolveRemoteBackend,
    profileHasRemoteOverride,
    configuredRemoteProfileNames,
    globalRemoteActive,
    registryPrimaryIsRemote,
    primaryBackendIsRemote,
    fetchJsonForProfile,
    requestJsonForProfile,
    probeRemoteAuthMode,
    testDesktopConnectionConfig,
    fetchConnectionStatus
  }
}
