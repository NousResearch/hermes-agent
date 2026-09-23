import { buildGatewayWsUrl, buildGatewayWsUrlWithTicket, connectionScopeKey, hostLabelFromBaseUrl, localProfileEntry, modeIsRemoteLike, normalizeRemoteBaseUrl, normalizeRemoteHeaders, normalizeSshConfig, normAuthMode, resolveAuthMode, savedProfileSsh, tokenPreview } from './connection-config'
import { resolvePersistedRemoteToken, resolveRemoteTokenPlainText } from './hardening'
import { oauthSessionIsLive } from './native-auth-decisions'
import { resolveRemoteOauthTicket } from './remote-oauth-ticket'

export function createDesktopConnectionDescriptorRuntime(deps: {
  readDesktopConnectionConfig: () => any
  decryptDesktopSecret: (secret: any) => any
  decryptRemoteHeaders: (headers: any) => any
  encryptDesktopSecret: (value: any, options?: any) => any
  probeSecureTokenStorage: () => boolean
  hasNativeSession: (url: string) => boolean
  hasLiveOauthSession: (url: string) => Promise<boolean>
  mintGatewayWsTicket: (url: string, headers?: any) => Promise<any>
  rememberRemoteWsHeaders: (url: string, headers?: any) => void
}) {
  const {
    readDesktopConnectionConfig,
    decryptDesktopSecret,
    decryptRemoteHeaders,
    encryptDesktopSecret,
    probeSecureTokenStorage,
    hasNativeSession,
    hasLiveOauthSession,
    mintGatewayWsTicket,
    rememberRemoteWsHeaders
  } = deps

// Sanitize a connection config into the renderer-facing shape. With no
// `profile` this describes the global/default connection (the existing
// behavior); with a `profile` it describes that profile's per-profile remote
// override (or an empty "local/inherit" view when the profile has none).
async function sanitizeDesktopConnectionConfig(config = readDesktopConnectionConfig(), profile = null) {
  const key = connectionScopeKey(profile)
  const scoped = key ? config.profiles?.[key] || null : null
  const block = key ? scoped || {} : config.remote || {}

  const envOverride = key ? false : Boolean(process.env.HERMES_DESKTOP_REMOTE_URL)
  const savedMode = key ? scoped?.mode : config.mode
  const ssh = savedMode === 'ssh' ? normalizeSshConfig(block) : null

  const savedSsh = savedMode === 'local' ? (key ? savedProfileSsh(config, key) : normalizeSshConfig(block)) : null

  const remoteToken = decryptDesktopSecret(block.token)
  const authMode = normAuthMode(block.authMode)
  const remoteUrl = envOverride ? String(process.env.HERMES_DESKTOP_REMOTE_URL || '') : String(block.url || '')
  const mode = envOverride ? 'remote' : savedMode === 'ssh' ? 'ssh' : modeIsRemoteLike(savedMode) ? savedMode : 'local'

  // Whether the OS keyring (safeStorage) can encrypt the saved token. When
  // false the renderer knows to offer the plain-text opt-in in Settings →
  // Gateway. With keychain encryption opted out (the default) this reports
  // true WITHOUT touching safeStorage — probing is itself a keychain touch
  // that raises the macOS password dialog (see probeSecureTokenStorage).
  const secureTokenStorage = probeSecureTokenStorage()

  // Whether the renderer should warn that the saved token sits in plain text.
  // resolveRemoteTokenPlainText keeps this silent while keychain encryption is
  // opted out (the default) — plain text is the chosen mode there, not a
  // degraded state — and fires only when the token is plain AND the machine
  // cannot secure it (see probeSecureTokenStorage). The env override supplies
  // its token from the environment, so it never reports as plain text here.
  const remoteTokenPlainText = resolveRemoteTokenPlainText({ envOverride, secureTokenStorage, token: block.token })

  let remoteOauthConnected = false

  if (authMode === 'oauth' && remoteUrl) {
    try {
      // Display signal: treat a live RT cookie as "connected" even if the AT
      // cookie has lapsed — the gateway refreshes the AT on the next request,
      // so the session is still usable. A stored native bearer token (cookieless
      // RFC 8252 flow) counts as connected too — otherwise a completed native
      // sign-in shows "not connected" in Settings. The authoritative liveness
      // check is the ws-ticket mint in resolveRemoteBackend at actual connect time.
      remoteOauthConnected = oauthSessionIsLive(hasNativeSession(remoteUrl), await hasLiveOauthSession(remoteUrl))
    } catch {
      remoteOauthConnected = false
    }
  }

  return {
    mode,
    // Echo the scope back so the UI knows which profile (if any) this reflects.
    profile: key,
    remoteAuthMode: authMode,
    remoteOauthConnected,
    remoteUrl,
    // The persisted Hermes Cloud org (slug/id) for a cloud connection, or '' for
    // remote/local. Lets Settings → Gateway reopen into the same org.
    cloudOrg: mode === 'cloud' ? String(block.org || '') : '',
    remoteTokenPreview: tokenPreview(remoteToken),
    remoteTokenSet: Boolean(remoteToken),
    // Whether the OS keyring can encrypt a token; drives the plain-text opt-in
    // affordance in Settings → Gateway on keyring-less Linux.
    secureTokenStorage,
    // Whether the saved token is persisted in plain text while this machine
    // cannot secure it (drives the warning banner in Settings → Gateway).
    remoteTokenPlainText,
    sshHost: (ssh || savedSsh)?.host || '',
    sshUser: (ssh || savedSsh)?.user || '',
    sshPort: (ssh || savedSsh)?.port || null,
    sshKeyPath: (ssh || savedSsh)?.keyPath || '',
    sshRemoteHermesPath: (ssh || savedSsh)?.remoteHermesPath || '',
    sshRemoteProfile: (ssh || savedSsh)?.remoteProfile || '',
    // The env override only forces the global/primary connection; a per-profile
    // scope is never overridden by HERMES_DESKTOP_REMOTE_URL.
    envOverride
  }
}

// Build + validate a `{ url, authMode, token }` remote block. OAuth gateways
// authenticate via the login-window session cookie (verified at connect time in
// resolveRemoteBackend), so only token-auth remotes require a saved token.
// `org` (optional) is the Hermes Cloud org slug/id the instance was discovered
// under — persisted so Settings can reopen into the same org; omitted from the
// block when empty so plain remote connections stay unchanged.
function buildRemoteBlock(remoteUrl, authMode, token, org?: string, headers?: object, name?: string) {
  if (authMode !== 'oauth' && !decryptDesktopSecret(token)) {
    throw new Error('Remote gateway session token is required.')
  }

  const block: { url: string; authMode: string; token: object; headers?: object; org?: string; name?: string } = {
    url: normalizeRemoteBaseUrl(remoteUrl),
    authMode,
    token
  }

  const remoteHeaders = normalizeRemoteHeaders(headers)

  if (Object.keys(remoteHeaders).length > 0) {
    block.headers = remoteHeaders
  }

  const nameValue = typeof name === 'string' ? name.trim() : ''

  if (nameValue) {
    block.name = nameValue
  }

  const orgValue = typeof org === 'string' ? org.trim() : ''

  if (orgValue) {
    block.org = orgValue
  }

  return block
}

function coerceDesktopConnectionConfig(input: any = {}, existing = readDesktopConnectionConfig(), options: any = {}) {
  const persistToken = options.persistToken !== false
  const key = connectionScopeKey(input.profile)
  // 'cloud' and 'remote' both persist a remote-shaped block; 'cloud' is
  // remembered as its own provenance (Q6) and resolves to remote downstream.
  // Anything else collapses to local.
  const mode = input.mode === 'ssh' ? 'ssh' : modeIsRemoteLike(input.mode) ? input.mode : 'local'
  const remoteLike = modeIsRemoteLike(mode)

  // The block being edited: a per-profile entry or the global remote block.
  const rawExistingBlock = key ? existing.profiles?.[key] || {} : existing.remote || {}
  // Leaving a CLOUD connection unselects it: a cloud block's url/org/token
  // describe a discovered Hermes Cloud instance, NOT a user-owned remote gateway,
  // so switching to local or remote must NOT inherit them (otherwise the stale
  // cloud URL lingers and re-selecting Cloud looks "already connected"). When the
  // saved block was cloud and the new mode is not cloud, start from an empty
  // block. (remote↔local toggles still preserve a real remote URL as before.)
  const existingMode = key ? existing.profiles?.[key]?.mode : existing.mode
  const leavingCloud = existingMode === 'cloud' && mode !== 'cloud'
  const leavingSsh = rawExistingBlock.mode === 'ssh' && mode !== 'ssh' && mode !== 'local'
  const existingBlock = leavingCloud || leavingSsh ? {} : rawExistingBlock
  const remoteUrl = String(input.remoteUrl ?? existingBlock.url ?? '').trim()
  // authMode: explicit input wins; otherwise inherit the saved value, default 'token'.
  const authMode = resolveAuthMode(input.remoteAuthMode, existingBlock.authMode)
  // Cloud org: only meaningful for 'cloud' mode. Explicit input wins; otherwise
  // inherit the saved org. A plain 'remote' connection never carries an org
  // (switching cloud→remote drops it), so it stays unset unless mode is cloud.
  const cloudOrg = mode === 'cloud' ? String(input.cloudOrg ?? existingBlock.org ?? '').trim() : ''

  // A saved name belongs to this exact gateway, not another instance in the same org.
  const cloudName =
    mode === 'cloud'
      ? String(
          input.cloudName ??
            (existingBlock.url && normalizeRemoteBaseUrl(remoteUrl) === normalizeRemoteBaseUrl(existingBlock.url)
              ? existingBlock.name
              : '') ??
            ''
        ).trim()
      : ''

  const incomingToken = typeof input.remoteToken === 'string' ? input.remoteToken.trim() : ''

  const remoteHeaders =
    input.remoteHeaders && typeof input.remoteHeaders === 'object' ? input.remoteHeaders : existingBlock.headers

  // Persist decision lives in hardening.resolvePersistedRemoteToken so the
  // IPC-propagation seam (allowPlainTextToken → encryptDesktopSecret opt-in) is
  // covered by a focused regression test. Pass allowPlainText through RAW — the
  // helper coerces with `=== true`, so a truthy-non-true value never enables
  // plain-text storage, and that strictness is asserted in exactly one place.
  const nextToken = resolvePersistedRemoteToken({
    incomingToken,
    persistToken,
    existingToken: existingBlock.token,
    allowPlainText: input.allowPlainTextToken,
    encryptSecret: encryptDesktopSecret
  })

  if (mode === 'ssh') {
    const sshBlock = buildSshBlock(input, savedProfileSsh(existing, key) || rawExistingBlock)

    if (key) {
      const profiles = { ...(existing.profiles || {}), [key]: sshBlock }

      return {
        mode: existing.mode === 'ssh' || modeIsRemoteLike(existing.mode) ? existing.mode : 'local',
        remote: existing.remote || {},
        profiles
      }
    }

    return { mode: 'ssh', remote: sshBlock, profiles: existing.profiles || {} }
  }

  if (key) {
    // Per-profile scope: a remote/cloud entry pins this profile to its own
    // backend; a local entry clears the override so the profile inherits the
    // default. The mode tag (remote vs cloud) is preserved on the entry.
    const profiles = { ...(existing.profiles || {}) }

    if (remoteLike) {
      profiles[key] = {
        mode,
        ...buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg, remoteHeaders, cloudName)
      }
    } else {
      const localEntry = localProfileEntry(rawExistingBlock)

      if (localEntry) {
        profiles[key] = localEntry
      } else {
        delete profiles[key]
      }
    }

    return {
      mode: existing.mode === 'ssh' || modeIsRemoteLike(existing.mode) ? existing.mode : 'local',
      remote: existing.remote || {},
      profiles
    }
  }

  const nextRemote = remoteLike
    ? buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg, remoteHeaders, cloudName)
    : existingMode === 'ssh'
      ? rawExistingBlock
      : { url: remoteUrl ? normalizeRemoteBaseUrl(remoteUrl) : remoteUrl, authMode, token: nextToken }

  // Preserve per-profile overrides when saving the global connection.
  return { mode, remote: nextRemote, profiles: existing.profiles || {} }
}

// Build an SSH connection block from a save payload, preserving an
// already-adopted dashboard token from the existing block (the token is minted
// + reconciled at bootstrap, never user-entered). `mode: 'ssh'` is stamped so
// normalizeSshConfig/profileSshOverride recognize it.
function buildSshBlock(input: any, existingBlock: any = {}) {
  // `??` (not `||`) so an explicit '' (user CLEARED the field) wins over the
  // saved value; only a truly absent (undefined) field inherits.
  const merged = normalizeSshConfig({
    mode: 'ssh',
    host: input.sshHost ?? existingBlock.host,
    user: input.sshUser ?? existingBlock.user,
    port: input.sshPort ?? existingBlock.port,
    keyPath: input.sshKeyPath ?? existingBlock.keyPath,
    remoteHermesPath: input.sshRemoteHermesPath ?? existingBlock.remoteHermesPath,
    remoteProfile: input.sshRemoteProfile ?? existingBlock.remoteProfile
  })

  if (!merged) {
    throw new Error('SSH host is required.')
  }

  // Carry forward an already-adopted dashboard token unless the host changed
  // (a different host invalidates the old dashboard's token).
  if (existingBlock.token && existingBlock.host === merged.host) {
    merged.token = existingBlock.token
  }

  return merged
}

// Build a remote backend connection descriptor from an already-resolved remote
// config. Handles both auth models (OAuth ws-ticket vs static session token)
// and is shared by the per-profile, env, and global resolution paths. `token`
// is the DECRYPTED static token (or null in OAuth mode). `source` is a label
// for diagnostics ('profile' | 'env' | 'settings').
async function buildRemoteConnection(
  rawUrl,
  authMode,
  token,
  source,
  remoteHost?,
  remoteKind = 'url',
  remoteIdentity?,
  headers?
) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)
  const remoteHeaders = decryptRemoteHeaders(headers)
  // For token/oauth remotes the meaningful host is the real backend URL; for
  // SSH remotes the caller passes the entered/resolved host explicitly (the
  // baseUrl is a 127.0.0.1 tunnel and would be useless in the pill).
  const host = remoteHost || hostLabelFromBaseUrl(baseUrl)

  if (authMode === 'oauth') {
    const ticket = await resolveRemoteOauthTicket(baseUrl, remoteHeaders, {
      hasNativeSession,
      mintGatewayWsTicket
    })

    const wsUrl = buildGatewayWsUrlWithTicket(baseUrl, ticket)

    rememberRemoteWsHeaders(wsUrl, remoteHeaders)

    return {
      baseUrl,
      mode: 'remote',
      source,
      authMode: 'oauth',
      remoteHost: host || undefined,
      remoteIdentity,
      remoteKind,
      headers: remoteHeaders,
      // No static token in OAuth mode; REST is cookie-authed via the partition.
      token: null,
      wsUrl
    }
  }

  if (!token) {
    throw new Error(
      'Remote Hermes gateway is selected, but no session token is saved. ' +
        'Open Settings → Gateway and save a token, or switch back to Local.'
    )
  }

  const wsUrl = buildGatewayWsUrl(baseUrl, token)

  rememberRemoteWsHeaders(wsUrl, remoteHeaders)

  return {
    baseUrl,
    mode: 'remote',
    source,
    authMode: 'token',
    remoteHost: host || undefined,
    remoteIdentity,
    remoteKind,
    headers: remoteHeaders,
    token,
    wsUrl
  }
}


  return { sanitizeDesktopConnectionConfig, coerceDesktopConnectionConfig, buildRemoteConnection }
}
