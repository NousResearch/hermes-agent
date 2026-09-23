import fs from 'node:fs'
import path from 'node:path'

import { connectionInstallIds, evictConnectionCaches } from './connection-caches'
import { modeIsRemoteLike, normalizeRemoteHeaders, normalizeSshConfig, normAuthMode, sanitizeRemoteHeaderValue, tokenPreview } from './connection-config'
import { connectionDialFieldsChanged, mergeConnectionInput, migrateV1ToRegistry, normalizeConnectionInput, normalizeRegistry, reconcileRegistryDrift, upsertConnection } from './connection-registry'
import { encryptDesktopSecret as encryptDesktopSecretStrict, resolvePersistedRemoteToken, SAFE_STORAGE_ENCODING, tightenSecretFileMode, writeSecretFileAtomic } from './hardening'
import type { NativeTokenStoreIo } from './native-token-store'
import { attachRemoteRequestHeaderListener, collectRemoteHeaderSources, createRemoteWsHeaderStore, resolveRemoteRequestHeaders } from './remote-ws-headers'
import { classifyStoredSecret, readSecretStoragePolicy, SECRET_STORAGE_POLICY_FILE, type SecretStoragePolicy, writeSecretStoragePolicy } from './secret-storage-policy'

export function createDesktopConnectionStorageRuntime(deps: {
  app: { getPath: (name: string) => string }
  safeStorage: any
  session: any
  connectionConfigPath: string
  connectionsRegistryPath: string
  profileNameRe: RegExp
  nativeTokenStoreIo: () => NativeTokenStoreIo
  rememberLog: (message: string) => void
  assertCanMutateRegistryConnection: (id: string) => void
  stopRegistryConnectionBackends: (id: string) => Promise<void>
  broadcastConnectionsChanged: (payload: { connectionId: string; reason: 'removed' | 'saved' | 'updated' }) => void
}) {
  const { app, safeStorage, session, rememberLog, stopRegistryConnectionBackends, broadcastConnectionsChanged } = deps
  const DESKTOP_CONNECTION_CONFIG_PATH = deps.connectionConfigPath
  const DESKTOP_CONNECTIONS_REGISTRY_PATH = deps.connectionsRegistryPath
  const PROFILE_NAME_RE = deps.profileNameRe
  const _nativeTokenStoreIo = deps.nativeTokenStoreIo
  const managedConnectionUpdateGate = { assertCanMutate: deps.assertCanMutateRegistryConnection }
let connectionConfigCache = null
let connectionConfigCacheMtime = null
let connectionRegistryCache = null
let connectionRegistryCacheMtime = null
const remoteHeaderSessions = new WeakSet<object>()
const remoteWsHeaderStore = createRemoteWsHeaderStore()
const SECRET_STORAGE_POLICY_PATH = path.join(app.getPath('userData'), SECRET_STORAGE_POLICY_FILE)

const _secretStoragePolicyIo = {
  readText: () => fs.readFileSync(SECRET_STORAGE_POLICY_PATH, 'utf8'),
  writeText: (text: string) => writeSecretFileAtomic(SECRET_STORAGE_POLICY_PATH, text, { encoding: 'utf8' })
}

let _secretStoragePolicy: SecretStoragePolicy | null = null

function secretStoragePolicy(): SecretStoragePolicy {
  if (!_secretStoragePolicy) {
    _secretStoragePolicy = readSecretStoragePolicy(_secretStoragePolicyIo)
  }

  return _secretStoragePolicy
}

function setSecretStoragePolicy(next: SecretStoragePolicy) {
  _secretStoragePolicy = { on: next.on === true, migrated: next.migrated === true }
  writeSecretStoragePolicy(_secretStoragePolicy, _secretStoragePolicyIo)
}

/**
 * Keychain availability as the renderer should see it. With encryption
 * opted out this must NOT probe safeStorage — isEncryptionAvailable() is
 * itself a keychain touch that raises the macOS dialog this feature exists
 * to avoid. We report `true` so no plain-text warning banners fire: storing
 * plaintext is the user's chosen (default) mode, not a degraded state.
 */
function probeSecureTokenStorage(): boolean {
  if (!secretStoragePolicy().on) {
    return true
  }

  try {
    return Boolean(safeStorage.isEncryptionAvailable())
  } catch {
    return false
  }
}

/**
 * Rewrite every stored desktop secret (v1 connection.json token/headers +
 * per-profile overrides, v2 registry connections, native OAuth token store)
 * through `reencode`. Returns true when any store was rewritten. Shared by
 * the one-shot legacy migration and the Settings encryption toggle.
 */
function rewriteAllStoredSecrets(shouldRewrite: (secret: any) => boolean, reencode: (secret: any) => any): boolean {
  let touched = false

  const rewriteBlock = (block: any) => {
    if (!block || typeof block !== 'object') {
      return block
    }

    const next = { ...block, ...(block.token ? { token: reencode(block.token) } : {}) }

    if (block.headers && typeof block.headers === 'object') {
      next.headers = Object.fromEntries(Object.entries(block.headers).map(([k, v]) => [k, reencode(v)]))
    }

    return next
  }

  const blockNeedsRewrite = (o: any) =>
    shouldRewrite(o?.token) ||
    Object.values(o?.headers && typeof o.headers === 'object' ? o.headers : {}).some(shouldRewrite)

  // v1 connection.json.
  const config = readDesktopConnectionConfig()

  if (blockNeedsRewrite(config.remote) || Object.values(config.profiles || {}).some(blockNeedsRewrite)) {
    touched = true
    writeDesktopConnectionConfig({
      ...config,
      remote: rewriteBlock(config.remote),
      profiles: Object.fromEntries(Object.entries(config.profiles || {}).map(([k, v]) => [k, rewriteBlock(v)]))
    })
  }

  // v2 connections.json registry.
  const registry = readDesktopConnectionsRegistry()

  if (registry.connections?.some(blockNeedsRewrite)) {
    touched = true
    writeDesktopConnectionsRegistry({ ...registry, connections: registry.connections.map(rewriteBlock) })
  }

  // Native OAuth token store: baseUrl → blob.
  const io = _nativeTokenStoreIo()

  try {
    const store = JSON.parse(io.readStoreText())

    if (store && typeof store === 'object' && !Array.isArray(store)) {
      const entries = Object.entries(store)

      if (entries.some(([, v]) => shouldRewrite(v))) {
        touched = true
        io.writeStoreText(JSON.stringify(Object.fromEntries(entries.map(([k, v]) => [k, reencode(v)]))))
      }
    }
  } catch {
    // Missing/corrupt native token store: nothing to rewrite.
  }

  return touched
}

/**
 * One-shot legacy migration: builds before the opt-in policy wrote every
 * secret as a safeStorage blob. With encryption now defaulting OFF, decrypt
 * each stored blob once and rewrite it as plain so no future launch touches
 * the keychain. Marked `migrated` whether or not every blob decrypts — a
 * broken keychain costs at most ONE prompt (this pass), never one per
 * launch; blobs that would not decrypt are left in place and simply read as
 * absent from then on (classifyStoredSecret → 'drop'), so opting encryption
 * back ON later can still recover them on a healthy keychain.
 *
 * Runs before createWindow() so every later read sees the final encodings.
 */
function migrateLegacyEncryptedSecretsOnce() {
  const policy = secretStoragePolicy()

  if (policy.on || policy.migrated) {
    return
  }

  const needsMigration = (secret: any) => classifyStoredSecret(secret, policy) === 'migrate'

  const reencode = (secret: any) => {
    if (!needsMigration(secret)) {
      return secret
    }

    const plaintext = decryptDesktopSecret(secret)

    // Undecryptable now (locked/absent keychain): keep the blob for a
    // potential future opt-in, but post-migration reads treat it as unset.
    return plaintext ? { encoding: 'plain', value: plaintext } : secret
  }

  let touchedKeychain = false

  try {
    touchedKeychain = rewriteAllStoredSecrets(needsMigration, reencode)
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)

    rememberLog(`[secret-storage] legacy migration pass failed: ${detail}`)
  }

  setSecretStoragePolicy({ on: false, migrated: true })

  if (touchedKeychain) {
    rememberLog('[secret-storage] migrated legacy keychain-encrypted secrets to opt-out storage (one-shot pass)')
  }
}

/**
 * Settings → Gateway toggle: flip keychain-backed encryption and re-encode
 * every stored secret to match. Turning ON encrypts plain blobs through
 * strict safeStorage (throws loudly when the keychain is unusable — the
 * toggle stays off and the renderer shows the error). Turning OFF decrypts
 * back to plain; this is user-initiated, so a keychain prompt here is
 * expected and acceptable.
 */
function applySecretStorageEncryption(on: boolean) {
  const enable = on === true

  if (secretStoragePolicy().on === enable) {
    return { on: enable }
  }

  if (enable) {
    const needsEncrypt = (secret: any) => secret?.encoding === 'plain' && Boolean(secret.value)

    // Probe FIRST so an unusable keychain fails before any store is touched.
    if (
      !(() => {
        try {
          return Boolean(safeStorage.isEncryptionAvailable())
        } catch {
          return false
        }
      })()
    ) {
      throw new Error(
        'OS keychain encryption is unavailable on this machine, so stored gateway secrets cannot be encrypted.'
      )
    }

    setSecretStoragePolicy({ on: true, migrated: true })

    try {
      rewriteAllStoredSecrets(needsEncrypt, secret =>
        needsEncrypt(secret) ? encryptDesktopSecretStrict(String(secret.value), safeStorage) : secret
      )
    } catch (error) {
      // Encryption failed midway: revert the policy so reads keep working
      // against whatever encodings are on disk (mixed stores read fine —
      // decryptDesktopSecret handles both encodings under either policy).
      setSecretStoragePolicy({ on: false, migrated: true })
      throw error
    }

    return { on: true }
  }

  // Turning OFF: decrypt everything back to plain while the keychain is
  // still readable, then flip the policy.
  const needsDecrypt = (secret: any) => secret?.encoding === SAFE_STORAGE_ENCODING

  rewriteAllStoredSecrets(needsDecrypt, (secret: any) => {
    if (!needsDecrypt(secret)) {
      return secret
    }

    const plaintext = decryptDesktopSecret(secret)

    return plaintext ? { encoding: 'plain', value: plaintext } : secret
  })

  setSecretStoragePolicy({ on: false, migrated: true })

  return { on: false }
}

function encryptDesktopSecret(value, options = {}) {
  if (!secretStoragePolicy().on) {
    const raw = String(value || '')

    return raw ? { encoding: 'plain', value: raw } : null
  }

  return encryptDesktopSecretStrict(value, safeStorage, options)
}

function decryptDesktopSecret(secret) {
  if (!secret || typeof secret !== 'object') {
    return ''
  }

  const value = String(secret.value || '')

  if (!value) {
    return ''
  }

  if (secret.encoding === SAFE_STORAGE_ENCODING) {
    // Legacy blob under an opted-out policy: once the one-shot migration pass
    // has run, never touch safeStorage again — a dead keychain would otherwise
    // prompt on every read. Before that pass, decryption is allowed so the
    // migration itself (and this launch's reads) can recover the value.
    if (classifyStoredSecret(secret, secretStoragePolicy()) === 'drop') {
      return ''
    }

    try {
      return safeStorage.decryptString(Buffer.from(value, 'base64'))
    } catch {
      return ''
    }
  }

  // Any other encoding (a hand-edited config, or one written by a pre-release
  // build) is returned verbatim on purpose: this fallback is what lets such a
  // config connect at all. Not a plaintext-writing path — nothing in this file
  // persists a token this way.
  return value
}

function decryptRemoteHeaders(headers) {
  const normalized = normalizeRemoteHeaders(headers)
  const out = {}

  for (const [name, secret] of Object.entries(normalized)) {
    // Sanitize AFTER decryption as well as at ingest: a safeStorage envelope
    // stores ciphertext, so normalizeRemoteHeaders never sees its plaintext.
    // This is the single funnel every consumer of header values goes through
    // (login window extraHeaders, onBeforeSendHeaders, electronNet setHeader,
    // and both connection-test paths), so CR/LF can't reach a request here.
    const value = sanitizeRemoteHeaderValue(decryptDesktopSecret(secret))

    if (value) {
      out[name] = value
    }
  }

  return out
}

/**
 * Turn an editor payload of remote gateway headers into stored secret
 * envelopes. The payload map is authoritative (a name missing from it is
 * cleared); per-name values are:
 *   - non-empty string  → new plaintext value, encrypted like a token
 *   - null              → keep the currently stored envelope for that name
 *                         (the editor shows a set-but-hidden secret)
 *   - envelope object   → stored verbatim (hand-edited import path)
 * Name filtering (forbidden/managed headers) happens in
 * normalizeRemoteHeaders at the registry/config layer.
 */
function encryptIncomingRemoteHeaders(raw, existing, options: { allowPlainText?: boolean } = {}) {
  const out = {}
  const stored = normalizeRemoteHeaders(existing)

  for (const [name, value] of Object.entries(raw || {})) {
    const key = String(name || '').trim()

    if (!key) {
      continue
    }

    if (typeof value === 'string') {
      const trimmed = value.trim()

      if (trimmed) {
        out[key] = encryptDesktopSecret(trimmed, { allowPlainText: options.allowPlainText === true })
      }

      continue
    }

    if (value === null) {
      if (stored[key]) {
        out[key] = stored[key]
      }

      continue
    }

    if (value && typeof value === 'object') {
      out[key] = value
    }
  }

  return out
}

function rememberRemoteWsHeaders(wsUrl, headers = {}) {
  remoteWsHeaderStore.remember(wsUrl, headers)
}

// Decrypted header sources, memoized against the two config caches this
// process already keys off mtime. onBeforeSendHeaders now runs on every OAuth
// partition as well as defaultSession, so without this every subresource
// request would decrypt EVERY registry connection's headers — and a
// safeStorage-encoded value costs a keychain round-trip per read.
// Both readers refresh their cache object whenever the file mtime moves, so
// identity comparison on the cached objects is a correct staleness check.
let remoteHeaderSourcesCache: any = null
let remoteHeaderSourcesConfigKey: any = null
let remoteHeaderSourcesRegistryKey: any = null

function remoteHeaderSources() {
  const config = readDesktopConnectionConfig()
  const registry = readDesktopConnectionsRegistry()

  if (
    remoteHeaderSourcesCache &&
    remoteHeaderSourcesConfigKey === config &&
    remoteHeaderSourcesRegistryKey === registry
  ) {
    return remoteHeaderSourcesCache
  }

  const sources = collectRemoteHeaderSources({
    connections: (registry?.connections || []).map(entry => ({
      kind: entry.kind,
      url: entry.url,
      headers: decryptRemoteHeaders(entry.headers)
    })),
    v1Remote:
      modeIsRemoteLike(config.mode) && config.remote?.url
        ? { url: config.remote.url, headers: decryptRemoteHeaders(config.remote.headers) }
        : null
  })

  remoteHeaderSourcesCache = sources
  remoteHeaderSourcesConfigKey = config
  remoteHeaderSourcesRegistryKey = registry

  return sources
}

function headersForRemoteRequest(requestUrl) {
  return resolveRemoteRequestHeaders(requestUrl, {
    exactHeaders: remoteWsHeaderStore.headersFor(requestUrl),
    sources: remoteHeaderSources()
  })
}

function installRemoteHeaderRulesOnSession(sess) {
  if (!sess || remoteHeaderSessions.has(sess)) {
    return
  }

  remoteHeaderSessions.add(sess)
  attachRemoteRequestHeaderListener(sess, headersForRemoteRequest)
}

function installRemoteHeaderRules() {
  installRemoteHeaderRulesOnSession(session.defaultSession)
}

// Validate + normalize the per-profile remote overrides map read from disk.
// Drops malformed names/entries and keeps only the recognized fields so a
// hand-edited or stale connection.json can't inject junk into resolution.
function sanitizeConnectionProfiles(raw: Record<string, any>) {
  if (!raw || typeof raw !== 'object') {
    return {}
  }

  const out = {}

  for (const [name, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    if (name !== 'default' && !PROFILE_NAME_RE.test(name)) {
      continue
    }

    if (entry.mode === 'ssh') {
      const ssh = normalizeSshConfig(entry)

      if (ssh) {
        if (entry.token && typeof entry.token === 'object') {
          ssh.token = entry.token
        }

        out[name] = ssh
      }

      continue
    }

    const cleaned: {
      mode: 'remote' | 'local' | 'cloud'
      url?: string
      authMode?: string
      token?: object
      headers?: object
      org?: string
      name?: string
      savedSsh?: object
    } = {
      mode: modeIsRemoteLike(entry.mode) ? entry.mode : 'local'
    }

    if (cleaned.mode === 'local') {
      const savedSsh = normalizeSshConfig(entry.savedSsh)

      if (savedSsh) {
        cleaned.savedSsh = savedSsh
      }
    }

    const url = String(entry.url || '').trim()

    if (url) {
      cleaned.url = url
    }

    cleaned.authMode = normAuthMode(entry.authMode)

    if ((entry as any).token && typeof entry.token === 'object') {
      cleaned.token = entry.token
    }

    const headers = normalizeRemoteHeaders((entry as any).headers)

    if (Object.keys(headers).length > 0) {
      cleaned.headers = headers
    }

    // Preserve the Hermes Cloud org tag on cloud-mode entries so Settings can
    // reopen into the same org for a per-profile cloud connection.
    if (cleaned.mode === 'cloud') {
      const cloudName = String(entry.name || '').trim()

      if (cloudName) {
        cleaned.name = cloudName
      }

      const org = String(entry.org || '').trim()

      if (org) {
        cleaned.org = org
      }
    }

    out[name] = cleaned
  }

  return out
}

function readDesktopConnectionConfig() {
  // Check if file changed on disk since last read (e.g. modified by another
  // process or an external tool).  Our own writes update the cache inline
  // via writeDesktopConnectionConfig, but external changes would be missed.
  let mtime = null

  try {
    mtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs
  } catch {
    mtime = null
  }

  if (connectionConfigCache && connectionConfigCacheMtime === mtime) {
    return connectionConfigCache
  }

  let config = { mode: 'local', remote: {}, profiles: {} }

  try {
    const raw = fs.readFileSync(DESKTOP_CONNECTION_CONFIG_PATH, 'utf8')
    // Tighten an install written before this file was owner-only. Every write
    // now goes out at 0600, but a file already on disk keeps its old 0644 bits
    // until something chmods it, and waiting for the user's next Settings save
    // would leave it group/other-readable indefinitely. Runs on a cache miss
    // only (once per launch, plus after an external edit); chmod moves ctime,
    // not mtime, so it cannot invalidate the cache it sits inside.
    //
    // Deliberately BEFORE JSON.parse, not after: a truncated or hand-mangled
    // connection.json still contains the token bytes, and parse throws into the
    // catch below, which swallows the error and falls back to local mode. With
    // the tighten after the parse, exactly the file that is both corrupt AND
    // world-readable would be the one file never tightened — and nothing would
    // ever retry it, because the fallback config is not written back. The chmod
    // needs only the path, so it has no reason to wait for valid JSON.
    tightenSecretFileMode(DESKTOP_CONNECTION_CONFIG_PATH)

    const parsed = JSON.parse(raw)

    // NOT done here: migrating a legacy non-safeStorage token payload to
    // ciphertext at rest. Deferred deliberately — it has to honor the opt-in
    // plaintext choice PR #62319 adds (re-encrypting it converts a portable
    // credential into a keychain-bound one and can lose the token), write
    // through sanitizeConnectionProfiles below rather than persisting raw
    // `parsed`, and tell the user to ROTATE, since every existing backup copy
    // still holds the old secret. Do not add it without those three.

    if (parsed && typeof parsed === 'object') {
      const remote = parsed.remote && typeof parsed.remote === 'object' ? parsed.remote : {}
      // authMode lives on the remote sub-object: 'oauth' (cookie + ws-ticket)
      // or 'token' (legacy static session token). Default to 'token' for
      // backward compatibility with configs written before OAuth support.
      remote.authMode = remote.authMode === 'oauth' ? 'oauth' : 'token'
      config = {
        mode: parsed.mode === 'ssh' ? 'ssh' : modeIsRemoteLike(parsed.mode) ? parsed.mode : 'local',
        remote,
        // Per-profile remote overrides: each profile may point at its own
        // backend (local spawn or its own remote URL). Preserved verbatim so
        // profileRemoteOverride() can resolve them; normalized lazily on save.
        profiles: sanitizeConnectionProfiles(parsed.profiles)
      }
    }
  } catch {
    // Missing or malformed connection settings should fall back to local.
  }

  connectionConfigCache = config
  connectionConfigCacheMtime = mtime

  return config
}

function writeDesktopConnectionConfig(config) {
  fs.mkdirSync(path.dirname(DESKTOP_CONNECTION_CONFIG_PATH), { recursive: true })
  // Owner-only, not writeFileAtomic: this is the single choke point for every
  // connection.json write (the IPC save/apply handlers and
  // persistSshConnectionToken all land here), and the file carries the
  // safeStorage-encrypted gateway token plus its URL and SSH host/user/keyPath.
  // safeStorage keeps the token opaque; 0600 keeps the whole record — and the
  // fields that are NOT encrypted — off other local accounts, matching
  // native-oauth-tokens.json and desktop-installation.json.
  writeSecretFileAtomic(DESKTOP_CONNECTION_CONFIG_PATH, JSON.stringify(config, null, 2))
  connectionConfigCache = config
  connectionConfigCacheMtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs
}

// ── v2 connection registry (multi-source) ──────────────────────────────────

/**
 * Read the v2 registry, importing from v1 connection.json exactly once (when
 * connections.json does not exist yet). Same mtime-cache + tighten-mode
 * discipline as readDesktopConnectionConfig; a corrupt registry degrades to
 * local-only via normalizeRegistry rather than throwing at boot.
 *
 * An EXISTING registry is additionally reconciled against v1 when the two have
 * drifted — see reconcileRegistryDrift. The one-shot migration cannot cover a
 * user who registered nothing and then pointed Settings -> Gateway at a remote,
 * and until that heals, every launch re-homes them onto a local backend.
 */
function readDesktopConnectionsRegistry() {
  let mtime = null

  try {
    mtime = fs.statSync(DESKTOP_CONNECTIONS_REGISTRY_PATH).mtimeMs
  } catch {
    mtime = null
  }

  if (connectionRegistryCache && connectionRegistryCacheMtime === mtime) {
    return connectionRegistryCache
  }

  let registry

  if (mtime === null) {
    // First run on this build: import the v1 single-connection config. The v1
    // file is NOT modified or deleted — older builds keep reading it. The
    // migration is deterministic over the v1 input, so even if two processes
    // race the first run (updater relaunch, second window), both derive the
    // same registry and the later atomic write is a no-op content-wise.
    registry = migrateV1ToRegistry(readDesktopConnectionConfig())

    try {
      writeDesktopConnectionsRegistry(registry)
    } catch {
      // Write failed (full disk, read-only userData). Keep the migrated
      // registry in memory so list/save keep working this session instead of
      // hard-failing every hermes:connections:* call.
      connectionRegistryCache = registry
      connectionRegistryCacheMtime = null
    }

    return connectionRegistryCache
  }

  try {
    // Same rationale as connection.json: tighten BEFORE parse so a corrupt
    // file that still holds token bytes gets its mode fixed anyway.
    tightenSecretFileMode(DESKTOP_CONNECTIONS_REGISTRY_PATH)
    registry = normalizeRegistry(JSON.parse(fs.readFileSync(DESKTOP_CONNECTIONS_REGISTRY_PATH, 'utf8')))
  } catch {
    // Whole-file corruption (truncated write, mangled hand-edit). The
    // degraded local-only registry keeps boot working, but the file BYTES are
    // the user's connection data — preserve them in a sidecar BEFORE any
    // later write (drift reconcile, connection save) overwrites the file
    // (#94246: recovery must never be data loss).
    preserveCorruptRegistrySidecar()
    registry = normalizeRegistry(null)
  }

  if (registry?.quarantined?.length) {
    rememberLog(
      `[connections] ${registry.quarantined.length} malformed registry entr${registry.quarantined.length === 1 ? 'y was' : 'ies were'} quarantined (kept under "quarantined" in connections.json); healthy connections loaded normally.`
    )
  }

  // Heal v1 -> v2 drift: the v1 global route names a remote this registry has
  // never heard of, so the live descriptor resolves to no connectionId and the
  // launch pick sends the window somewhere else. Persist so the repair is a
  // one-time event rather than a recomputation on every read; a failed write
  // still returns the healed registry for this session.
  const reconciled = reconcileRegistryDrift(registry, readDesktopConnectionConfig())

  if (reconciled.changed) {
    registry = reconciled.registry

    try {
      writeDesktopConnectionsRegistry(registry)

      return connectionRegistryCache
    } catch {
      connectionRegistryCache = registry
      connectionRegistryCacheMtime = null

      return registry
    }
  }

  connectionRegistryCache = registry
  connectionRegistryCacheMtime = mtime

  return registry
}

// Copy an unparseable connections.json aside (once per corruption event) so a
// later registry write can never destroy the only copy of the user's saved
// connections (#94246). Best effort: failure to preserve must not block boot.
function preserveCorruptRegistrySidecar() {
  try {
    const rawText = fs.readFileSync(DESKTOP_CONNECTIONS_REGISTRY_PATH, 'utf8')

    if (!rawText.trim()) {
      return
    }

    const sidecar = `${DESKTOP_CONNECTIONS_REGISTRY_PATH}.corrupt-${new Date().toISOString().replace(/[:.]/g, '-')}`

    if (!fs.existsSync(sidecar)) {
      fs.writeFileSync(sidecar, rawText, { mode: 0o600 })
    }

    rememberLog(
      `[connections] connections.json could not be parsed; preserved the original file at ${sidecar} and continuing with a local-only registry. No connection data was deleted.`
    )
  } catch {
    // The read itself failed (missing file, permissions) — nothing to save.
  }
}

function writeDesktopConnectionsRegistry(registry) {
  fs.mkdirSync(path.dirname(DESKTOP_CONNECTIONS_REGISTRY_PATH), { recursive: true })
  // Owner-only for the same reason as connection.json: entries carry
  // safeStorage-encrypted tokens plus URLs and SSH host/user/keyPath.
  writeSecretFileAtomic(DESKTOP_CONNECTIONS_REGISTRY_PATH, JSON.stringify(registry, null, 2))
  connectionRegistryCache = registry
  connectionRegistryCacheMtime = fs.statSync(DESKTOP_CONNECTIONS_REGISTRY_PATH).mtimeMs
}

/**
 * Renderer-facing view of a registry entry: token bytes never cross the IPC
 * boundary — the renderer gets a preview + set flag, mirroring
 * sanitizeDesktopConnectionConfig.
 */
function sanitizeRegistryConnection(entry) {
  const { token, headers, ...rest } = entry
  const decrypted = decryptDesktopSecret(token)
  // Last-known stable backend identity (from roster enumeration / Test) so
  // Settings can hint "Same backend as <label>" on connections that are two
  // addresses for one box. Display-only; absent until a probe has seen it.
  const knownInstallId = connectionInstallIds.get(entry.id)?.id

  return {
    ...rest,
    tokenSet: Boolean(decrypted),
    tokenPreview: tokenPreview(decrypted),
    ...(knownInstallId ? { installId: knownInstallId } : {}),
    // Header VALUES are secrets (Cloudflare Access client secrets etc.) and
    // never cross the IPC boundary — the renderer only needs the names to
    // render the edit form.
    headerNames: headers && typeof headers === 'object' ? Object.keys(headers) : []
  }
}

function sanitizeConnectionsRegistry(registry = readDesktopConnectionsRegistry()) {
  // Same keyring signal the v1 sanitize exposes: lets the Connections panel
  // offer the plain-text opt-in on keyring-less Linux instead of failing.
  // Policy-aware: never touches safeStorage while encryption is opted out.
  const secureTokenStorage = probeSecureTokenStorage()

  return {
    version: registry.version,
    primary: registry.primary,
    launchMode: registry.launchMode,
    lastUsed: registry.lastUsed,
    secureTokenStorage,
    connections: registry.connections.map(sanitizeRegistryConnection),
    // Surface quarantined-entry NOTICES only (reason + best-effort label) —
    // the raw entries can carry token envelopes and stay in the file (#94246).
    quarantined: (registry.quarantined || []).map(q => ({
      reason: String(q?.reason || 'unknown'),
      label:
        q && q.entry && typeof q.entry === 'object' && typeof (q.entry as any).label === 'string'
          ? (q.entry as any).label
          : ''
    }))
  }
}

/**
 * Save (create or edit) a registry connection from a renderer payload.
 * Edits merge over the stored entry (mergeConnectionInput) so fields the
 * editor doesn't carry — cloud `org`, ssh `remoteHermesPath`/`remoteProfile` —
 * survive a rename. Token handling mirrors coerceDesktopConnectionConfig: an
 * incoming plaintext token is encrypted (honoring the same allowPlainTextToken
 * opt-in seam as Settings → Gateway); an absent token field inherits the
 * stored envelope on edit; switching auth away from 'token' clears it
 * (normalizeConnectionInput drops tokens on non-token entries).
 */
async function saveRegistryConnection(input: any = {}) {
  const registry = readDesktopConnectionsRegistry()
  const existing = input.id ? registry.connections.find(c => c.id === input.id) : null
  const incomingToken = typeof input.token === 'string' ? input.token.trim() : ''

  const token = resolvePersistedRemoteToken({
    incomingToken,
    persistToken: true,
    existingToken: existing?.token,
    allowPlainText: input.allowPlainTextToken,
    encryptSecret: encryptDesktopSecret
  })

  // Extra gateway headers arrive as plaintext strings from the editor (or
  // envelopes from a hand-edited import). Encrypt plaintext values the same
  // way tokens are stored; a null/empty value drops that header. An absent
  // `headers` field inherits the stored set via mergeConnectionInput.
  const headers =
    input.headers && typeof input.headers === 'object'
      ? encryptIncomingRemoteHeaders(input.headers, existing?.headers, {
          allowPlainText: input.allowPlainTextToken
        })
      : input.headers

  const merged = mergeConnectionInput({ ...input, token, headers }, existing)
  const entry = normalizeConnectionInput(merged, registry)

  // Token-auth remotes must actually have a token to be dialable. OAuth and
  // cloud entries authenticate via cookies/native tokens instead.
  if (entry.kind === 'remote' && entry.authMode !== 'oauth' && !decryptDesktopSecret(entry.token)) {
    throw new Error('Remote gateway session token is required.')
  }

  if (existing && connectionDialFieldsChanged(existing, entry)) {
    managedConnectionUpdateGate.assertCanMutate(entry.id)
  }

  writeDesktopConnectionsRegistry(upsertConnection(registry, entry))

  // A dial-material edit (endpoint/auth/ssh routing — NOT a label rename)
  // leaves pooled backends under `conn:<id>::*` and renderer sockets pointing
  // at the OLD target while the UI shows the new one. Recycle them: stop this
  // connection's pooled backends/tunnels and tell renderers to dispose+redial
  // their secondaries for this connection id.
  if (existing && connectionDialFieldsChanged(existing, entry)) {
    await stopRegistryConnectionBackends(entry.id)
    // The id now names a different machine: its cached roster/identity describe the old one,
    // and a cached ssh inventory is never retried (`shouldRetrySshInventory`).
    evictConnectionCaches(entry.id)
    broadcastConnectionsChanged({ connectionId: entry.id, reason: 'updated' })
  } else {
    // Every OTHER successful save (a brand-new connection, a label rename)
    // must still republish the registry snapshot, or windows that didn't
    // perform the save — and the switcher menu fed by $connectionsRegistry —
    // keep painting the stale list until reload (#95393). 'saved' is a pure
    // registry-refresh signal: no sockets moved, so listeners must not
    // dispose or redial anything for it.
    broadcastConnectionsChanged({ connectionId: entry.id, reason: 'saved' })
  }

  return sanitizeRegistryConnection(entry)
}

  return {
    secretStoragePolicy,
    applySecretStorageEncryption,
    migrateLegacyEncryptedSecretsOnce,
    probeSecureTokenStorage,
    encryptDesktopSecret,
    decryptDesktopSecret,
    decryptRemoteHeaders,
    encryptIncomingRemoteHeaders,
    rememberRemoteWsHeaders,
    headersForRemoteRequest,
    installRemoteHeaderRulesOnSession,
    installRemoteHeaderRules,
    readDesktopConnectionConfig,
    writeDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    writeDesktopConnectionsRegistry,
    sanitizeConnectionsRegistry,
    sanitizeRegistryConnection,
    saveRegistryConnection
  }
}
